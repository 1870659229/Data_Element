"""
下载广东省 shapefile 并提取水域面数据
用法：python scripts/fetch_water_polygons.py

提供两种下载方式：
  方式1（推荐）：下载 shapefile (302MB)，直接提取水域面
  方式2：下载 PBF (158MB)，用 pyrosm 提取水域面

如果下载慢，可以手动用浏览器/迅雷下载后放到 data_osm/ 目录
"""
import os
import sys
import zipfile
import json
import urllib.request

# ============ 配置 ============
SHP_URL = 'https://download.geofabrik.de/asia/china/guangdong-latest-free.shp.zip'
PBF_URL = 'https://download.geofabrik.de/asia/china/guangdong-latest.osm.pbf'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_osm')
SHP_PATH = os.path.join(DATA_DIR, 'guangdong-latest-free.shp.zip')
PBF_PATH = os.path.join(DATA_DIR, 'guangdong-latest.osm.pbf')
OUT_PATH = os.path.join(DATA_DIR, 'water_polygons.geojson')

# 珠江口区域裁剪范围
BBOX = (113.0, 22.5, 114.0, 23.5)  # (min_lon, min_lat, max_lon, max_lat)


def download_file(url, path, label):
    """通用下载函数"""
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        if size > 50:
            print(f'{label}已存在 ({size:.1f} MB)，跳过下载')
            return True
        else:
            print(f'{label}不完整 ({size:.1f} MB)，重新下载')
            os.remove(path)

    print(f'开始下载: {url}')
    print(f'保存到: {path}')
    print(f'请耐心等待...\n')

    def progress(count, block_size, total_size):
        downloaded = count * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        if count % 500 == 0:
            print(f'  {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({pct:.1f}%)', flush=True)

    try:
        urllib.request.urlretrieve(url, path, reporthook=progress)
        size = os.path.getsize(path) / 1024 / 1024
        print(f'\n下载完成: {size:.1f} MB')
        return True
    except Exception as e:
        print(f'\n下载失败: {e}')
        print('\n请手动下载（推荐用浏览器/迅雷）：')
        print(f'  URL: {url}')
        print(f'  保存到: {path}')
        print('下载完成后重新运行此脚本')
        return False


def extract_from_shp():
    """方式1：从 shapefile 中提取水域面数据"""
    import geopandas as gpd
    from shapely.geometry import box

    print('\n--- 从 shapefile 提取水域面 ---')
    with zipfile.ZipFile(SHP_PATH, 'r') as z:
        names = z.namelist()
        water_shp = [n for n in names if 'water_a' in n and n.endswith('.shp')]
        if not water_shp:
            water_shp = [n for n in names if 'water' in n.lower() and n.endswith('.shp') and '_a_' in n]
        if not water_shp:
            print('未找到水域面 shapefile，可用 .shp 文件:')
            for f in [n for n in names if n.endswith('.shp')]:
                print(f'  {f}')
            return False

        print(f'找到水域面文件: {water_shp[0]}')
        extract_dir = os.path.join(DATA_DIR, '_temp_shp')
        os.makedirs(extract_dir, exist_ok=True)
        base = os.path.splitext(water_shp[0])[0]
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            fname = base + ext
            if fname in names:
                z.extract(fname, extract_dir)

    shp_path = os.path.join(extract_dir, water_shp[0])
    print(f'读取水域面数据...')
    gdf = gpd.read_file(shp_path)
    print(f'  总要素数: {len(gdf)}')

    bbox_geom = box(*BBOX)
    gdf_clipped = gdf[gdf.geometry.intersects(bbox_geom)].copy()
    gdf_clipped = gdf_clipped[gdf_clipped.geometry.notna()]
    print(f'  裁剪到珠江口: {len(gdf_clipped)} 个水域面')

    if len(gdf_clipped) == 0:
        print('裁剪后无数据！')
        return False

    _save_geojson(gdf_clipped)

    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)
    return True


def extract_from_pbf():
    """方式2：从 PBF 中提取水域面数据"""
    from shapely.geometry import box, Polygon, MultiPolygon
    import pyrosm

    print('\n--- 从 PBF 提取水域面 ---')
    print('读取 PBF 文件（需要几分钟）...')
    osm = pyrosm.OSM(PBF_PATH)

    # 获取水域面数据 (natural=water, waterway=*, landuse=reservoir 等)
    print('提取水域面...')
    try:
        # pyrosm 的 get_natural() 包含水域面
        natural = osm.get_natural()
        if natural is not None:
            water = natural[natural['natural'].isin(['water', 'wetland', 'bay', 'strait'])]
            print(f'  natural=water: {len(water)} 个')
        else:
            water = None
            print('  natural 数据为空')

        # 也获取 landuse=reservoir
        landuse = osm.get_landuse()
        if landuse is not None:
            reservoir = landuse[landuse['landuse'] == 'reservoir']
            print(f'  landuse=reservoir: {len(reservoir)} 个')
        else:
            reservoir = None

        # 合并
        import geopandas as gpd
        frames = []
        if water is not None and len(water) > 0:
            frames.append(water)
        if reservoir is not None and len(reservoir) > 0:
            frames.append(reservoir)
        if not frames:
            print('未找到水域面数据！')
            return False

        gdf = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=frames[0].crs)
        print(f'  合并后: {len(gdf)} 个水域面')

        # 只保留面数据
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
        print(f'  面数据: {len(gdf)} 个')

        # 裁剪
        bbox_geom = box(*BBOX)
        gdf_clipped = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        print(f'  裁剪到珠江口: {len(gdf_clipped)} 个水域面')

        if len(gdf_clipped) == 0:
            print('裁剪后无数据！')
            return False

        _save_geojson(gdf_clipped)
        return True

    except Exception as e:
        print(f'提取失败: {e}')
        import traceback
        traceback.print_exc()
        return False


def _save_geojson(gdf):
    """保存 GeoJSON"""
    print(f'\n保存到: {OUT_PATH}')
    geojson = json.loads(gdf.to_json())
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)
    size = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f'保存完成: {size:.1f} MB, {len(geojson["features"])} 个水域面')


def verify():
    """验证水域面数据"""
    from shapely.geometry import Point, shape

    with open(OUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f'\n--- 验证水域面数据 ---')
    print(f'  要素数: {len(data["features"])}')

    # 检查可疑边采样点是否在水域内
    # 节点 393: (23.1138, 113.2567), 节点 40: (23.1113, 113.2261)
    test_points = [
        ('393->40 中点', 23.1126, 113.2414),
        ('393->40 1/4', 23.1132, 113.2494),
        ('393->40 3/4', 23.1120, 113.2334),
        ('节点393', 23.1138, 113.2567),
        ('节点40', 23.1113, 113.2261),
    ]

    water_geoms = []
    for feat in data['features']:
        geom = shape(feat['geometry'])
        if geom.geom_type == 'Polygon':
            water_geoms.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            water_geoms.extend(geom.geoms)

    print(f'  水域面几何数: {len(water_geoms)}')

    for name, lat, lon in test_points:
        pt = Point(lon, lat)
        in_water = any(g.contains(pt) for g in water_geoms)
        print(f'  {name} ({lat:.4f}, {lon:.4f}): {"在水域内" if in_water else "在陆地上"}')


if __name__ == '__main__':
    print('=' * 60)
    print('广东省水域面数据下载与提取工具')
    print('=' * 60)

    # 优先使用 shapefile（数据更完整）
    use_shp = os.path.exists(SHP_PATH) or not os.path.exists(PBF_PATH)

    if use_shp:
        print('\n[方式1] 使用 shapefile (302MB)')
        if not os.path.exists(SHP_PATH):
            if not download_file(SHP_URL, SHP_PATH, 'shapefile'):
                # shapefile 下载失败，尝试 PBF
                print('\nshapefile 下载失败，尝试 PBF 方式...')
                use_shp = False
        if use_shp:
            if extract_from_shp():
                verify()
                print('\n完成！水域面数据已保存到:', OUT_PATH)
            else:
                print('\nshapefile 提取失败')
                sys.exit(1)

    if not use_shp:
        print('\n[方式2] 使用 PBF (158MB)')
        if not os.path.exists(PBF_PATH):
            if not download_file(PBF_URL, PBF_PATH, 'PBF'):
                sys.exit(1)
        if extract_from_pbf():
            verify()
            print('\n完成！水域面数据已保存到:', OUT_PATH)
        else:
            print('\nPBF 提取失败')
            sys.exit(1)

    print('\n请重启 app.py 以加载新数据')
