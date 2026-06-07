"""
通过 Overpass API 下载珠江口水域面数据
只查询核心区域，数据量小，应该不会超时
用法：python scripts/fetch_water_overpass.py
"""
import os
import json
import urllib.request
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_osm')
OUT_PATH = os.path.join(DATA_DIR, 'water_polygons.geojson')

# 珠江口核心区域（比之前小很多）
BBOX = '22.8,113.0,23.3,113.8'  # south,west,north,east

# Overpass API 服务器（选一个快的）
OVERPASS_SERVERS = [
    'https://overpass-api.de/api/interpreter',
    'https://lz4.overpass-api.de/api/interpreter',
    'https://z.overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
]

# 查询水域面：natural=water, water=*, waterway=riverbank, landuse=reservoir
QUERY = f"""
[out:json][timeout:120];
(
  way["natural"="water"]({BBOX});
  way["water"]({BBOX});
  way["waterway"="riverbank"]({BBOX});
  way["landuse"="reservoir"]({BBOX});
  relation["natural"="water"]({BBOX});
  relation["water"]({BBOX});
);
out body;
>;
out skel qt;
"""


def download_overpass():
    """从 Overpass API 下载水域面数据"""
    print(f'查询范围: {BBOX}')
    print(f'查询内容: natural=water, water=*, waterway=riverbank, landuse=reservoir')

    for server in OVERPASS_SERVERS:
        print(f'\n尝试服务器: {server}')
        try:
            req = urllib.request.Request(
                server,
                data=QUERY.encode('utf-8'),
                headers={'User-Agent': 'DataElement/1.0'}
            )
            start = time.time()
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - start
            print(f'  下载成功! 耗时 {elapsed:.1f}s, {len(data["elements"])} 个元素')
            return data
        except Exception as e:
            print(f'  失败: {e}')
            continue

    print('\n所有服务器都失败了')
    return None


def osm_to_geojson(osm_data):
    """将 OSM 数据转换为 GeoJSON 多边形"""
    from shapely.geometry import Polygon, MultiPolygon, LineString
    from shapely.ops import polygonize, linemerge

    # 构建节点索引
    nodes = {}
    for el in osm_data['elements']:
        if el['type'] == 'node':
            nodes[el['id']] = (el['lon'], el['lat'])

    # 提取 way
    ways = {}
    for el in osm_data['elements']:
        if el['type'] == 'way':
            ways[el['id']] = el

    # 构建 Polygon
    polygons = []
    for el in osm_data['elements']:
        if el['type'] == 'way' and 'tags' in el:
            tags = el['tags']
            if not any(k in tags for k in ['natural', 'water', 'waterway', 'landuse']):
                continue

            coords = []
            valid = True
            for nid in el.get('nodes', []):
                if nid in nodes:
                    coords.append(nodes[nid])
                else:
                    valid = False
                    break

            if not valid or len(coords) < 4:
                continue

            # 闭合检查
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    polygons.append((poly, tags))
            except Exception:
                continue

    # 处理 relation (multipolygon)
    relations = []
    for el in osm_data['elements']:
        if el['type'] == 'relation' and 'tags' in el:
            tags = el['tags']
            if tags.get('type') == 'multipolygon' and any(
                k in tags for k in ['natural', 'water', 'waterway', 'landuse']
            ):
                relations.append(el)

    # 构建 relation 的多边形
    for rel in relations:
        outer_ways = []
        inner_ways = []
        for member in rel.get('members', []):
            if member['type'] == 'way':
                way = ways.get(member['ref'])
                if way:
                    coords = []
                    valid = True
                    for nid in way.get('nodes', []):
                        if nid in nodes:
                            coords.append(nodes[nid])
                        else:
                            valid = False
                            break
                    if valid and len(coords) >= 2:
                        if member.get('role') == 'outer':
                            outer_ways.append(coords)
                        elif member.get('role') == 'inner':
                            inner_ways.append(coords)

        # 合并 outer rings
        if outer_ways:
            try:
                lines = [LineString(c) for c in outer_ways if len(c) >= 2]
                if lines:
                    merged = linemerge(lines)
                    from shapely.ops import polygonize
                    for poly in polygonize(merged):
                        if poly.is_valid and poly.area > 0:
                            # 减去 inner rings
                            for inner in inner_ways:
                                try:
                                    inner_line = LineString(inner)
                                    for inner_poly in polygonize(inner_line):
                                        poly = poly.difference(inner_poly)
                                except Exception:
                                    pass
                            if hasattr(poly, 'is_valid') and poly.is_valid:
                                polygons.append((poly, rel['tags']))
            except Exception as e:
                print(f'  relation {rel["id"]} 处理失败: {e}')

    print(f'\n生成 {len(polygons)} 个水域面')

    # 构建 GeoJSON
    features = []
    for i, (geom, tags) in enumerate(polygons):
        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "fclass": tags.get('natural', tags.get('waterway', tags.get('landuse', tags.get('water', 'unknown')))),
                "name": tags.get('name', ''),
            },
            "geometry": geom.__geo_interface__
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return geojson


def verify(geojson):
    """验证水域面数据"""
    from shapely.geometry import Point, shape

    print(f'\n--- 验证水域面数据 ---')
    print(f'  要素数: {len(geojson["features"])}')

    test_points = [
        ('393->40 中点', 23.1126, 113.2414),
        ('393->40 1/4', 23.1132, 113.2494),
        ('393->40 3/4', 23.1120, 113.2334),
        ('节点393', 23.1138, 113.2567),
        ('节点40', 23.1113, 113.2261),
    ]

    water_geoms = []
    for feat in geojson['features']:
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
    print('Overpass API 珠江口水域面数据下载')
    print('=' * 60)

    osm_data = download_overpass()
    if not osm_data:
        print('\n下载失败！请尝试:')
        print('  1. 用迅雷/IDM下载 shapefile:')
        print('     https://download.geofabrik.de/asia/china/guangdong-latest-free.shp.zip')
        print('  2. 然后运行: python scripts/fetch_water_polygons.py')
        exit(1)

    geojson = osm_to_geojson(osm_data)

    print(f'\n保存到: {OUT_PATH}')
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)
    size = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f'保存完成: {size:.1f} MB')

    verify(geojson)

    print('\n完成！请重启 app.py 以加载新数据')
