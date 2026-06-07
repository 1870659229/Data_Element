"""从已有 shapefile 提取珠江口核心水域面数据（只保留大面积水域）"""
import zipfile, os, json, tempfile, shutil
import geopandas as gpd
from shapely.geometry import box, Point, shape

SHP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_osm', 'guangdong-260606-free.shp.zip')
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_osm', 'water_polygons.geojson')
# 更精确的珠江口区域
BBOX = (113.0, 22.5, 114.0, 23.5)

tmpdir = tempfile.mkdtemp()
with zipfile.ZipFile(SHP_PATH, 'r') as z:
    for name in z.namelist():
        if 'water_a' in name:
            z.extract(name, tmpdir)
            print(f'Extracted: {name}')

shp_file = os.path.join(tmpdir, 'gis_osm_water_a_free_1.shp')
print(f'Reading {shp_file}...')
gdf = gpd.read_file(shp_file)
print(f'Total features: {len(gdf)}')

bbox_geom = box(*BBOX)
gdf_clipped = gdf[gdf.geometry.intersects(bbox_geom)].copy()
gdf_clipped = gdf_clipped[gdf_clipped.geometry.notna()]
print(f'Clipped to Pearl River: {len(gdf_clipped)} features')

# 保留面积 >= 5000 平方米的水域面（约0.005km²）
# 降低阈值以保留窄航道面数据，减少误判
MIN_AREA = 5000  # 平方米（WGS84坐标系下约 5e-6 度²）
gdf_clipped = gdf_clipped[gdf_clipped.geometry.area >= MIN_AREA / (111000 * 111000)]
print(f'After area filter (>={MIN_AREA}m2): {len(gdf_clipped)} features')

geojson = json.loads(gdf_clipped.to_json())
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(geojson, f)
size = os.path.getsize(OUT_PATH) / 1024 / 1024
nfeat = len(geojson['features'])
print(f'Saved: {size:.1f} MB, {nfeat} features')

# Verify
water_geoms = []
for feat in geojson['features']:
    geom = shape(feat['geometry'])
    if geom.geom_type == 'Polygon':
        water_geoms.append(geom)
    elif geom.geom_type == 'MultiPolygon':
        water_geoms.extend(geom.geoms)
print(f'Water polygon geometries: {len(water_geoms)}')

test_points = [
    ('393->40 mid', 113.2414, 23.1126),
    ('393->40 1/4', 113.2494, 23.1132),
    ('393->40 3/4', 113.2334, 23.1120),
    ('Node 393', 113.2567, 23.1138),
    ('Node 40', 113.2261, 23.1113),
]
for name, lon, lat in test_points:
    pt = Point(lon, lat)
    in_water = any(g.contains(pt) for g in water_geoms)
    status = 'IN WATER' if in_water else 'ON LAND'
    print(f'  {name} ({lat:.4f}, {lon:.4f}): {status}')

shutil.rmtree(tmpdir, ignore_errors=True)
print('Done!')
