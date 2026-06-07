"""用河道线 buffer 生成水域面，并检测穿陆地边"""
import json
import sys
sys.path.insert(0, '.')
from shapely.geometry import Point, LineString, shape, MultiPolygon, Polygon
from shapely.ops import unary_union
import math

# 1. 加载河道线数据
with open('data_osm/waterways.geojson', 'r', encoding='utf-8') as f:
    waterways = json.load(f)

print(f"waterways.geojson: {len(waterways['features'])} features")

# 2. 提取所有河道线几何
line_geoms = []
for feat in waterways['features']:
    geom = shape(feat['geometry'])
    if geom.geom_type == 'LineString':
        line_geoms.append(geom)
    elif geom.geom_type == 'MultiLineString':
        for g in geom.geoms:
            line_geoms.append(g)

print(f"河道线数量: {len(line_geoms)}")

# 3. 珠江口核心区域裁剪
# 只保留路径规划涉及的区域
bbox = Polygon([
    (113.0, 22.5), (114.0, 22.5), (114.0, 23.5), (113.0, 23.5), (113.0, 22.5)
])
print(f"裁剪范围: lon 113.0~114.0, lat 22.5~23.5")

# 裁剪到 bbox 范围内的线段
clipped = []
for g in line_geoms:
    if not g.intersects(bbox):
        continue
    try:
        intersection = g.intersection(bbox)
        if intersection.is_empty:
            continue
        if intersection.geom_type == 'LineString':
            clipped.append(intersection)
        elif intersection.geom_type == 'MultiLineString':
            for sub in intersection.geoms:
                clipped.append(sub)
    except Exception:
        continue
print(f"珠江口区域河道线: {len(clipped)}")

# 4. Buffer 生成水域面 (500m ≈ 0.0045度)
BUFFER_DEG = 0.0045  # ~500m
print(f"Buffer 宽度: {BUFFER_DEG*111000:.0f}m")

# 分批 buffer 避免内存溢出
water_polygons = []
batch_size = 500
for i in range(0, len(clipped), batch_size):
    batch = clipped[i:i+batch_size]
    buffered = [g.buffer(BUFFER_DEG) for g in batch]
    union = unary_union(buffered)
    if union.geom_type == 'Polygon':
        water_polygons.append(union)
    elif union.geom_type == 'MultiPolygon':
        water_polygons.extend(union.geoms)

print(f"水域面数量: {len(water_polygons)}")

# 5. 合并所有水域面
water_union = unary_union(water_polygons)
if water_union.geom_type == 'Polygon':
    water_union = MultiPolygon([water_union])
print(f"合并后水域面: {len(water_union.geoms)} polygons")

# 6. 保存水域面数据
water_feature = {
    "type": "Feature",
    "properties": {"source": "waterways_buffer_500m"},
    "geometry": water_union.__geo_interface__
}

water_geojson = {
    "type": "FeatureCollection",
    "features": [water_feature]
}

out_path = 'data_osm/water_polygons_buffer500m.geojson'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(water_geojson, f)
print(f"\n已保存: {out_path}")

# 7. 测试：检测穿陆地边
import pandas as pd
nodes_df = pd.read_csv('output/topology_nodes.csv')
edges_df = pd.read_csv('output/topology_edges.csv')

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * math.asin(math.sqrt(a)) * 6371

# 检查可疑边
suspicious_edges = [(393, 40), (40, 405), (393, 405)]
print("\n穿陆地检测 (采样点不在水域面内的比例):")
for u, v in suspicious_edges:
    n1 = nodes_df[nodes_df['node_id']==u].iloc[0]
    n2 = nodes_df[nodes_df['node_id']==v].iloc[0]
    dist = haversine(n1['lon'], n1['lat'], n2['lon'], n2['lat'])
    
    # 沿边采样 10 个点
    on_land = 0
    sample_count = 10
    for i in range(sample_count):
        t = i / (sample_count - 1)
        lat = n1['lat'] + t * (n2['lat'] - n1['lat'])
        lon = n1['lon'] + t * (n2['lon'] - n1['lon'])
        pt = Point(lon, lat)
        if not water_union.contains(pt):
            on_land += 1
    
    land_pct = on_land / sample_count * 100
    print(f"  {u}->{v}: {dist:.2f}km, {on_land}/{sample_count} 点在陆地上 ({land_pct:.0f}%)")

# 检查正常边
normal_edges = [(716, 393), (405, 164)]
print("\n正常边检测:")
for u, v in normal_edges:
    n1 = nodes_df[nodes_df['node_id']==u].iloc[0]
    n2 = nodes_df[nodes_df['node_id']==v].iloc[0]
    dist = haversine(n1['lon'], n1['lat'], n2['lon'], n2['lat'])
    
    on_land = 0
    sample_count = 10
    for i in range(sample_count):
        t = i / (sample_count - 1)
        lat = n1['lat'] + t * (n2['lat'] - n1['lat'])
        lon = n1['lon'] + t * (n2['lon'] - n1['lon'])
        pt = Point(lon, lat)
        if not water_union.contains(pt):
            on_land += 1
    
    land_pct = on_land / sample_count * 100
    print(f"  {u}->{v}: {dist:.2f}km, {on_land}/{sample_count} 点在陆地上 ({land_pct:.0f}%)")
