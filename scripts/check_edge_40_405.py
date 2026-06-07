"""检查 40->405 边是否穿陆地"""
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from app import load_data, nodes_data, haversine_distance, _is_point_in_water, _is_edge_near_waterway

load_data()

n40 = nodes_data.get(40)
n405 = nodes_data.get(405)
print(f'Node 40: {n40}', flush=True)
print(f'Node 405: {n405}', flush=True)

d = haversine_distance(n40['lat'], n40['lon'], n405['lat'], n405['lon'])
print(f'Distance: {d:.0f}m', flush=True)

land = 0
for i in range(10):
    t = i / 9
    lat = n40['lat'] + t * (n405['lat'] - n40['lat'])
    lon = n40['lon'] + t * (n405['lon'] - n40['lon'])
    in_water = _is_point_in_water(lat, lon)
    if not in_water:
        land += 1
    print(f'  Sample {i}: ({lat:.4f}, {lon:.4f}) -> {"WATER" if in_water else "LAND"}', flush=True)

print(f'Land samples: {land}/10', flush=True)
near = _is_edge_near_waterway(40, 405)
print(f'_is_edge_near_waterway(40, 405): {near}', flush=True)
