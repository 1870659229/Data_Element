"""找出被连续3+陆地策略额外过滤的边，并检查其连续陆地情况"""
import sys
sys.path.insert(0, '.')

import app
import csv

app._load_waterways()

nodes_data = {}
with open('output/topology_nodes.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        nid = int(row['node_id'])
        nodes_data[nid] = {'lat': float(row['lat']), 'lon': float(row['lon'])}
app.nodes_data.update(nodes_data)

raw_edges = []
with open('output/topology_edges.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        u, v = int(row['from_node']), int(row['to_node'])
        raw_edges.append((u, v))

extra = []
for (u, v) in raw_edges:
    n1 = nodes_data.get(u)
    n2 = nodes_data.get(v)
    if not n1 or not n2:
        continue
    d = app.haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])
    if d < app.LONG_EDGE_STRICT_THRESHOLD_M:
        continue

    # Test WITH polygon
    result_with = app._is_edge_near_waterway(u, v)

    # Test WITHOUT polygon
    saved = app._water_polygon_geoms
    saved_grid = app._water_polygon_grid
    app._water_polygon_geoms = []
    app._water_polygon_grid = {}
    result_without = app._is_edge_near_waterway(u, v)
    app._water_polygon_geoms = saved
    app._water_polygon_grid = saved_grid

    if result_without and not result_with:
        # Compute consecutive land
        land_samples = []
        max_consecutive = 0
        consecutive = 0
        total_land = 0
        for i in range(10):
            t = i / 9
            lat = n1['lat'] + t * (n2['lat'] - n1['lat'])
            lon = n1['lon'] + t * (n2['lon'] - n1['lon'])
            in_water = app._is_point_in_water(lat, lon)
            if not in_water:
                land_samples.append(i)
                total_land += 1
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        extra.append((u, v, d, total_land, max_consecutive, land_samples))

with open('scripts/_extra_consecutive.txt', 'w', encoding='utf-8') as f:
    f.write(f'Edges additionally filtered by consecutive-3 strategy: {len(extra)}\n\n')
    for u, v, d, land, maxc, samples in sorted(extra, key=lambda x: x[4]):
        f.write(f'  {u}->{v}: {d:.0f}m, land={land}/10, max_consecutive={maxc}, land_indices={samples}\n')

print('Done')
