"""验证水域面过滤后的路径可达性"""
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

print('Loading data...', flush=True)
from app import load_data, nodes_data, graph_edges, haversine_distance
import networkx as nx

load_data()
print('Data loaded, building graph...', flush=True)

G = nx.Graph()
for nid, attrs in nodes_data.items():
    G.add_node(nid, **attrs)
for (u, v), attrs in graph_edges.items():
    G.add_edge(u, v, weight=attrs.get('weight', 1))

components = list(nx.connected_components(G))
components.sort(key=len, reverse=True)
print(f'Connected components: {len(components)}', flush=True)
print(f'Largest component: {len(components[0])} nodes', flush=True)

if 716 in G and 96 in G:
    if nx.has_path(G, 716, 96):
        path = nx.shortest_path(G, 716, 96)
        print(f'Path 716->96: reachable, length={len(path)}', flush=True)
    else:
        print('Path 716->96: NOT REACHABLE!', flush=True)
else:
    print(f'716 in graph: {716 in G}, 96 in graph: {96 in G}', flush=True)

if (393, 40) in graph_edges:
    print('Edge 393->40: STILL PRESENT (should be filtered)', flush=True)
else:
    print('Edge 393->40: FILTERED (correct)', flush=True)

if (40, 405) in graph_edges:
    print('Edge 40->405: STILL PRESENT', flush=True)
else:
    print('Edge 40->405: FILTERED', flush=True)

print(f'Total edges after filtering: {len(graph_edges)}', flush=True)
