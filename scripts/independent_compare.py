"""验证连通性恢复策略"""
import subprocess, sys

script = '''
import sys
sys.path.insert(0, '.')
import app
app.load_data()
import networkx as nx
G = nx.Graph()
for nid, attrs in app.nodes_data.items():
    G.add_node(nid, **attrs)
for (u, v), attrs in app.graph_edges.items():
    G.add_edge(u, v, weight=attrs.get('weight', 1))
c = list(nx.connected_components(G))
c.sort(key=len, reverse=True)
with open('scripts/_result_restore.txt', 'w', encoding='utf-8') as f:
    f.write(f'edges={len(app.graph_edges)}\\n')
    f.write(f'main_comp={len(c[0])}\\n')
    f.write(f'comp_sizes={[len(x) for x in c[:5]]}\\n')
    f.write(f'isolated={sum(1 for n in G.nodes() if G.degree(n) == 0)}\\n')
    for key in [(393, 40), (40, 405)]:
        f.write(f'edge_{key[0]}_{key[1]}={"PRESENT" if key in app.graph_edges else "FILTERED"}\\n')
    for src, dst in [(716, 96), (393, 96)]:
        if nx.has_path(G, src, dst):
            path = nx.shortest_path(G, src, dst)
            f.write(f'path_{src}_{dst}=reachable_len{len(path)}\\n')
        else:
            f.write(f'path_{src}_{dst}=NOT_REACHABLE\\n')
    f.write(f'node40_degree={G.degree(40) if 40 in G else "N/A"}\\n')
print('Done')
'''
with open('scripts/_run_verify.py', 'w') as f:
    f.write(script)

subprocess.run([sys.executable, 'scripts/_run_verify.py'], cwd='d:/py_project/Data_Element')

with open('scripts/_result_restore.txt') as f:
    print(f.read())
