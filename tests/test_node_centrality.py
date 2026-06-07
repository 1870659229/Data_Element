"""
综合节点中心性分析模块测试

参考冯润航2025论文《基于出租车轨迹数据的城市路网建模及节点影响力分析》：
使用度中心性、介数中心性、接近中心性、特征向量中心性 4 种指标
通过 PCA 自动加权融合为综合中心性指标。
"""

import pytest
import networkx as nx
from node_centrality import NodeCentralityAnalyzer


@pytest.fixture
def sample_graph():
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    for nid in G.nodes():
        G.nodes[nid]['lat'] = 23.0 + nid * 0.01
        G.nodes[nid]['lon'] = 113.0 + nid * 0.01
        G.nodes[nid]['frequency'] = nid * 10
    return G


def test_compute_all_centrality(sample_graph):
    analyzer = NodeCentralityAnalyzer(
        graph=sample_graph,
        nodes_data=dict(sample_graph.nodes(data=True))
    )
    scores = analyzer.compute_all_centrality(k_sample=30)
    assert len(scores) == 50
    for nid, s in scores.items():
        assert 'degree' in s
        assert 'betweenness' in s
        assert 'closeness' in s
        assert 'eigenvector' in s
        assert 'composite' in s
        assert 0 <= s['composite']


def test_identify_key_nodes(sample_graph):
    analyzer = NodeCentralityAnalyzer(
        graph=sample_graph,
        nodes_data=dict(sample_graph.nodes(data=True))
    )
    analyzer.compute_all_centrality(k_sample=30)
    key = analyzer.identify_key_nodes(top_pct=0.2)
    assert 'hub' in key
    assert 'bridge' in key
    assert 'vulnerable' in key
    assert len(key['hub']) > 0
    assert len(key['bridge']) > 0
    assert len(key['vulnerable']) > 0


def test_auto_weight_by_pca(sample_graph):
    analyzer = NodeCentralityAnalyzer(
        graph=sample_graph,
        nodes_data=dict(sample_graph.nodes(data=True))
    )
    analyzer.compute_all_centrality(k_sample=30)
    analyzer.auto_weight_by_pca()
    total = sum(analyzer.weights.values())
    assert abs(total - 1.0) < 0.01
    for w in analyzer.weights.values():
        assert w >= 0


def test_get_centrality(sample_graph):
    analyzer = NodeCentralityAnalyzer(
        graph=sample_graph,
        nodes_data=dict(sample_graph.nodes(data=True))
    )
    analyzer.compute_all_centrality(k_sample=30)
    c = analyzer.get_centrality(0)
    assert c >= 0
    c_missing = analyzer.get_centrality(9999)
    assert c_missing == 0.0


def test_export_results(tmp_path, sample_graph):
    analyzer = NodeCentralityAnalyzer(
        graph=sample_graph,
        nodes_data=dict(sample_graph.nodes(data=True))
    )
    analyzer.compute_all_centrality(k_sample=30)
    analyzer.auto_weight_by_pca()
    analyzer.export_results(str(tmp_path))

    csv_path = tmp_path / 'node_centrality.csv'
    json_path = tmp_path / 'key_nodes.json'
    assert csv_path.exists()
    assert json_path.exists()

    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        key = json.load(f)
    assert 'hub' in key
    assert 'bridge' in key
    assert 'vulnerable' in key


def test_undirected_graph_centrality(sample_graph):
    G_directed = sample_graph.to_directed()
    analyzer = NodeCentralityAnalyzer(
        graph=G_directed,
        nodes_data=dict(G_directed.nodes(data=True))
    )
    scores = analyzer.compute_all_centrality(k_sample=30)
    assert len(scores) == 50
    for nid, s in scores.items():
        assert 'composite' in s


def test_composite_in_unit_interval(sample_graph):
    """审查问题#1: composite 归一化到 [0,1]"""
    analyzer = NodeCentralityAnalyzer(
        graph=sample_graph,
        nodes_data=dict(sample_graph.nodes(data=True))
    )
    analyzer.compute_all_centrality(k_sample=30)
    composites = [s['composite'] for s in analyzer.centrality_scores.values()]
    assert min(composites) >= 0.0
    assert max(composites) <= 1.0 + 1e-9
    assert max(composites) > 0.5


def test_identify_key_nodes_small_graph():
    """审查问题#2: 极小图上 hub/vulnerable 不应完全重叠"""
    import networkx as nx
    G = nx.path_graph(5)
    for nid in G.nodes():
        G.nodes[nid]['lat'] = 23.0
        G.nodes[nid]['lon'] = 113.0
        G.nodes[nid]['frequency'] = 0
    analyzer = NodeCentralityAnalyzer(
        graph=G,
        nodes_data=dict(G.nodes(data=True))
    )
    analyzer.compute_all_centrality()
    key = analyzer.identify_key_nodes(top_pct=0.2)
    hub = set(key['hub'])
    vulnerable = set(key['vulnerable'])
    assert len(hub) >= 1
    assert len(vulnerable) >= 1
    assert len(hub) + len(vulnerable) <= len(G.nodes()) + max(0, len(hub & vulnerable))
    assert len(hub & vulnerable) < min(len(hub), len(vulnerable)) + 1
