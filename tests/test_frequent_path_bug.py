# -*- coding: utf-8 -*-
"""
测试"频次优先"路径是否真的被生成

Bug描述(从 benchmark_frequent.py 实测数据)：
- 0/10 船型的 alternative_paths 中包含'频次优先'路径
- 实际跑出来的导航结果 total_frequency 字段全为 0
- _bidirectional_a_star_frequent 是死代码，从未被 find_paths() 调用
- adra_star_planner._theta_shortcut_cost 的 'frequent' 分支返回常数 50.0

此测试将证明这些 bug 的存在，并作为修复后的回归保护。
"""

import pytest
import networkx as nx
from ship_navigator import (
    ShipCharacteristics,
    PhysicalConstraintChecker,
    MultiObjectiveNavigator,
    PathType,
)


def _make_skewed_frequency_graph():
    """
    构造一个具有非均匀频次的图：
    - 主路径 (0→1→2→3) 频次低 (3, 5, 3) → 长但不通航
    - 备用路径 (0→10→20→3) 频次高 (50, 80, 50) → 短且习惯
    - 备用路径 (0→100→200→3) 频次中等 (10, 10, 10) → 短
    """
    G = nx.DiGraph()
    coords = {
        0: (23.0, 113.0),
        1: (23.1, 113.1),
        2: (23.2, 113.2),
        3: (23.3, 113.3),
        10: (23.05, 113.0),
        20: (23.15, 113.05),
        100: (23.05, 113.15),
        200: (23.15, 113.15),
    }
    for nid, (lat, lon) in coords.items():
        G.add_node(nid, lat=lat, lon=lon)

    # 主路径 (低频)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    # 备用1 (高频 - 习惯航线)
    G.add_edge(0, 10)
    G.add_edge(10, 20)
    G.add_edge(20, 3)
    # 备用2 (中频)
    G.add_edge(0, 100)
    G.add_edge(100, 200)
    G.add_edge(200, 3)

    edge_features = {}
    # 低频
    for u, v, c in [(0, 1, 3), (1, 2, 5), (2, 3, 3)]:
        edge_features[(u, v)] = {
            'segment_count': c, 'avg_distance': 15000, 'avg_travel_time': 50,
            'waterway_type': 'open', 'risk_score': 0.1
        }
    # 高频
    for u, v, c in [(0, 10, 50), (10, 20, 80), (20, 3, 50)]:
        edge_features[(u, v)] = {
            'segment_count': c, 'avg_distance': 6000, 'avg_travel_time': 20,
            'waterway_type': 'open', 'risk_score': 0.1
        }
    # 中频
    for u, v, c in [(0, 100, 10), (100, 200, 10), (200, 3, 10)]:
        edge_features[(u, v)] = {
            'segment_count': c, 'avg_distance': 6000, 'avg_travel_time': 20,
            'waterway_type': 'open', 'risk_score': 0.1
        }

    return G, edge_features


def _make_ship():
    return ShipCharacteristics(
        ship_name="test_ship",
        length=100.0,
        width=15.0,
        draft=5.0,
        height=20.0,
        tonnage=5000.0,
        ship_type="货船",
        max_speed=15.0,
    )


def _make_navigator():
    G, edge_features = _make_skewed_frequency_graph()
    ship = _make_ship()
    nodes = dict(G.nodes(data=True))
    cc = PhysicalConstraintChecker(edge_features, nodes, G)
    navigator = MultiObjectiveNavigator(G, edge_features, cc)
    return navigator, ship, edge_features, G


def test_frequent_path_exists_in_find_paths():
    """
    BUG: find_paths() 的结果中应包含'频次优先'类型的路径
    """
    navigator, ship, edge_features, G = _make_navigator()
    paths = navigator.find_paths(start=0, end=3, ship=ship, hour=12, max_paths=3)

    assert len(paths) >= 2, f"应至少2条路径, 实际{len(paths)}条"

    path_types = [p.path_type for p in paths]
    print(f"  生成的路径类型: {path_types}")
    assert PathType.FREQUENT in path_types, \
        f"应包含'频次优先'路径, 实际类型: {path_types}"


def test_frequent_path_prefers_high_frequency_edges():
    """
    BUG: '频次优先'路径应优先走高频段
    期望: 路径经过备用1(0→10→20→3), total_frequency=180
    而非: 主路径(0→1→2→3), total_frequency=11
    """
    navigator, ship, edge_features, G = _make_navigator()
    paths = navigator.find_paths(start=0, end=3, ship=ship, hour=12, max_paths=3)

    frequent_paths = [p for p in paths if p.path_type == PathType.FREQUENT]
    assert len(frequent_paths) >= 1, "必须有频次优先路径"

    frequent_path = frequent_paths[0]
    total_freq = sum(
        edge_features.get((frequent_path.nodes[i], frequent_path.nodes[i+1]), {}).get('segment_count', 0)
        for i in range(len(frequent_path.nodes) - 1)
    )
    print(f"  频次优先路径节点: {frequent_path.nodes}")
    print(f"  路径总频次: {total_freq}")
    # 高频段总频次=180, 中频段=30, 低频段=11
    # 至少应>50才合理
    assert total_freq >= 50, \
        f"频次优先路径频次太低: {total_freq} (应>=50, 表明走了高频段)"


def test_frequent_path_is_different_from_safest():
    """
    BUG: '频次优先'路径应与'安全优先'路径不同
    (如果相同, 说明只是改了label)
    """
    navigator, ship, edge_features, G = _make_navigator()
    paths = navigator.find_paths(start=0, end=3, ship=ship, hour=12, max_paths=3)

    frequent = next((p for p in paths if p.path_type == PathType.FREQUENT), None)
    safest = next((p for p in paths if p.path_type == PathType.SAFEST), None)

    assert frequent is not None, "无频次优先路径"
    assert safest is not None, "无安全优先路径"
    print(f"  安全优先节点: {safest.nodes}")
    print(f"  频次优先节点: {frequent.nodes}")
    assert frequent.nodes != safest.nodes, \
        "频次优先与安全优先路径相同 (BUG: 仅仅是改了label)"


def test_bidirectional_a_star_frequent_returns_valid_path():
    """
    _bidirectional_a_star_frequent 实际可调用, 应返回有效路径
    """
    navigator, ship, edge_features, G = _make_navigator()
    result = navigator._bidirectional_a_star_frequent(0, 3, ship, blocked_edges=set(), hour=12)
    print(f"  _bidirectional_a_star_frequent 返回: {'有路径' if result else 'None'}")
    assert result is not None, "_bidirectional_a_star_frequent 应返回路径"
    assert len(result.nodes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
