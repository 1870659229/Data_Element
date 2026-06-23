# -*- coding: utf-8 -*-
"""
回归测试：约束放宽路径兜底机制的契约。

核心契约：当安全优先路径因船舶吃水/限高/限宽不可达时，
find_paths 必须返回 path_type == PathType.RELAXED 的兜底路径。

设计约定：
- find_paths() 第 1685 行会自动调 get_blocked_edges(ship) 填充 blocked_edges，
  即使调用方不传，也会基于船-航道兼容性算出"不可达边集合"。
- 当 _dijkstra_safest 返回 None（即无安全路径）时，应走"约束放宽"分支。

历史背景：
- 早期曾担心 ship_navigator.py:1726 的 `if blocked_edges:` 守卫会让
  正常使用（船型正常时 blocked_edges 为空）永远走不到 RELAXED 分支。
- 验证：find_paths 会用 ship 自身兼容性自动填充 blocked_edges，
  所以这条分支在物理约束真实存在时确实会触发。
- 本测试不构成 RED 测试（当前代码已满足契约），是防回归。
"""

import pytest
import networkx as nx
from ship_navigator import (
    ShipCharacteristics,
    PhysicalConstraintChecker,
    MultiObjectiveNavigator,
    PathType,
)


def _make_inaccessible_graph():
    """
    构造一个对"大船"完全不可达的图：
    - 节点 0 -> 1 边
    - 边的水深/宽/高都低于船舶，吃水/宽度/高度三重不通过
    """
    G = nx.DiGraph()
    G.add_node(0, lat=23.0, lon=113.0)
    G.add_node(1, lat=23.1, lon=113.1)
    G.add_edge(0, 1)

    edge_features = {
        (0, 1): {
            'segment_count': 5, 'avg_distance': 10000, 'avg_travel_time': 30,
            'waterway_type': 'shallow', 'risk_score': 0.5
        }
    }
    # 关键约束：水深 2m、宽 5m、高 3m —— 大船全都过不去
    depth_map = {(0, 1): 2.0}
    width_map = {(0, 1): 5.0}
    height_map = {(0, 1): 3.0}

    return G, edge_features, depth_map, width_map, height_map


def _make_huge_ship():
    return ShipCharacteristics(
        ship_name="huge_ship",
        length=200.0, width=20.0, draft=8.0,  # 吃水 8 > 水深 2
        height=15.0, tonnage=20000.0, ship_type="大型油轮", max_speed=12.0,
    )


def test_relaxed_path_fallback_when_safe_path_unreachable():
    """
    严格物理约束下无安全路径时，规划器必须返回"约束放宽路径"作为兜底。
    调用方不传 blocked_edges 也应触发。
    """
    G, edge_features, depth_map, width_map, height_map = _make_inaccessible_graph()
    ship = _make_huge_ship()
    nodes = dict(G.nodes(data=True))
    cc = PhysicalConstraintChecker(edge_features, nodes, G)
    cc.depth_map = depth_map
    cc.width_map = width_map
    cc.height_map = height_map
    nav = MultiObjectiveNavigator(G, edge_features, cc)

    # 关键：不传 blocked_edges（模拟正常调用）
    paths = nav.find_paths(start=0, end=1, ship=ship, hour=12, max_paths=3)
    path_types = [p.path_type for p in paths]
    print(f"  生成路径类型: {path_types}")

    assert PathType.RELAXED in path_types, \
        f"应包含'约束放宽路径'作为兜底，实际: {path_types}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
