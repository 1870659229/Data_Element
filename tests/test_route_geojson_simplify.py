# -*- coding: utf-8 -*-
"""
测试 build_route_geojson 在拼接 edge_waypoints 时会跳过距离 < 30m 的噪声点。
根因：edge_waypoints.csv 中部分 edge 含有 60+ 个间距 10~20m 的重复点，
直接拼接导致渲染折线 22% 段折返，地图上呈"乱画"状。
"""
import os
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import pytest
import app as app_module
from app import build_route_geojson, haversine_distance


@pytest.fixture
def setup_globals():
    """准备 2 个真实节点 + 1 个含密集噪声的 edge_waypoints 段。"""
    app_module.nodes_data.clear()
    app_module.edge_waypoints.clear()

    # 两个相距 200m 的"节点"，中间是噪声 waypoint 列表
    app_module.nodes_data[10] = {'lat': 23.1000, 'lon': 113.0000, 'type': 'wp', 'frequency': 100, 'ship_count': 0}
    app_module.nodes_data[20] = {'lat': 23.1018, 'lon': 113.0000, 'type': 'wp', 'frequency': 100, 'ship_count': 0}

    # 10→20 方向：1 个真实中间点 + 5 个挤在 < 30m 的噪声点 + 1 个真实中间点
    noisy = [
        # 第一个真实点，距起点 70m
        {'lat': 23.10063, 'lon': 113.0000, 'sequence': 0},
        # 5 个噪声点，挤在第一个真实点周围 10~20m
        {'lat': 23.10070, 'lon': 113.0000, 'sequence': 1},
        {'lat': 23.10078, 'lon': 113.0000, 'sequence': 2},
        {'lat': 23.10085, 'lon': 113.0000, 'sequence': 3},
        {'lat': 23.10092, 'lon': 113.0000, 'sequence': 4},
        {'lat': 23.10099, 'lon': 113.0000, 'sequence': 5},
        # 第二个真实点，距第一个真实点约 80m
        {'lat': 23.10135, 'lon': 113.0000, 'sequence': 6},
    ]
    app_module.edge_waypoints[(10, 20)] = noisy

    yield

    app_module.nodes_data.clear()
    app_module.edge_waypoints.clear()


def _build_path_info():
    return {
        'nodes': [10, 20],
        'type': 'TEST',
        'total_distance_km': 0.2,
        'total_time_min': 1.0,
        'avg_speed_knots': 5.0,
        'risk_score': 0,
        'safety_score': 100,
    }


def test_no_adjacent_coords_closer_than_30m(setup_globals):
    """距离上一个已收点 < 30m 的 waypoint 应被跳过。"""
    path_info = _build_path_info()
    result = build_route_geojson(path_info, path_index=0)
    coords = result['coordinates']  # [[lon, lat], ...]

    # 至少包含起点、中间、终点 = 3 个点
    assert len(coords) >= 3, f"应保留至少起点+真实中间+终点，得到 {len(coords)}"

    # 任意相邻两点距离 >= 30m（噪声点应被过滤）
    for i in range(1, len(coords)):
        prev_lon, prev_lat = coords[i - 1]
        cur_lon, cur_lat = coords[i]
        d = haversine_distance(prev_lat, prev_lon, cur_lat, cur_lon)
        assert d >= 30, f"相邻坐标 #{i-1}→#{i} 距离仅 {d:.1f}m（应 ≥ 30m）"


def test_keeps_real_movement_above_30m(setup_globals):
    """真实轨迹段（>30m）必须保留——不能把所有稠密点都过滤掉。"""
    path_info = _build_path_info()
    result = build_route_geojson(path_info, path_index=0)
    coords = result['coordinates']

    # 输入：起点 + 7 个 waypoint (2 真实 + 5 噪声挤在 30m 内) + 终点 = 9
    # 期望：起点 + wp0 (真实 70m) + wp6 (真实 80m) + 终点 = 4
    # 噪声段被 RDP + 30m 双重过滤压平
    assert len(coords) == 4, f"应保留 4 个点（起点+2真实+终点），得到 {len(coords)}: {coords}"


def test_rdp_removes_foldback_cluster(setup_globals):
    """RDP 应消除 fold-back 密集簇：起点→终点连线相近的中间点被剔除。"""
    app_module.edge_waypoints.clear()
    # 11 个点全部挤在 80m 范围内，模拟 GPS 锚地采样残留
    # 起点 23.1560/112.8200 → 终点 23.1570/112.8190 直线 ~140m
    # 中间 9 个 fold-back 点偏离首尾连线 < 20m
    app_module.edge_waypoints[(10, 20)] = [
        {'lat': 23.1560, 'lon': 112.8200, 'sequence': 0},
        {'lat': 23.1561, 'lon': 112.8198, 'sequence': 1},
        {'lat': 23.1562, 'lon': 112.8197, 'sequence': 2},
        {'lat': 23.1561, 'lon': 112.8196, 'sequence': 3},
        {'lat': 23.1563, 'lon': 112.8195, 'sequence': 4},
        {'lat': 23.1564, 'lon': 112.8194, 'sequence': 5},
        {'lat': 23.1563, 'lon': 112.8193, 'sequence': 6},
        {'lat': 23.1565, 'lon': 112.8192, 'sequence': 7},
        {'lat': 23.1566, 'lon': 112.8191, 'sequence': 8},
        {'lat': 23.1567, 'lon': 112.8190, 'sequence': 9},
        {'lat': 23.1568, 'lon': 112.8190, 'sequence': 10},
    ]
    # node 20 (23.1018, 113.0) 离 fold-back 簇 ~18km，会被拼接但 RDP/30m 不影响
    path_info = {
        'nodes': [10, 20],
        'type': 'TEST',
        'total_distance_km': 1.0,
        'total_time_min': 5.0,
        'avg_speed_knots': 5.0,
        'risk_score': 0,
        'safety_score': 100,
    }
    result = build_route_geojson(path_info, path_index=0)
    coords = result['coordinates']

    # 起点(10) + RDP 简化后的 fold-back 簇(≤ 2 个偏离 > 30m 的点) + 终点(20)
    # 由于 node 20 与 fold-back 簇相距 18km，30m 过滤不影响
    # 关键是 fold-back 簇 RDP 简化后 ≤ 2 个
    # 起点与 fold-back 第一个点 (23.1568) 距离 ~140m > 30m
    # 所以输出：起点 + 1~2 个 fold-back + 终点 = 3~4
    assert len(coords) <= 4, f"RDP 应大幅简化 fold-back 簇，期望 ≤ 4 个点，得到 {len(coords)}: {coords}"
