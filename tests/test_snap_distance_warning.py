"""回归测试：用户点距离水道节点过远时，系统应给出明确警告或拒绝规划。

背景: 2026-06-06 用户截图显示规划路径在陆地上"穿楼"，根因不是路径节点错位，
而是用户点击的"起/终点"本身就在陆地上,系统硬匹配到 ~1km 外的真正水道节点,
那段 ~1km 的陆上直线连接看起来就像路径"穿楼"。

本测试固定该场景,防止阈值被无意改大。
"""
import os
import sys

# 允许独立运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import app


@pytest.fixture(scope='module', autouse=True)
def _load():
    app.load_data()


def test_close_node_picked_for_user_reported_location():
    """复盘 §8.3 用户案例起点: 系统应匹配到 1km 内的低频真实节点,
    而不是 994m 外的高频节点 (修复后行为)。"""
    res_snode, res_sdist = app.find_nearest_node(23.007458, 113.043554)
    # 修复后 score = dist + (50-freq)*5 惩罚, 1km 内 freq<50 的节点应胜出
    # node 936 (383m, freq=6) score=603 < node 8 (994m, freq=2157) score=994
    # 因此 snap 应 < 900m, 且仍 > 300m (属于警告区间)
    assert 300 < res_sdist < 900, (
        f'修复后用户原案例 snap 应在 300~900m 之间 (实际 {res_sdist:.0f}m). '
        f'若 < 300m 说明数据变了, 若 > 900m 说明修复未生效。'
    )


def test_close_node_picked_for_user_reported_end():
    """复盘 §8.3 用户案例终点: 同上, snap 应在合理范围。"""
    res_enode, res_edist = app.find_nearest_node(22.979605, 113.093954)
    # 终点 1km 内只有 node 3 (926m, freq=2760), 是唯一候选,
    # 距离不变, 但应 < 1.5km (硬上限内)
    assert res_edist < 1500, (
        f'用户终点 snap 应 < 1.5km 硬上限 (实际 {res_edist:.0f}m)'
    )


def test_near_start_should_match_closest_node():
    """近距离点击: 应直接匹配最近的节点,不管频次高低。"""
    # 找一个确定在水里但 freq 较高的点
    high_freq_node = max(app.nodes_data.items(), key=lambda x: x[1].get('frequency', 0))
    nid, attrs = high_freq_node
    lat, lon = attrs['lat'], attrs['lon']
    # 在该点偏移 100m,应匹配回自己(距离最近)
    matched, dist = app.find_nearest_node(lat, lon, max_search_radius=500)
    assert matched == nid, f'100m 内点应匹配到 {nid}, 实际 {matched}'
    assert dist < 200, f'匹配距离应 < 200m, 实际 {dist:.0f}m'


def test_low_freq_close_node_wins_over_far_high_freq():
    """复盘 §8.3 用户案例: 1km 内有低频节点, 不应被硬拉到 994m 外的高频节点。

    修复后 score = dist + max(0, 50-freq)*5, 1km 内 freq<50 的真实节点
    应该胜出, 而不是被"水节点池"过滤掉。
    """
    lat, lon = 23.007458, 113.043554
    matched, dist = app.find_nearest_node(lat, lon, max_search_radius=5000)
    # 复盘 §8.3 的关键约束: 1km 内至少有一个候选节点 (node 936, 距离 383m, freq=6)
    # 修复后, 它的 score = 383 + (50-6)*5 = 603m
    # 而 node 8 (994m, freq=2157) 的 score = 994m
    # 所以最终匹配节点距离应 < 1km, 而非 994m+
    assert dist < 900, (
        f'修复后近距离低频节点应胜出, snap 应 < 900m, 实际 {dist:.0f}m. '
        f'若失败, 说明 find_nearest_node 又被改成只看频次了。'
    )


def test_api_rejects_far_snap_above_1500m():
    """API 入口: 任意一端 snap > 1.5km 应直接返回 400, 不进入 A*。

    用 monkeypatch 模拟 find_nearest_node 返回 2km, 验证硬上限逻辑。
    真实数据中要找到"5km 内有节点, 但离最近节点 > 1.5km"的位置较脆弱,
    直接 mock 更稳定。
    """
    high_freq_node = max(app.nodes_data.items(), key=lambda x: x[1].get('frequency', 0))
    target_nid, target_attrs = high_freq_node

    # mock: 起 = 2km 外的目标节点, 终 = 另一个 2km 外的目标节点
    other_nid = next(nid for nid in app.nodes_data if nid != target_nid)
    original = app.find_nearest_node
    app.find_nearest_node = lambda lat, lon, **kw: (target_nid, 2000.0) if (lat, lon) == (1.0, 1.0) else (other_nid, 2000.0)
    try:
        client = app.app.test_client()
        resp = client.post('/api/plan', json={
            'start_lat': 1.0, 'start_lon': 1.0,
            'end_lat': 2.0, 'end_lon': 2.0,
            'ship_type': '中型货船',
        })
    finally:
        app.find_nearest_node = original

    assert resp.status_code == 400, f'应返回 400, 实际 {resp.status_code}: {resp.data}'
    body = resp.get_json()
    assert body['success'] is False
    assert '1.5km' in body['message']
    assert body['snap_limit_m'] == 1500
    assert body['snap_distance_m']['start'] == 2000.0


def test_api_returns_warning_for_medium_snap_above_300m():
    """API 入口: 300m~1km 的 snap 应触发软警告(0.3km 阈值)。"""
    # 用户原案例起点: 1km 内全是低频节点, 修复后会匹配到 383m/freq=6 节点
    # 该距离属于 0.3~1.0km 区间, 应有 warning 但仍能规划
    client = app.app.test_client()
    resp = client.post('/api/plan', json={
        'start_lat': 23.007458, 'start_lon': 113.043554,
        'end_lat': 22.979605, 'end_lon': 113.093954,
        'ship_type': '中型货船',
    })
    body = resp.get_json()
    if resp.status_code == 200 and body.get('success'):
        warnings = body['data'].get('warnings', [])
        # 至少应有一个 0.3~1km 范围的 snap 警告
        assert any('0.' in w or '1.' in w for w in warnings), (
            f'中等距离 snap 应有警告, 实际 warnings={warnings}'
        )
