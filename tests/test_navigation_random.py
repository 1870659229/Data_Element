# -*- coding: utf-8 -*-
"""
随机采样测试：替代 main.py 中硬编码的 10 船型演示

背景:
- 原 main.py::_task7_navigation 会对 10 种船型各生成 navigation_*.json/.txt
  共 20 个样本文件,信息冗余且污染 output/ 目录
- 本测试随机采样 N 组 (ship_type, OD, hour) 组合,验证 plan_route 鲁棒性
- 所有结果汇总到单文件 output/navigation_random_sample.json

用法:
    pytest tests/test_navigation_random.py -v
    或: python tests/test_navigation_random.py
"""

import os
import json
import random
from datetime import datetime

import pytest

from ship_navigator import ShipNavigationSystem


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
SAMPLE_OUTPUT = os.path.join(OUTPUT_DIR, 'navigation_random_sample.json')
N_SAMPLES = 30  # 随机采样规模


def _data_available() -> bool:
    """检查依赖的拓扑/特征文件是否已生成"""
    required = [
        os.path.join(OUTPUT_DIR, 'topology_nodes.csv'),
        os.path.join(OUTPUT_DIR, 'topology_edges.csv'),
        os.path.join(OUTPUT_DIR, 'edge_features_dynamic_weights.csv'),
    ]
    return all(os.path.exists(p) for p in required)


@pytest.fixture(scope='module')
def nav_system():
    """共享一个 ShipNavigationSystem 实例,避免重复加载模型"""
    if not _data_available():
        pytest.skip("缺少拓扑/特征数据,请先运行 main.py Task 1-5")
    sys = ShipNavigationSystem(output_dir=OUTPUT_DIR)
    sys.constraint_checker.train_models()
    return sys


def _build_reachable_od_pool(graph, rng, target: int = 200, max_scan: int = 600):
    """在最大 WCC 内扫一组已知可达的 (start, end) 对，供随机采样复用。

    图是稀疏 DAG（1589 节点 / 2665 边），随机抽样 100 对中可达仅 1 对，
    因此必须预先发现可达对，再从中随机抽取。
    """
    import networkx as nx
    largest_wcc = max(nx.weakly_connected_components(graph), key=len)
    sample_nodes = rng.sample(list(largest_wcc), min(max_scan, len(largest_wcc)))
    pairs = []
    for i, s in enumerate(sample_nodes):
        for e in sample_nodes[i + 1:]:
            try:
                if nx.has_path(graph, s, e):
                    pairs.append((s, e))
                    if len(pairs) >= target:
                        return pairs
            except Exception:
                continue
    return pairs


def test_random_plan_route_robustness(nav_system):
    """随机 30 组 (ship_type, OD, hour) 验证 plan_route 鲁棒性"""
    rng = random.Random(20260605)
    ship_types = nav_system.list_ship_types()
    nodes = list(nav_system.graph.nodes())
    assert len(ship_types) >= 5, "船型模板数量异常"
    assert len(nodes) >= 10, "图中节点数过少"

    # 图是稀疏 DAG，必须先建立可达 OD 池
    reachable_pool = _build_reachable_od_pool(nav_system.graph, rng, target=200)
    if len(reachable_pool) < 5:
        pytest.skip(f"可达 OD 对不足({len(reachable_pool)}), 跳过测试")

    samples = []
    # 30 条样本的设计：25 条用 SHIP_TEMPLATES（多走 FREQUENT 分支），
    # 5 条用 custom_ship 注入"巨型船"参数（draft=12 / width=30 / height=30），
    # 巨型船与所有 narrow 边冲突，会被 get_blocked_edges 标红，
    # 触发 _dijkstra_safest 返回 None，从而走 PathType.RELAXED 兜底。
    # 这是为了保证 carbon_savings.png 至少有 2 个 path_type 分类。
    N_RELAXED_FORCED = 5
    for i in range(N_SAMPLES):
        ship_type = rng.choice(ship_types)
        start, end = rng.choice(reachable_pool)
        hour = rng.randint(0, 23)
        record = {
            'index': i,
            'ship_type': ship_type,
            'start': int(start),
            'end': int(end),
            'hour': hour,
        }
        try:
            kwargs = dict(
                start=int(start), end=int(end), ship_type=ship_type,
                departure_time=datetime(2026, 6, 5, hour, 0),
            )
            if i < N_RELAXED_FORCED:
                # 强制约束放宽：用巨型船模板（不存在于 SHIP_TEMPLATES 中），
                # 通过 custom_ship 注入。吃水 12m / 宽 30m / 高 30m
                # 超过所有 narrow 边的 depth=6/8 / width=40/60 / height=20/30。
                kwargs.pop('ship_type', None)
                kwargs['custom_ship'] = {
                    'ship_name': f'巨型油轮_{i:02d}',
                    'ship_type': '巨型油轮',
                    'length': 300.0, 'width': 30.0, 'draft': 12.0,
                    'height': 30.0, 'tonnage': 50000.0, 'max_speed': 14.0,
                }
                record['ship_type'] = '巨型油轮(custom)'
            decision = nav_system.plan_route(**kwargs)
            rec = decision.get('recommended_path', {})
            record.update({
                'success': decision.get('success', False),
                'path_type': rec.get('type'),
                'distance_km': rec.get('total_distance_km'),
                'time_min': rec.get('total_time_min'),
                'risk_score': rec.get('risk_score'),
                'constraints_met': rec.get('constraints_met'),
            })
        except Exception as e:  # 捕获异常但不中断整批
            record.update({'success': False, 'error': type(e).__name__ + ': ' + str(e)[:80]})
        samples.append(record)

    # 至少 60% 随机组合能找到路径(DAG 图 + 物理约束的自然失败率)
    success_count = sum(1 for s in samples if s.get('success'))
    assert success_count >= int(N_SAMPLES * 0.6), \
        f"成功率过低: {success_count}/{N_SAMPLES}"

    # 汇总写入单文件,替代 20 个分散的 navigation_*.json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SAMPLE_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(samples),
            'success': success_count,
            'samples': samples,
        }, f, ensure_ascii=False, indent=2)

    assert os.path.exists(SAMPLE_OUTPUT)
    # 输出仅 1 个新文件,不污染 output/
    nav_files = [f for f in os.listdir(OUTPUT_DIR)
                 if f.startswith('navigation_') and f != 'navigation_random_sample.json']
    assert len(nav_files) == 0, \
        f"不应再有 navigation_*.json/.txt 样本,发现: {nav_files[:5]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
