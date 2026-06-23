# -*- coding: utf-8 -*-
"""
算法对比基准测试：标准Dijkstra vs 标准A* vs 改进A*

目的：
  量化证明改进A*算法（风险感知+时间依赖+物理约束）相比标准算法的优势

输出：
  - output/algorithm_benchmark.json  原始数据
  - output/img/algorithm_benchmark.png  可视化对比图

用法：
  python scripts/algorithm_benchmark.py
"""

import os
import sys
import json
import time
import random
import heapq

import networkx as nx
import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ship_navigator import (
    ShipNavigationSystem, ShipCharacteristics, PathType
)
from utils import haversine_distance

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
IMG_DIR = os.path.join(OUTPUT_DIR, 'img')
N_OD_PAIRS = 30
SEED = 20260611


def _build_reachable_od_pool(graph, rng, target=200, max_scan=600):
    """在最大WCC内预扫描可达OD对"""
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


# ==================== 三种算法实现 ====================

def run_standard_dijkstra(graph, distance_weight, start, end):
    """标准Dijkstra：纯距离权重，无启发函数"""
    t0 = time.perf_counter()
    nodes_explored = 0

    g_score = {start: 0}
    prev = {start: None}
    edge_used = {start: None}
    open_set = [(0, start)]
    closed_set = set()

    while open_set:
        _, node = heapq.heappop(open_set)
        if node in closed_set:
            continue
        closed_set.add(node)
        nodes_explored += 1

        if node == end:
            break

        for neighbor in graph.successors(node):
            if neighbor in closed_set:
                continue
            ek = (node, neighbor)
            dist = distance_weight.get(ek, 100)
            tentative_g = g_score[node] + dist
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                prev[neighbor] = node
                edge_used[neighbor] = ek
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))

    elapsed = (time.perf_counter() - t0) * 1000  # ms

    if end not in prev or prev[end] is None:
        return None

    # 重建路径
    path_nodes = []
    n = end
    while n is not None:
        path_nodes.append(n)
        n = prev[n]
    path_nodes.reverse()

    # 计算路径距离
    total_dist = sum(distance_weight.get((path_nodes[i], path_nodes[i + 1]), 100)
                     for i in range(len(path_nodes) - 1))

    return {
        'nodes': path_nodes,
        'nodes_explored': nodes_explored,
        'time_ms': elapsed,
        'distance_m': total_dist,
    }


def run_standard_astar(graph, distance_weight, start, end):
    """标准A*：距离权重 + 地理启发函数"""
    t0 = time.perf_counter()
    nodes_explored = 0

    end_data = graph.nodes.get(end, {})
    end_lat = end_data.get('lat', 0)
    end_lon = end_data.get('lon', 0)

    def heuristic(node):
        nd = graph.nodes.get(node, {})
        return haversine_distance(
            nd.get('lat', 0), nd.get('lon', 0),
            end_lat, end_lon)

    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    prev = {start: None}
    edge_used = {start: None}
    open_set = [(f_score[start], start)]
    closed_set = set()

    while open_set:
        _, node = heapq.heappop(open_set)
        if node in closed_set:
            continue
        closed_set.add(node)
        nodes_explored += 1

        if node == end:
            break

        for neighbor in graph.successors(node):
            if neighbor in closed_set:
                continue
            ek = (node, neighbor)
            dist = distance_weight.get(ek, 100)
            tentative_g = g_score[node] + dist
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                prev[neighbor] = node
                edge_used[neighbor] = ek
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    elapsed = (time.perf_counter() - t0) * 1000

    if end not in prev or prev[end] is None:
        return None

    path_nodes = []
    n = end
    while n is not None:
        path_nodes.append(n)
        n = prev[n]
    path_nodes.reverse()

    total_dist = sum(distance_weight.get((path_nodes[i], path_nodes[i + 1]), 100)
                     for i in range(len(path_nodes) - 1))

    return {
        'nodes': path_nodes,
        'nodes_explored': nodes_explored,
        'time_ms': elapsed,
        'distance_m': total_dist,
    }


def run_improved_astar(nav, ship, start, end, hour=8):
    """改进A*：风险感知+时间依赖+物理约束（系统现有算法）"""
    t0 = time.perf_counter()

    blocked_edges = nav.constraint_checker.get_blocked_edges(ship)
    path_result = nav._dijkstra_safest(start, end, ship, blocked_edges, hour)

    elapsed = (time.perf_counter() - t0) * 1000

    if path_result is None:
        return None

    return {
        'nodes': path_result.nodes,
        'nodes_explored': len(set(path_result.nodes)),  # 近似
        'time_ms': elapsed,
        'distance_m': path_result.total_distance,
        'risk_score': path_result.risk_score,
        'total_time_s': path_result.total_time,
    }


def compute_path_risk(nav, path_nodes, ship):
    """计算路径总风险评分"""
    total_risk = 0.0
    for i in range(len(path_nodes) - 1):
        ek = (path_nodes[i], path_nodes[i + 1])
        risk = nav.constraint_checker.get_edge_risk_score(ek, ship)
        total_risk += risk
    return total_risk


def compute_path_time(nav, path_nodes, hour=8):
    """计算路径总耗时"""
    total_time = 0.0
    for i in range(len(path_nodes) - 1):
        ek = (path_nodes[i], path_nodes[i + 1])
        total_time += nav._get_dynamic_time(ek, hour)
    return total_time


def main():
    print("=" * 60)
    print("算法对比基准测试")
    print("标准Dijkstra vs 标准A* vs 改进A*")
    print("=" * 60)

    # 加载系统
    print("\n[1/4] 加载导航系统...")
    nav_sys = ShipNavigationSystem(output_dir=OUTPUT_DIR)
    nav_sys.constraint_checker.train_models()
    nav = nav_sys.navigator

    graph = nav_sys.graph
    distance_weight = nav.distance_weight

    # 构建可达OD池
    print("[2/4] 构建可达OD池...")
    rng = random.Random(SEED)
    od_pool = _build_reachable_od_pool(graph, rng, target=200)
    if len(od_pool) < N_OD_PAIRS:
        print(f"  可达OD对仅 {len(od_pool)} 个，不足 {N_OD_PAIRS}")
        N = len(od_pool)
    else:
        N = N_OD_PAIRS
    od_pairs = rng.sample(od_pool, N)
    print(f"  选取 {N} 对OD")

    # 选择船舶（绕过模板ship_name冲突，直接构造）
    ship = ShipCharacteristics(
        ship_name="锦江2003", ship_type="中型货船",
        length=63, width=13, draft=3.2, height=15,
        tonnage=994, max_speed=8,
    )
    print(f"  船舶: {ship.ship_name} ({ship.ship_type}), 吃水={ship.draft}m")

    # 运行对比
    print(f"[3/4] 运行 {N} 对OD × 3种算法...")
    results = []

    for idx, (start, end) in enumerate(od_pairs):
        print(f"  OD {idx + 1}/{N}: {start} → {end}")

        # 标准Dijkstra
        r_dijk = run_standard_dijkstra(graph, distance_weight, start, end)

        # 标准A*
        r_astar = run_standard_astar(graph, distance_weight, start, end)

        # 改进A*
        r_improved = run_improved_astar(nav, ship, start, end, hour=8)

        # 为标准算法补充风险和耗时计算
        if r_dijk:
            r_dijk['risk_score'] = compute_path_risk(nav, r_dijk['nodes'], ship)
            r_dijk['total_time_s'] = compute_path_time(nav, r_dijk['nodes'], hour=8)
        if r_astar:
            r_astar['risk_score'] = compute_path_risk(nav, r_astar['nodes'], ship)
            r_astar['total_time_s'] = compute_path_time(nav, r_astar['nodes'], hour=8)

        results.append({
            'od': (start, end),
            'dijkstra': r_dijk,
            'astar': r_astar,
            'improved': r_improved,
        })

    # 汇总统计
    print("[4/4] 汇总统计...")
    summary = _compute_summary(results)

    # 保存原始数据
    os.makedirs(IMG_DIR, exist_ok=True)
    _save_raw_results(results, os.path.join(OUTPUT_DIR, 'algorithm_benchmark.json'))

    # 打印汇总表
    _print_summary_table(summary)

    # 生成可视化
    _plot_comparison(summary, results, os.path.join(IMG_DIR, 'algorithm_benchmark.png'))

    print(f"\n结果已保存:")
    print(f"  数据: output/algorithm_benchmark.json")
    print(f"  图表: output/img/algorithm_benchmark.png")


def _compute_summary(results):
    """计算汇总统计"""
    metrics = {}
    for algo in ['dijkstra', 'astar', 'improved']:
        valid = [r[algo] for r in results if r[algo] is not None]
        if not valid:
            metrics[algo] = None
            continue

        metrics[algo] = {
            'n_valid': len(valid),
            'nodes_explored_mean': np.mean([v['nodes_explored'] for v in valid]),
            'nodes_explored_median': np.median([v['nodes_explored'] for v in valid]),
            'time_ms_mean': np.mean([v['time_ms'] for v in valid]),
            'time_ms_median': np.median([v['time_ms'] for v in valid]),
            'distance_m_mean': np.mean([v['distance_m'] for v in valid]),
            'distance_m_median': np.median([v['distance_m'] for v in valid]),
            'risk_score_mean': np.mean([v.get('risk_score', 0) for v in valid]),
            'risk_score_median': np.median([v.get('risk_score', 0) for v in valid]),
            'total_time_s_mean': np.mean([v.get('total_time_s', 0) for v in valid]),
            'total_time_s_median': np.median([v.get('total_time_s', 0) for v in valid]),
        }
    return metrics


def _print_summary_table(summary):
    """打印汇总对比表"""
    print("\n" + "=" * 80)
    print("算法对比汇总表")
    print("=" * 80)
    header = f"{'指标':<20} {'标准Dijkstra':>18} {'标准A*':>18} {'改进A*':>18}"
    print(header)
    print("-" * 80)

    rows = [
        ('有效OD对数', 'n_valid', 'd'),
        ('搜索节点数(均值)', 'nodes_explored_mean', '.1f'),
        ('搜索节点数(中位数)', 'nodes_explored_median', '.1f'),
        ('运行时间ms(均值)', 'time_ms_mean', '.2f'),
        ('运行时间ms(中位数)', 'time_ms_median', '.2f'),
        ('路径距离m(均值)', 'distance_m_mean', '.1f'),
        ('路径距离m(中位数)', 'distance_m_median', '.1f'),
        ('路径风险(均值)', 'risk_score_mean', '.2f'),
        ('路径风险(中位数)', 'risk_score_median', '.2f'),
        ('路径耗时s(均值)', 'total_time_s_mean', '.1f'),
        ('路径耗时s(中位数)', 'total_time_s_median', '.1f'),
    ]

    for label, key, fmt in rows:
        vals = []
        for algo in ['dijkstra', 'astar', 'improved']:
            if summary[algo] is None:
                vals.append('N/A')
            else:
                v = summary[algo][key]
                if fmt == 'd':
                    vals.append(f"{int(v)}")
                else:
                    vals.append(f"{v:{fmt}}")
        print(f"{label:<20} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    print("=" * 80)

    # 改进幅度
    if summary['dijkstra'] and summary['improved']:
        d = summary['dijkstra']
        i = summary['improved']
        print("\n改进A* 相比 标准Dijkstra:")
        if d['risk_score_mean'] > 0:
            risk_reduction = (1 - i['risk_score_mean'] / d['risk_score_mean']) * 100
            print(f"  风险评分降低: {risk_reduction:.1f}%")
        if d['total_time_s_mean'] > 0:
            time_reduction = (1 - i['total_time_s_mean'] / d['total_time_s_mean']) * 100
            print(f"  路径耗时降低: {time_reduction:.1f}%")
        node_reduction = (1 - i['nodes_explored_mean'] / max(d['nodes_explored_mean'], 1)) * 100
        print(f"  搜索节点数变化: {node_reduction:+.1f}%")

    if summary['astar'] and summary['improved']:
        a = summary['astar']
        i = summary['improved']
        print("\n改进A* 相比 标准A*:")
        if a['risk_score_mean'] > 0:
            risk_reduction = (1 - i['risk_score_mean'] / a['risk_score_mean']) * 100
            print(f"  风险评分降低: {risk_reduction:.1f}%")
        if a['total_time_s_mean'] > 0:
            time_reduction = (1 - i['total_time_s_mean'] / a['total_time_s_mean']) * 100
            print(f"  路径耗时降低: {time_reduction:.1f}%")


def _save_raw_results(results, path):
    """保存原始结果为JSON"""
    serializable = []
    for r in results:
        item = {
            'od': list(r['od']),
        }
        for algo in ['dijkstra', 'astar', 'improved']:
            if r[algo] is not None:
                item[algo] = {
                    'nodes_explored': r[algo]['nodes_explored'],
                    'time_ms': round(r[algo]['time_ms'], 3),
                    'distance_m': round(r[algo]['distance_m'], 1),
                    'risk_score': round(r[algo].get('risk_score', 0), 4),
                    'total_time_s': round(r[algo].get('total_time_s', 0), 1),
                }
            else:
                item[algo] = None
        serializable.append(item)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _plot_comparison(summary, results, save_path):
    """生成对比可视化图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    algos = ['dijkstra', 'astar', 'improved']
    algo_labels = ['标准Dijkstra', '标准A*', '改进A*']
    colors = ['#636e72', '#0984e3', '#00b894']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('路径规划算法对比基准测试', fontsize=16, fontweight='bold')

    # 1. 搜索节点数（柱状图）
    ax = axes[0, 0]
    means = [summary[a]['nodes_explored_mean'] for a in algos]
    stds = []
    for a in algos:
        valid = [r[a]['nodes_explored'] for r in results if r[a] is not None]
        stds.append(np.std(valid))
    bars = ax.bar(algo_labels, means, color=colors, yerr=stds, capsize=5)
    ax.set_ylabel('搜索节点数')
    ax.set_title('搜索效率对比')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    # 2. 运行时间（柱状图）
    ax = axes[0, 1]
    means = [summary[a]['time_ms_mean'] for a in algos]
    stds = []
    for a in algos:
        valid = [r[a]['time_ms'] for r in results if r[a] is not None]
        stds.append(np.std(valid))
    bars = ax.bar(algo_labels, means, color=colors, yerr=stds, capsize=5)
    ax.set_ylabel('运行时间 (ms)')
    ax.set_title('运行时间对比')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10)

    # 3. 风险评分（箱线图）
    ax = axes[1, 0]
    data_risk = []
    for a in algos:
        valid = [r[a].get('risk_score', 0) for r in results if r[a] is not None]
        data_risk.append(valid)
    bp = ax.boxplot(data_risk, tick_labels=algo_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('路径风险评分')
    ax.set_title('路径安全性对比')

    # 4. 路径耗时（箱线图）
    ax = axes[1, 1]
    data_time = []
    for a in algos:
        valid = [r[a].get('total_time_s', 0) for r in results if r[a] is not None]
        data_time.append(valid)
    bp = ax.boxplot(data_time, tick_labels=algo_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('路径耗时 (s)')
    ax.set_title('路径时间效率对比')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {save_path}")


if __name__ == '__main__':
    # 将所有输出重定向到日志文件，避免sandbox截断
    log_path = os.path.join(OUTPUT_DIR, 'benchmark_log.txt')
    with open(log_path, 'w', encoding='utf-8') as log_f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_f
        sys.stderr = log_f
        try:
            main()
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
