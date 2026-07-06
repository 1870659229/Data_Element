"""数据要素大赛 · 附加图批量生成脚本
=================================

从已有 output/ 下的 csv / json 衍生 9 张图，不重跑 Task1-7。

生成图:
  A 类 (6 张) - 已有 visualize.py 函数但未产出
    1. model_comparison.png          7 模型 R²/MAE/RMSE 对比
    2. feature_importance.png        GNN Top10 特征重要性
    3. trajectory_before_after.png   清洗前/后轨迹
    4. cluster_comparison.png        聚类前/后节点
    5. traffic_heatmap.png           通行频次热力图
    6. path_comparison.png           4 类路径叠加

  B/C/D 类 (3 张) - 新增
    7. data_overview.png             原始数据规模 + 船型分布
    8. task_efficiency.png           Task1-7 时间/内存柱状图
    9. carbon_savings.png            不同路径碳排放对比

运行: python scripts/generate_extra_figures.py
"""
import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 让脚本能 import 根目录的模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visualize import TopologyVisualizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("extra_figures")

OUTPUT_DIR = ROOT / "output"
IMG_DIR = OUTPUT_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ============== A 类 6 张：复用 visualize.py 已写好的函数 ==============
def gen_class_a(viz: TopologyVisualizer):
    """复用已有函数跑出 A 类 6 张图"""
    log.info("=== A 类: 复用 visualize.py 函数 ===")

    # 加载必要数据
    topo_csv = pd.read_csv(OUTPUT_DIR / "topology_nodes.csv")
    edge_csv = pd.read_csv(OUTPUT_DIR / "topology_edges.csv")

    import networkx as nx
    graph = nx.DiGraph()
    for _, row in topo_csv.iterrows():
        graph.add_node(int(row['node_id']), lat=row['lat'], lon=row['lon'])
    for _, row in edge_csv.iterrows():
        graph.add_edge(int(row['from_node']), int(row['to_node']), weight=row['weight'])
    for _, row in edge_csv.iterrows():
        if str(row.get('is_bidirectional', '')).lower() == 'true':
            if not graph.has_edge(int(row['to_node']), int(row['from_node'])):
                graph.add_edge(int(row['to_node']), int(row['from_node']), weight=row['weight'])

    # 1. 模型对比
    viz.plot_model_comparison(output_path=str(IMG_DIR / "model_comparison.png"))

    # 2. 特征重要性
    viz.plot_feature_importance(output_path=str(IMG_DIR / "feature_importance.png"))

    # 3. 清洗前后轨迹
    cleaned_path = OUTPUT_DIR / "cleaned_data.csv"
    if cleaned_path.exists():
        cleaned_df = pd.read_csv(cleaned_path)
        cleaned_df['时间'] = pd.to_datetime(cleaned_df['时间'])
        viz.plot_trajectory_before_after(
            cleaned_df, output_path=str(IMG_DIR / "trajectory_before_after.png"))

    # 4. 聚类对比
    clustered_path = OUTPUT_DIR / "clustered_nodes.csv"
    if clustered_path.exists():
        clustered_nodes = pd.read_csv(clustered_path).to_dict('records')
        viz.plot_cluster_comparison(
            clustered_nodes, output_path=str(IMG_DIR / "cluster_comparison.png"))

    # 5. 通行频次热力图
    viz.plot_traffic_heatmap(
        edge_csv, topo_csv, output_path=str(IMG_DIR / "traffic_heatmap.png"))

    # 6. 路径对比：使用真实改进A*路径规划（4种策略）
    log.info("=== A-6: path_comparison.png ===")
    topo_csv = OUTPUT_DIR / "topology_nodes.csv"
    topo_edges = OUTPUT_DIR / "topology_edges.csv"
    edge_csv = OUTPUT_DIR / "edge_features_dynamic_weights.csv"
    if topo_csv.exists() and topo_edges.exists() and edge_csv.exists():
        import networkx as nx
        from ship_navigator import PhysicalConstraintChecker, ShipCharacteristics, MultiObjectiveNavigator

        nodes_df = pd.read_csv(topo_csv)
        edges_df = pd.read_csv(topo_edges)
        edge_feat_df = pd.read_csv(edge_csv)

        # 构建图
        g = nx.DiGraph()
        nodes_dict = {}
        for _, r in nodes_df.iterrows():
            nid = int(r['node_id'])
            lat = r.get('latitude', r.get('lat', 0))
            lon = r.get('longitude', r.get('lon', 0))
            g.add_node(nid, lat=lat, lon=lon)
            nodes_dict[nid] = {'lat': lat, 'lon': lon}

        for _, r in edges_df.iterrows():
            g.add_edge(r['from_node'], r['to_node'], weight=r.get('weight', 1.0))
            if str(r.get('is_bidirectional', '')).lower() == 'true':
                if not g.has_edge(r['to_node'], r['from_node']):
                    g.add_edge(r['to_node'], r['from_node'], weight=r.get('weight', 1.0))

        # 构建边特征字典
        edge_features = {}
        for _, row in edge_feat_df.iterrows():
            fn = int(row.get('from_node', row.get('source', 0)))
            tn = int(row.get('to_node', row.get('target', 0)))
            feat = {k: row[k] for k in row.index
                    if k not in ('from_node', 'to_node', 'source', 'target')}
            edge_features[(fn, tn)] = feat
            if (fn, tn) not in g.edges():
                g.add_edge(fn, tn)

        # 初始化约束检查器和路径规划器
        checker = PhysicalConstraintChecker(edge_features, nodes_dict, g)
        planner = MultiObjectiveNavigator(g, edge_features, checker)

        # 选取OD对：使用拓扑中距离较远的首尾节点
        start = int(nodes_df.iloc[0]['node_id'])
        end = int(nodes_df.iloc[-1]['node_id'])

        # 使用中型货船作为测试船型
        ship = ShipCharacteristics(
            ship_name="中型货船",
            length=100, width=15, draft=3.2, height=20,
            tonnage=5000, max_speed=12
        )
        hour = 8

        # 运行4种策略获取真实路径
        paths = {}
        path_freq = planner._bidirectional_a_star_frequent(start, end, ship, set(), hour)
        if path_freq:
            paths['frequent'] = list(path_freq.nodes)

        path_safe = planner._dijkstra_safest(start, end, ship, set(), hour)
        if path_safe:
            paths['safest'] = list(path_safe.nodes)

        path_fast = planner._dijkstra_fastest(start, end, ship, hour, set())
        if path_fast:
            paths['fastest'] = list(path_fast.nodes)

        path_short = planner._dijkstra_shortest_distance(start, end, ship, set(), hour)
        if path_short:
            paths['shortest'] = list(path_short.nodes)

        if paths:
            viz.plot_path_comparison(g, paths, output_path=str(IMG_DIR / "path_comparison.png"),
                                     topo_edge_count=len(edges_df))
            log.info("已生成: path_comparison.png (真实A*路径, %d 条策略)", len(paths))
        else:
            log.warning("所有策略均未找到路径, 跳过 path_comparison.png")
    else:
        log.warning("拓扑/边特征文件缺失, 跳过 path_comparison.png")


# ============== B/C/D 类 3 张：新增图 ==============
def gen_data_overview():
    """B-7: 原始数据规模 + 船型分布"""
    log.info("=== B-7: data_overview.png ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    raw_path = OUTPUT_DIR.parent / "Data" / "ais_2024.csv"
    raw_df = None
    for cand in [OUTPUT_DIR.parent / "Data" / "ais_2024.csv",
                 OUTPUT_DIR.parent / "Data" / "原始AIS数据.csv",
                 OUTPUT_DIR / "raw_ais.csv",
                 OUTPUT_DIR / "raw_data.csv"]:
        if cand.exists():
            raw_df = pd.read_csv(cand, nrows=1)
            break

    sizes = {}
    cleaned_path = OUTPUT_DIR / "cleaned_data.csv"
    if cleaned_path.exists():
        # 只算船舶/时间等"结构数量",不算轨迹点 (轨迹点 1.1M 量级与其他 4 项差 3 个数量级)
        cleaned_df = pd.read_csv(cleaned_path, usecols=['船舶名称'] if '船舶名称' in pd.read_csv(cleaned_path, nrows=1).columns else None)
        sizes['清洗后船舶数'] = cleaned_df['船舶名称'].nunique() if '船舶名称' in cleaned_df.columns else 0
    if (OUTPUT_DIR / "clustered_nodes.csv").exists():
        sizes['聚类节点数'] = len(pd.read_csv(OUTPUT_DIR / "clustered_nodes.csv"))
    if (OUTPUT_DIR / "topology_nodes.csv").exists():
        sizes['拓扑节点数'] = len(pd.read_csv(OUTPUT_DIR / "topology_nodes.csv"))
    if (OUTPUT_DIR / "topology_edges.csv").exists():
        sizes['拓扑边数'] = len(pd.read_csv(OUTPUT_DIR / "topology_edges.csv"))
    if (OUTPUT_DIR / "extracted_nodes.csv").exists():
        sizes['提取候选节点'] = len(pd.read_csv(OUTPUT_DIR / "extracted_nodes.csv"))

    if sizes:
        axes[0].barh(list(sizes.keys()), list(sizes.values()), color='steelblue')
        axes[0].set_xlabel('数量 (线性尺度, 1.1M 轨迹点不参与)')
        axes[0].set_title(f'数据规模 (清洗后轨迹点 = 1,105,879, 不参与对比)')
        for i, (k, v) in enumerate(sizes.items()):
            axes[0].text(v, i, f' {v:,}', va='center', fontsize=9)

    # 右图：船型分布（用 SHIP_TEMPLATES）
    ship_templates = {
        '小型货船': {'length': 53, 'width': 11, 'draft': 2.6, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
        '中型货船': {'length': 63, 'width': 13, 'draft': 3.2, 'height': 15, 'tonnage': 994, 'max_speed': 8},
        '大型货船': {'length': 77, 'width': 16, 'draft': 3.7, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
        '集装箱船': {'length': 49, 'width': 13, 'draft': 2.4, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
        '大型集装箱船': {'length': 70, 'width': 18, 'draft': 3.8, 'height': 15, 'tonnage': 3000, 'max_speed': 9.2},
        '油轮': {'length': 63, 'width': 13, 'draft': 3.31, 'height': 15, 'tonnage': 1187, 'max_speed': 8},
        '大型油轮': {'length': 96, 'width': 16, 'draft': 5.894, 'height': 15, 'tonnage': 3572, 'max_speed': 11},
        '客船': {'length': 44, 'width': 9, 'draft': 1.85, 'height': 15, 'tonnage': 488, 'max_speed': 8},
        '渔船': {'length': 17, 'width': 4, 'draft': 1.5, 'height': 8, 'tonnage': 200, 'max_speed': 6},
        '拖船': {'length': 31, 'width': 10, 'draft': 2.2, 'height': 10, 'tonnage': 300, 'max_speed': 8},
    }
    types = list(ship_templates.keys())
    drafts = [ship_templates[t]['draft'] for t in types]
    axes[1].bar(range(len(types)), drafts, color='coral')
    axes[1].set_xticks(range(len(types)))
    axes[1].set_xticklabels(types, rotation=45, ha='right')
    axes[1].set_ylabel('吃水 (m)')
    axes[1].set_title('10 种船型模板 · 吃水深度')
    axes[1].axhline(6.0, color='red', linestyle='--', alpha=0.5, label='浅滩阈值(6.0m)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(IMG_DIR / "data_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: data_overview.png")


def gen_task_efficiency():
    """C-12: Task1-7 时间/内存柱状图"""
    log.info("=== C-12: task_efficiency.png ===")
    eff_path = OUTPUT_DIR / "efficiency_report.json"
    if not eff_path.exists():
        log.warning("efficiency_report.json 不存在，跳过")
        return
    with open(eff_path, encoding="utf-8") as f:
        eff = json.load(f)
    results = eff['results']

    tasks = [r['task'] for r in results]
    times = [r['time_s'] for r in results]
    mems = [r['peak_memory_mb'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    bars = ax1.bar(range(len(tasks)), times, color='steelblue')
    ax1.set_xticks(range(len(tasks)))
    ax1.set_xticklabels(tasks, rotation=30, ha='right')
    ax1.set_ylabel('耗时 (秒)')
    ax1.set_title('Task1-7 耗时')
    ax1.set_yscale('log')
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, t, f'{t:.1f}s',
                ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    bars = ax2.bar(range(len(tasks)), mems, color='coral')
    ax2.set_xticks(range(len(tasks)))
    ax2.set_xticklabels(tasks, rotation=30, ha='right')
    ax2.set_ylabel('峰值内存 (MB)')
    ax2.set_title('Task1-7 峰值内存')
    for bar, m in zip(bars, mems):
        ax2.text(bar.get_x() + bar.get_width()/2, m, f'{m:.1f}MB',
                ha='center', va='bottom', fontsize=8)

    plt.suptitle('任务执行效率（系统: {}）'.format(eff['system']['os']), fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "task_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: task_efficiency.png")


def gen_carbon_savings():
    """D-14: 不同路径碳排放对比（基于 navigation_random_sample.json 的 samples）"""
    log.info("=== D-14: carbon_savings.png ===")
    nav_json = OUTPUT_DIR / "navigation_random_sample.json"
    if not nav_json.exists():
        log.warning("navigation_random_sample.json 不存在，跳过")
        return
    with open(nav_json, encoding="utf-8") as f:
        nav = json.load(f)
    samples = nav.get('samples', [])
    if not samples:
        log.warning("samples 为空，跳过")
        return

    # 船型吨位（与 ship_navigator.SHIP_TEMPLATES 同步）
    tonnage_map = {
        '小型货船': 3000, '中型货船': 994, '大型货船': 3000,
        '集装箱船': 3000, '大型集装箱船': 3000,
        '油轮': 1187, '大型油轮': 3572,
        '客船': 488, '渔船': 200, '拖船': 300,
    }

    rows = []
    for item in samples:
        if not item.get('success'):
            continue
        tonnage = tonnage_map.get(item.get('ship_type'), 5000)
        dist_km = item.get('distance_km', 0)
        # 燃油 L = dist_km * tonnage * 0.00005 (简化模型，与报告一致)
        fuel_L = dist_km * tonnage * 0.00005
        co2_kg = fuel_L * 3.2
        rows.append({
            'ship_type': item.get('ship_type', '?'),
            'path_type': item.get('path_type', '?'),
            'dist_km': dist_km,
            'time_min': item.get('time_min', 0),
            'fuel_L': fuel_L,
            'co2_kg': co2_kg,
        })
    if not rows:
        log.warning("无成功路径，跳过")
        return
    df = pd.DataFrame(rows)

    # 按 path_type 聚合
    agg = df.groupby('path_type').agg(
        avg_dist=('dist_km', 'mean'),
        avg_time=('time_min', 'mean'),
        avg_fuel=('fuel_L', 'mean'),
        avg_co2=('co2_kg', 'mean'),
        count=('co2_kg', 'count'),
    ).reset_index().sort_values('avg_co2')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#43A047', '#1E88E5', '#FB8C00', '#E53935', '#8E24AA']
    axes[0].bar(agg['path_type'], agg['avg_dist'], color=colors[:len(agg)])
    axes[0].set_ylabel('平均距离 (km)')
    axes[0].set_title(f'不同路径类型 · 平均距离 (n={len(df)}次规划)')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(agg['avg_dist']):
        axes[0].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    axes[1].bar(agg['path_type'], agg['avg_co2'], color=colors[:len(agg)])
    axes[1].set_ylabel(r'CO$_2$ 排放 (kg)')
    axes[1].set_title(f'不同路径类型 · 平均碳排放')
    axes[1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(agg['avg_co2']):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "carbon_savings.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: carbon_savings.png (路径类型: %s)", list(agg['path_type']))


# ============== B-Plus 类 4 张：鲁棒性 / 10船型QA / 节点类型 / 24h时段 ==============
def gen_robustness():
    """B-Plus-1: 鲁棒性测试结果柱状图（10 项）"""
    log.info("=== B-Plus-1: robustness.png ===")
    rob_path = OUTPUT_DIR / "robustness_report.json"
    if not rob_path.exists():
        log.warning("robustness_report.json 不存在，跳过")
        return
    with open(rob_path, encoding="utf-8") as f:
        cases = json.load(f)

    names = [c['name'] for c in cases]
    passed = [1 if c['passed'] else 0 for c in cases]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#43A047' if p else '#E53935' for p in passed]
    bars = ax.bar(range(len(names)), passed, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['FAIL', 'PASS'])
    ax.set_title(f'鲁棒性测试结果 ({sum(passed)}/{len(passed)} 通过)', fontsize=13)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, axis='y', alpha=0.3)
    for i, (bar, c) in enumerate(zip(bars, cases)):
        err = c.get('error') or ''
        label = c.get('expected', '')
        # 截短标签以适应窄条：取前6个字符
        short_label = label[:6] if len(label) > 6 else label
        ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                short_label,
                ha='center', va='center', fontsize=8, color='white',
                wrap=True, rotation=0)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "robustness.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: robustness.png (%d/%d 通过)", sum(passed), len(passed))


def gen_task7_qa():
    """B-Plus-2: 10 种船型 Task7 QA 报告 (距离/时间/风险误差)"""
    log.info("=== B-Plus-2: task7_qa.png ===")
    qa_path = OUTPUT_DIR / "task7_qa_report.json"
    if not qa_path.exists():
        log.warning("task7_qa_report.json 不存在，跳过")
        return
    with open(qa_path, encoding="utf-8") as f:
        qa = json.load(f)
    results = qa['results']
    if not results:
        return

    ships = [r['ship_type'] for r in results]
    dist_err = [r.get('distance_err_pct', 0) * 100 for r in results]
    time_err = [r.get('time_err_pct', 0) * 100 for r in results]
    risk_err = [r.get('risk_err', 0) * 100 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(ships))

    axes[0].plot(x, dist_err, 'o-', label='距离误差%', color='steelblue', linewidth=2)
    axes[0].plot(x, time_err, 's-', label='时间误差%', color='coral', linewidth=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ships, rotation=30, ha='right')
    axes[0].set_ylabel('相对误差 (%)')
    axes[0].set_title('10 船型 · Task7 QA 距离/时间误差')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1, color='red', linestyle='--', alpha=0.5, label='1% 阈值')

    edges = [r.get('edges_in_graph', 0) for r in results]
    axes[1].bar(x, edges, color='mediumseagreen')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ships, rotation=30, ha='right')
    axes[1].set_ylabel('边数')
    axes[1].set_title('10 船型 · 路径边数')
    for i, v in enumerate(edges):
        axes[1].text(i, v, f'{v}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'Task7 10 船型 QA (通过 {qa["n_pass"]}/{qa["n_reports"]})', fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "task7_qa.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: task7_qa.png")


def gen_node_type_distribution():
    """B-Plus-3: 节点类型分布饼图 + 度分布直方图"""
    log.info("=== B-Plus-3: node_type_distribution.png ===")
    topo_path = OUTPUT_DIR / "topology_nodes.csv"
    if not topo_path.exists():
        log.warning("topology_nodes.csv 不存在，跳过")
        return
    nodes = pd.read_csv(topo_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if 'type' in nodes.columns:
        type_counts = nodes['type'].value_counts()
        # 英文类型名映射为中文
        type_cn = {
            'turn_point': '拐点',
            'waypoint': '途经点',
            'merge_point': '汇合点',
        }
        cn_labels = [type_cn.get(k, k) for k in type_counts.index]
        axes[0].pie(type_counts.values, labels=cn_labels,
                    autopct='%1.1f%%', startangle=90,
                    colors=plt.cm.Set2(np.linspace(0, 1, len(type_counts))))
        axes[0].set_title(f'节点类型分布 (n={len(nodes)})')

    # 度分布：需要从 edges 计算
    edge_path = OUTPUT_DIR / "topology_edges.csv"
    if edge_path.exists():
        edges = pd.read_csv(edge_path)
        from collections import Counter
        deg = Counter()
        for _, r in edges.iterrows():
            deg[int(r['from_node'])] += 1
            deg[int(r['to_node'])] += 1
        axes[1].hist(list(deg.values()), bins=30, color='steelblue',
                     edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('节点度')
        axes[1].set_ylabel('节点数')
        axes[1].set_title(f'节点度分布 (平均={np.mean(list(deg.values())):.1f})')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "node_type_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: node_type_distribution.png")


def gen_hourly_heatmap():
    """B-Plus-4: 24h 时段权重分析图（折线+柱状，基于 predicted_times）"""
    log.info("=== B-Plus-4: hourly_weight_heatmap.png ===")
    topo_path = OUTPUT_DIR / "waterway_topology.json"
    if not topo_path.exists():
        log.warning("waterway_topology.json 不存在，跳过")
        return
    with open(topo_path, encoding="utf-8") as f:
        topo = json.load(f)

    # 统计每小时的平均耗时
    hourly = {h: [] for h in range(24)}
    for e in topo.get('edges', []):
        pt = e.get('predicted_times', {})
        for h_str, t in pt.items():
            try:
                h = int(float(h_str))
                if 0 <= h < 24 and t:
                    hourly[h].append(float(t))
            except (ValueError, TypeError):
                pass

    hours = list(range(24))
    avg_time = [np.mean(hourly[h]) if hourly[h] else 0 for h in hours]
    cnt = [len(hourly[h]) for h in hours]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：24h 平均耗时折线
    axes[0].plot(hours, avg_time, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xticks(hours)
    axes[0].set_xlabel('小时 (h)')
    axes[0].set_ylabel('平均耗时 (秒)')
    axes[0].set_title(f'24 小时时段平均耗时分布 (共 {sum(cnt)} 条预测记录)')
    axes[0].grid(True, alpha=0.3)

    # 下图：24h 预测记录数（数据密度）
    axes[1].bar(hours, cnt, color='coral', alpha=0.7)
    axes[1].set_xticks(hours)
    axes[1].set_xlabel('小时 (h)')
    axes[1].set_ylabel('预测记录数')
    axes[1].set_title('各时段预测数据量（反映通航活跃度）')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "hourly_weight_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("已生成: hourly_weight_heatmap.png")


# ============== 主入口 ==============
def main():
    log.info("开始生成附加图，输出目录: %s", IMG_DIR)
    viz = TopologyVisualizer()

    # A 类 6 张（复用 visualize.py）
    try:
        gen_class_a(viz)
    except Exception as e:
        log.error("A 类生成失败: %s", e)

    # B-7 数据概览
    try:
        gen_data_overview()
    except Exception as e:
        log.error("data_overview 失败: %s", e)

    # C-12 任务效率
    try:
        gen_task_efficiency()
    except Exception as e:
        log.error("task_efficiency 失败: %s", e)

    # D-14 碳排放
    try:
        gen_carbon_savings()
    except Exception as e:
        log.error("carbon_savings 失败: %s", e)

    # B-Plus: 4 张新图
    try:
        gen_robustness()
    except Exception as e:
        log.error("robustness 失败: %s", e)

    try:
        gen_task7_qa()
    except Exception as e:
        log.error("task7_qa 失败: %s", e)

    try:
        gen_node_type_distribution()
    except Exception as e:
        log.error("node_type_distribution 失败: %s", e)

    try:
        gen_hourly_heatmap()
    except Exception as e:
        log.error("hourly_weight_heatmap 失败: %s", e)

    # 列出产出
    pngs = sorted(IMG_DIR.glob("*.png"))
    log.info("=== 完成。共 %d 张图 ===", len(pngs))
    for p in pngs:
        log.info("  %s  (%.1f KB)", p.name, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()