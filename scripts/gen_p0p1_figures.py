"""参赛报告 P0+P1 附加图批量生成脚本
==================================

从已有 output/ 下的 csv / json / pkl 衍生 9 张图,不重跑模型。

生成图:
  P0:
    1. architecture_overview.png     系统架构流程图
    2. residual_distribution.png     残差分布直方图 + Q-Q 图
    3. training_efficiency_bubble.png 训练效率气泡图(替代学习曲线)
    4. model_radar.png               7 模型 4 指标雷达图(替代多场景)
  P1:
    5. constraint_violation.png      物理约束违反率(按船型)
    6. constraint_visualization.png  物理约束可视化(地图叠加)
    7. failure_cases.png             预测失败案例分析
    8. code_structure.png            代码模块结构图
    9. config_example.png            配置与使用示例

运行: python scripts/gen_p0p1_figures.py
"""
import json
import os
import sys
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "output"
IMG_DIR = OUTPUT_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("gen_p0p1")


def load_metadata():
    with open(OUTPUT_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_pkl():
    pkl_path = OUTPUT_DIR / "weight_model_pna_stability_5seed.pkl"
    if not pkl_path.exists():
        pkl_path = OUTPUT_DIR / "weight_model_pna.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ======================== P0-1: 系统架构图 ========================
def gen_architecture_overview():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("系统架构总览", fontsize=16, fontweight='bold', pad=15)

    # 5 个主模块
    boxes = [
        (0.5, 2.5, "AIS 轨迹数据\n清洗与平滑", '#E3F2FD'),
        (3.3, 2.5, "航道拓扑\n节点提取与聚类", '#E8F5E9'),
        (6.1, 2.5, "动态耗时\n权重建模", '#FFF3E0'),
        (8.9, 2.5, "物理约束\n校验引擎", '#FCE4EC'),
        (11.7, 2.5, "多目标\n路径规划", '#F3E5F5'),
    ]
    bw, bh = 2.4, 2.0
    for (x, y, txt, color) in boxes:
        rect = mpatches.FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + bw/2, y + bh/2, txt, ha='center', va='center',
                fontsize=11, fontweight='bold', linespacing=1.4)

    # 箭头
    arrow_kw = dict(arrowstyle='->', color='#555', lw=2)
    for i in range(4):
        x_start = boxes[i][0] + bw
        x_end = boxes[i+1][0]
        y_mid = boxes[i][1] + bh/2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid), arrowprops=arrow_kw)

    # 子模块标注
    sub_labels = [
        (1.7, 2.2, "Douglas-Peucker\n异常漂移剔除"),
        (4.5, 2.2, "DBSCAN 聚类\n拐/岔/汇点识别"),
        (7.3, 2.2, "7 模型对比\nPNA 5-seed 集成"),
        (10.1, 2.2, "吃水/限高/宽度\n3 维约束检查"),
        (12.9, 2.2, "改进 A*/Dijkstra\n3 类差异化路径"),
    ]
    for (x, y, txt) in sub_labels:
        ax.text(x, y, txt, ha='center', va='top', fontsize=8, color='#666', style='italic')

    # 数据流标注
    flow_labels = [
        (1.7, 5.3, "原始 GPS"),
        (4.5, 5.3, "拓扑网络"),
        (7.3, 5.3, "边特征+权重"),
        (10.1, 5.3, "约束字典"),
        (12.9, 5.3, "2-3 条路径"),
    ]
    for (x, y, txt) in flow_labels:
        ax.text(x, y, txt, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#FBC02D'))

    out = IMG_DIR / "architecture_overview.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("架构图已保存: %s", out)


def _load_predictions_from_csv():
    """从 edge_features_dynamic_weights.csv 提取实际/预测耗时"""
    csv_path = OUTPUT_DIR / "edge_features_dynamic_weights.csv"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    y_true = df['avg_travel_time'].values
    # 用 24 小时预测均值作为模型预测
    pred_cols = [c for c in df.columns if c.startswith('predicted_time_h')]
    y_pred = df[pred_cols].mean(axis=1).values if pred_cols else None
    return y_true, y_pred


# ======================== P0-2: 残差分布图 ========================
def gen_residual_distribution():
    y_true, y_pred = _load_predictions_from_csv()
    if y_true is None or y_pred is None:
        log.warning("无法加载预测数据, 跳过残差图")
        return

    meta = load_metadata()
    comp = meta.get("model_comparison", {})

    # 主图: PNA 集成模型的残差
    residual = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图
    ax_hist = axes[0]
    ax_hist.hist(residual, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax_hist.set_title("PNA 集成模型残差分布", fontsize=12, fontweight='bold')
    ax_hist.set_xlabel("残差 (秒)")
    ax_hist.set_ylabel("频次")

    # Q-Q 图
    ax_qq = axes[1]
    stats.probplot(residual, dist="norm", plot=ax_qq)
    ax_qq.set_title("残差 Q-Q 图", fontsize=12, fontweight='bold')
    ax_qq.get_lines()[0].set_markerfacecolor('steelblue')
    ax_qq.get_lines()[0].set_markersize(3)

    fig.suptitle("模型残差分析", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "residual_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("残差分布图已保存: %s", out)


# ======================== P0-3: 训练效率气泡图 ========================
def gen_training_efficiency_bubble():
    meta = load_metadata()
    comp = meta.get("model_comparison", {})

    names, r2s, maes, times = [], [], [], []
    label_map = {
        'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
        'lightgbm_tweedie': 'LightGBM-Tweedie', 'random_forest': 'RandomForest',
        'ngboost': 'NGBoost', 'gnn': 'GAT', 'pna': 'PNA',
        'pna_stability_5seed': 'PNA-5seed',
    }
    for k, v in comp.items():
        names.append(label_map.get(k, k))
        r2s.append(v['r2'])
        maes.append(v['mae'])
        times.append(v['train_time'])

    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = [max(t * 8, 60) for t in times]  # 气泡大小 = 训练时间
    colors = plt.cm.RdYlGn(np.array(r2s) / max(r2s))

    scatter = ax.scatter(maes, r2s, s=sizes, c=colors, alpha=0.75, edgecolors='#333', linewidth=0.8)
    for i, name in enumerate(names):
        ax.annotate(name, (maes[i], r2s[i]), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel("MAE (秒)", fontsize=12)
    ax.set_ylabel("R2", fontsize=12)
    ax.set_title("模型效率-精度权衡 (气泡大小 = 训练时间)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图例:训练时间
    for t_val, t_label in [(5, '5s'), (15, '15s'), (30, '30s')]:
        ax.scatter([], [], s=max(t_val*8, 60), c='gray', alpha=0.5, edgecolors='#333',
                   label=f'训练 {t_label}')
    ax.legend(loc='lower left', title="训练时间", fontsize=9)

    out = IMG_DIR / "training_efficiency_bubble.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("训练效率气泡图已保存: %s", out)


# ======================== P0-4: 雷达图 ========================
def gen_model_radar():
    meta = load_metadata()
    comp = meta.get("model_comparison", {})

    # 4 个指标,归一化到 [0, 1]
    metrics = ['r2', 'mae', 'rmse', 'mape']
    metric_labels = ['R2', 'MAE', 'RMSE', 'MAPE(%)']
    # MAE/RMSE/MAPE 越小越好,取反
    all_vals = {m: [v[m] for v in comp.values()] for m in metrics}

    # 归一化: R2 越大越好, 其余越小越好
    norm_vals = {}
    for m in metrics:
        vals = all_vals[m]
        mn, mx = min(vals), max(vals)
        if mx == mn:
            norm_vals[m] = [1.0] * len(vals)
        elif m == 'r2':
            norm_vals[m] = [(v - mn) / (mx - mn) for v in vals]
        else:
            norm_vals[m] = [(mx - v) / (mx - mn) for v in vals]  # 反转

    label_map = {
        'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
        'lightgbm_tweedie': 'LightGBM-Tw', 'random_forest': 'RF',
        'ngboost': 'NGBoost', 'gnn': 'GAT', 'pna': 'PNA',
        'pna_stability_5seed': 'PNA-5seed',
    }
    # 只画 top-4
    sorted_keys = sorted(comp.keys(), key=lambda k: comp[k]['r2'], reverse=True)[:4]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

    for idx, key in enumerate(sorted_keys):
        values = [norm_vals[m][list(comp.keys()).index(key)] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label_map.get(key, key),
                color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Top-4 模型综合性能雷达图", fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    out = IMG_DIR / "model_radar.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("雷达图已保存: %s", out)


# ======================== P1-6: 物理约束违反率 ========================
def gen_constraint_violation():
    from ship_navigator import PhysicalConstraintChecker, ShipCharacteristics, ShipCharacteristicsManager

    # 加载边特征和节点
    edge_csv = OUTPUT_DIR / "edge_features_dynamic_weights.csv"
    if not edge_csv.exists():
        log.warning("边特征 CSV 不存在, 跳过约束违反率图")
        return

    # 加载拓扑
    nodes_csv = OUTPUT_DIR / "topology_nodes.csv"
    edges_csv = OUTPUT_DIR / "topology_edges.csv"
    if not nodes_csv.exists() or not edges_csv.exists():
        log.warning("拓扑文件不存在, 跳过约束违反率图")
        return

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    edge_feat_df = pd.read_csv(edge_csv)

    # 构建 graph
    import networkx as nx
    G = nx.DiGraph()
    nodes_dict = {}
    for _, row in nodes_df.iterrows():
        nid = int(row['node_id'])
        lat = row.get('latitude', row.get('lat', 0))
        lon = row.get('longitude', row.get('lon', 0))
        G.add_node(nid, lat=lat, lon=lon)
        nodes_dict[nid] = {'lat': lat, 'lon': lon}

    # 构建边特征字典
    edge_features = {}
    for _, row in edge_feat_df.iterrows():
        fn = int(row.get('from_node', row.get('source', 0)))
        tn = int(row.get('to_node', row.get('target', 0)))
        feat = {k: row[k] for k in row.index if k not in ('from_node', 'to_node', 'source', 'target')}
        edge_features[(fn, tn)] = feat
        G.add_edge(fn, tn)

    # 初始化约束检查器
    checker = PhysicalConstraintChecker(edge_features, nodes_dict, G)

    # 按船型模板统计
    templates = ShipCharacteristicsManager.SHIP_TEMPLATES
    ship_types = list(templates.keys())
    violation_counts = {'draft': [], 'height': [], 'width': []}
    total_edges = len(checker.depth_map)

    for stype, params in templates.items():
        ship = ShipCharacteristics(
            ship_name=stype,
            length=params['length'], width=params['width'],
            draft=params['draft'], height=params['height'],
            tonnage=params['tonnage'], max_speed=params['max_speed']
        )
        draft_v = sum(1 for ek, d in checker.depth_map.items()
                      if ship.draft > d * 1.2)
        height_v = sum(1 for ek, h in checker.height_map.items()
                       if ship.height > h * 1.2)
        width_v = sum(1 for ek, w in checker.width_map.items()
                      if ship.width > w * 1.2)
        violation_counts['draft'].append(draft_v / total_edges * 100)
        violation_counts['height'].append(height_v / total_edges * 100)
        violation_counts['width'].append(width_v / total_edges * 100)

    x = np.arange(len(ship_types))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, violation_counts['draft'], width, label='吃水超限', color='#E53935')
    ax.bar(x, violation_counts['height'], width, label='高度超限', color='#1E88E5')
    ax.bar(x + width, violation_counts['width'], width, label='宽度超限', color='#43A047')

    ax.set_xticks(x)
    ax.set_xticklabels(ship_types, fontsize=9, rotation=20, ha='right')
    ax.set_ylabel("违反率 (%)")
    ax.set_title("各船型物理约束违反率", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    out = IMG_DIR / "constraint_violation.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("约束违反率图已保存: %s", out)


# ======================== P1-7: 物理约束可视化 ========================
def gen_constraint_visualization():
    from ship_navigator import PhysicalConstraintChecker, ShipCharacteristics, ShipCharacteristicsManager

    nodes_csv = OUTPUT_DIR / "topology_nodes.csv"
    edges_csv = OUTPUT_DIR / "topology_edges.csv"
    edge_csv = OUTPUT_DIR / "edge_features_dynamic_weights.csv"

    if not all(p.exists() for p in [nodes_csv, edges_csv, edge_csv]):
        log.warning("拓扑/边特征文件缺失, 跳过约束可视化")
        return

    import networkx as nx
    nodes_df = pd.read_csv(nodes_csv)
    edge_feat_df = pd.read_csv(edge_csv)

    G = nx.DiGraph()
    nodes_dict = {}
    for _, row in nodes_df.iterrows():
        nid = int(row['node_id'])
        lat = row.get('latitude', row.get('lat', 0))
        lon = row.get('longitude', row.get('lon', 0))
        G.add_node(nid, lat=lat, lon=lon)
        nodes_dict[nid] = {'lat': lat, 'lon': lon}

    edge_features = {}
    for _, row in edge_feat_df.iterrows():
        fn = int(row.get('from_node', row.get('source', 0)))
        tn = int(row.get('to_node', row.get('target', 0)))
        feat = {k: row[k] for k in row.index if k not in ('from_node', 'to_node', 'source', 'target')}
        edge_features[(fn, tn)] = feat
        G.add_edge(fn, tn)

    checker = PhysicalConstraintChecker(edge_features, nodes_dict, G)

    # 分类边:浅水/限高/窄航/正常
    shallow_edges, low_edges, narrow_edges, normal_edges = [], [], [], []
    for ek in checker.depth_map:
        fn, tn = ek
        if fn not in nodes_dict or tn not in nodes_dict:
            continue
        lat1, lon1 = nodes_dict[fn]['lat'], nodes_dict[fn]['lon']
        lat2, lon2 = nodes_dict[tn]['lat'], nodes_dict[tn]['lon']
        is_shallow = checker.depth_map[ek] < 8
        is_low = checker.height_map.get(ek, 100) < 30
        is_narrow = checker.width_map.get(ek, 100) < 60

        if is_shallow:
            shallow_edges.append(([lon1, lon2], [lat1, lat2]))
        elif is_low:
            low_edges.append(([lon1, lon2], [lat1, lat2]))
        elif is_narrow:
            narrow_edges.append(([lon1, lon2], [lat1, lat2]))
        else:
            normal_edges.append(([lon1, lon2], [lat1, lat2]))

    fig, ax = plt.subplots(figsize=(10, 10))
    for xs, ys in normal_edges:
        ax.plot(xs, ys, color='#BDBDBD', linewidth=0.5, alpha=0.5)
    for xs, ys in narrow_edges:
        ax.plot(xs, ys, color='#FB8C00', linewidth=1.5, alpha=0.7, label='_nolegend_')
    for xs, ys in low_edges:
        ax.plot(xs, ys, color='#1E88E5', linewidth=1.5, alpha=0.7, label='_nolegend_')
    for xs, ys in shallow_edges:
        ax.plot(xs, ys, color='#E53935', linewidth=2, alpha=0.8, label='_nolegend_')

    # 图例
    ax.plot([], [], color='#BDBDBD', linewidth=2, label='正常航道')
    ax.plot([], [], color='#FB8C00', linewidth=2, label='窄航段 (<60m)')
    ax.plot([], [], color='#1E88E5', linewidth=2, label='限高段 (<30m)')
    ax.plot([], [], color='#E53935', linewidth=2, label='浅水段 (<8m)')

    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("航道物理约束分布", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')

    out = IMG_DIR / "constraint_visualization.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("约束可视化图已保存: %s", out)


# ======================== P1-8: 失败案例分析 ========================
def gen_failure_cases():
    y_true, y_pred = _load_predictions_from_csv()
    if y_true is None or y_pred is None:
        log.warning("无法加载预测数据, 跳过失败案例图")
        return

    abs_err = np.abs(y_true - y_pred)
    top_idx = np.argsort(abs_err)[-6:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, idx in enumerate(top_idx):
        ax = axes[i]
        actual = y_true[idx]
        predicted = y_pred[idx]
        err = abs_err[idx]

        ax.barh(['预测', '实际'], [predicted, actual],
                color=['#1E88E5', '#43A047'], height=0.5)
        ax.set_title(f"路段 {idx}: 误差={err:.1f}s", fontsize=10)
        ax.set_xlabel("耗时 (秒)")

        pct = err / actual * 100 if actual != 0 else 0
        ax.text(max(predicted, actual) * 0.5, -0.3,
                f"偏差 {pct:.1f}%", ha='center', fontsize=9, color='red')

    fig.suptitle("Top-6 预测误差路段 (PNA 集成模型)", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "failure_cases.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("失败案例图已保存: %s", out)


# ======================== P1-9: 代码模块结构图 ========================
def gen_code_structure():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("代码模块结构", fontsize=16, fontweight='bold', pad=15)

    modules = [
        # (x, y, name, desc, color)
        (0.5, 5.5, "main.py", "入口 + 流程编排", '#E3F2FD'),
        (3.5, 5.5, "topology_builder.py", "轨迹清洗 + 拓扑提取", '#E8F5E9'),
        (6.5, 5.5, "advanced_weight_model.py", "7 模型训练 + PNA 集成", '#FFF3E0'),
        (9.5, 5.5, "ship_navigator.py", "约束校验 + 路径规划", '#FCE4EC'),
        (0.5, 2.5, "utils.py", "地理计算工具", '#F5F5F5'),
        (3.5, 2.5, "navigation_models.py", "风险/通行概率模型", '#F5F5F5'),
        (6.5, 2.5, "visualize.py", "21 张图可视化", '#F5F5F5'),
        (9.5, 2.5, "config.py", "全局配置", '#F5F5F5'),
    ]

    bw, bh = 2.6, 1.6
    for (x, y, name, desc, color) in modules:
        rect = mpatches.FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='#333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + bw/2, y + bh*0.65, name, ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.text(x + bw/2, y + bh*0.3, desc, ha='center', va='center',
                fontsize=8, color='#666')

    # 依赖箭头: main -> 3 个核心模块
    arrow_kw = dict(arrowstyle='->', color='#555', lw=1.5)
    main_cx, main_cy = 0.5 + bw/2, 5.5 + bh/2
    for target_x in [3.5, 6.5, 9.5]:
        tcx = target_x + bw/2
        ax.annotate('', xy=(tcx, main_cy), xytext=(0.5 + bw, main_cy), arrowprops=arrow_kw)

    # 核心模块 -> 工具模块
    for src_x, tgt_x in [(3.5, 0.5), (6.5, 3.5), (9.5, 6.5), (9.5, 9.5)]:
        sx = src_x + bw/2
        tx = tgt_x + bw/2
        ax.annotate('', xy=(tx, 2.5 + bh), xytext=(sx, 5.5), arrowprops=arrow_kw)

    out = IMG_DIR / "code_structure.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("代码结构图已保存: %s", out)


# ======================== P1-10: 配置示例 ========================
def gen_config_example():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title("配置与使用示例", fontsize=14, fontweight='bold', pad=15)

    config_text = """# ===== 典型使用流程 =====

# 1. 训练模型 (自动 7 模型对比 + PNA 5-seed 稳定性验证)
python advanced_weight_model.py

# 2. 路径规划 (输入: 起终点 GPS; 输出: 2-3 条差异化路径)
python ship_navigator.py --start 121.47,31.23 --end 121.85,31.35

# 3. 生成可视化 (21 张图 + 9 张附加图)
python visualize.py
python scripts/gen_p0p1_figures.py

# ===== 关键配置 (config.py) =====
GRID_SEARCH = True          # Optuna 贝叶斯调参
GNN_ARCH = 'pna'            # GNN 架构: gat / pna
STABILITY_SEEDS = [42,123,456,789,1011]  # 5-seed 验证
EARLY_STOP_PATIENCE = 25    # GNN 早停耐心值
DROP_EDGE_P = 0.1           # GNN 边丢弃正则化"""

    ax.text(0.05, 0.95, config_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

    out = IMG_DIR / "config_example.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("配置示例图已保存: %s", out)


# ======================== 主入口 ========================
def main():
    log.info("=" * 50)
    log.info("开始生成 P0+P1 附加图 (9 张)")
    log.info("=" * 50)

    generators = [
        ("P0-1: 系统架构图", gen_architecture_overview),
        ("P0-2: 残差分布图", gen_residual_distribution),
        ("P0-3: 训练效率气泡图", gen_training_efficiency_bubble),
        ("P0-4: 雷达图", gen_model_radar),
        ("P1-6: 约束违反率", gen_constraint_violation),
        ("P1-7: 约束可视化", gen_constraint_visualization),
        ("P1-8: 失败案例", gen_failure_cases),
        ("P1-9: 代码结构图", gen_code_structure),
        ("P1-10: 配置示例", gen_config_example),
    ]

    ok, fail = 0, 0
    for label, func in generators:
        try:
            log.info(">>> %s", label)
            func()
            ok += 1
        except Exception as e:
            log.error("!!! %s 失败: %s", label, e)
            fail += 1

    log.info("=" * 50)
    log.info("完成: %d 成功, %d 失败", ok, fail)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
