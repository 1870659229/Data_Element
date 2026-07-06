# -*- coding: utf-8 -*-
"""生成 diff_routes.png — 不同船型路径差异对比图

在同一OD对上，分别用小型货船、大型油轮、渔船三种船型规划路径，
叠加绘制在拓扑网络底图上，展示物理约束对路径可行域的差异化裁剪效果。

运行: py -3.13 scripts/gen_diff_routes.py
"""
import os, sys, json, logging, random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "output"
IMG_DIR = OUTPUT_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("diff_routes")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 三种典型船型参数 (与 app.py SHIP_TEMPLATES 一致)
SHIP_TYPES = {
    '小型货船': {'length': 53, 'width': 11, 'draft': 2.6, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
    '大型油轮': {'length': 96, 'width': 16, 'draft': 5.894, 'height': 15, 'tonnage': 3572, 'max_speed': 11},
    '渔船':     {'length': 17, 'width': 4,  'draft': 1.5,  'height': 8,  'tonnage': 200,  'max_speed': 6},
}


def load_topology():
    """加载拓扑网络和边特征

    图结构基于 edge_features_dynamic_weights.csv（966条展开有向边，
    含双向边的两个方向），用于路径搜索和约束检查。
    拓扑边数（506）单独从 topology_edges.csv 读取，用于标题显示。
    """
    nodes_df = pd.read_csv(OUTPUT_DIR / "topology_nodes.csv")
    edge_feat_path = OUTPUT_DIR / "edge_features_dynamic_weights.csv"
    if not edge_feat_path.exists():
        edge_feat_path = OUTPUT_DIR / "topology_edges.csv"
    edge_feat_df = pd.read_csv(edge_feat_path)

    # 拓扑边数：从原始拓扑文件读取（506条有向边，非展开后的966条）
    topo_edge_count = len(pd.read_csv(OUTPUT_DIR / "topology_edges.csv"))

    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        nid = int(row['node_id'])
        G.add_node(nid, lat=row['lat'], lon=row['lon'],
                   frequency=row.get('frequency', 1))

    edge_features = {}
    for _, row in edge_feat_df.iterrows():
        fn = int(row.get('from_node', row.get('source', 0)))
        tn = int(row.get('to_node', row.get('target', 0)))
        feat = {k: row[k] for k in row.index if k not in ('from_node', 'to_node', 'source', 'target')}
        edge_features[(fn, tn)] = feat
        G.add_edge(fn, tn, weight=feat.get('avg_distance', feat.get('weight', 100)))

    return G, edge_features, topo_edge_count


def find_best_od_pair(navigator, constraint_checker, G, n_trials=20, seed=42):
    """搜索路径差异最大的OD对

    策略：
    1. 先用严格约束（去掉20%裕量）找大型油轮被剪枝的边
    2. 在差异边附近选OD对
    3. 如果严格约束下也无差异，则用常规约束搜索
    """
    from ship_navigator import ShipCharacteristics

    nodes = list(G.nodes())
    rng = random.Random(seed)

    # 尝试用严格约束找差异边
    large_tanker = ShipCharacteristics(
        ship_name='模板_大型油轮', ship_type='大型油轮',
        **SHIP_TYPES['大型油轮']
    )
    small_cargo = ShipCharacteristics(
        ship_name='模板_小型货船', ship_type='小型货船',
        **SHIP_TYPES['小型货船']
    )

    # 检查常规约束下的剪枝情况
    blocked_large = set(constraint_checker.get_blocked_edges(large_tanker))
    blocked_small = set(constraint_checker.get_blocked_edges(small_cargo))
    diff_edges = blocked_large - blocked_small
    log.info("常规约束: 大型油轮被剪枝边=%d, 小型货船被剪枝边=%d, 差异边=%d",
             len(blocked_large), len(blocked_small), len(diff_edges))

    # 如果常规约束无差异，手动检查严格约束（无20%裕量）
    if not diff_edges:
        strict_diff = []
        for edge_key, depth in constraint_checker.depth_map.items():
            # 严格约束：draft > depth (无1.2倍裕量)
            if large_tanker.draft > depth and small_cargo.draft <= depth:
                strict_diff.append(edge_key)
        log.info("严格约束(无裕量): 大型油轮独有剪枝边=%d", len(strict_diff))
        if strict_diff:
            diff_edges = set(strict_diff)

    # 收集差异边两端的节点作为候选起终点
    candidate_nodes = set()
    for (u, v) in diff_edges:
        candidate_nodes.add(u)
        candidate_nodes.add(v)
    candidate_nodes = list(candidate_nodes)
    log.info("差异边关联节点: %d", len(candidate_nodes))

    # 如果差异节点不足，用高频节点补充
    if len(candidate_nodes) < 10:
        high_freq = sorted(G.nodes(data=True),
                           key=lambda x: x[1].get('frequency', 0), reverse=True)
        for n, _ in high_freq[:30]:
            candidate_nodes.append(n)
        candidate_nodes = list(set(candidate_nodes))

    best_od = None
    best_score = -1

    for trial in range(n_trials):
        if len(candidate_nodes) >= 2:
            src, tgt = rng.sample(candidate_nodes, 2)
        else:
            src, tgt = rng.sample(nodes, 2)

        paths = {}
        path_results = {}
        for type_name, tpl in SHIP_TYPES.items():
            ship = ShipCharacteristics(
                ship_name=f'模板_{type_name}',
                ship_type=type_name,
                **tpl
            )
            result = navigator.find_paths(src, tgt, ship, hour=8, max_paths=1)
            if result:
                paths[type_name] = set(result[0].nodes)
                path_results[type_name] = result[0]

        if len(paths) < 2:
            continue

        # 差异度：Jaccard距离 + 属性差异
        type_names = list(paths.keys())
        diff_score = 0
        for i in range(len(type_names)):
            for j in range(i + 1, len(type_names)):
                s1, s2 = paths[type_names[i]], paths[type_names[j]]
                union = s1 | s2
                if union:
                    jaccard_dist = 1 - len(s1 & s2) / len(union)
                    diff_score += jaccard_dist * 10  # 放大节点差异权重

        # 补充属性差异：风险评分差异
        if len(path_results) >= 2:
            risks = [p.risk_score for p in path_results.values()]
            risk_range = max(risks) - min(risks)
            diff_score += risk_range / 100  # 风险差异贡献

        if diff_score > best_score:
            best_score = diff_score
            best_od = (src, tgt, paths, path_results)
            log.info("  试验 %d: OD %d→%d, 差异度=%.3f, 船型数=%d",
                     trial, src, tgt, diff_score, len(paths))

    return best_od, best_score


def plot_diff_routes(G, od_info, output_path, topo_edge_count):
    """绘制不同船型路径差异对比图（含 inset 局部放大）"""
    src, tgt, paths, path_results = od_info
    pos = {n: (d.get('lon', 0), d.get('lat', 0)) for n, d in G.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
    ax.set_facecolor('#F5F7FA')

    # ---- 1) 底层: 全网背景航道 ----
    for u, v in G.edges():
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#9E9E9E', linewidth=0.5, alpha=0.25, zorder=1)

    # ---- 2) 底层: 节点 (按 frequency 着色) ----
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    freqs = np.array([G.nodes[n].get('frequency', 1) for n in G.nodes()])
    sizes = 8 + 30 * (np.log1p(freqs) / (np.log1p(freqs.max()) + 1e-9))
    ax.scatter(xs, ys, s=sizes, c=freqs, cmap='YlOrBr',
               alpha=0.55, edgecolors='white', linewidths=0.4, zorder=2)

    # ---- 3) 三种船型路径 ----
    ship_colors = {
        '小型货船': '#1E88E5',   # 蓝色
        '大型油轮': '#E53935',   # 红色
        '渔船':     '#43A047',   # 绿色
    }
    ship_labels = {
        '小型货船': f'小型货船 (吃水2.6m)',
        '大型油轮': f'大型油轮 (吃水5.9m)',
        '渔船':     f'渔船 (吃水1.5m, 限高8m)',
    }

    legend_handles = []
    path_coords = {}  # 存储每条路径的坐标，供 inset 复用
    for type_name, node_set in paths.items():
        color = ship_colors[type_name]
        pr = path_results.get(type_name)
        if pr:
            dist_km = f"{pr.total_distance / 1000:.1f}km"
            risk = f"{pr.risk_score:.0f}"
            label = f'{ship_labels[type_name]}  距离{dist_km} 风险{risk}'
        else:
            label = ship_labels[type_name]
        node_list = sorted(node_set, key=lambda n: list(G.nodes()).index(n) if n in G else 0)

        path_nodes = []
        for n in node_list:
            if n in pos:
                path_nodes.append(n)

        if len(path_nodes) < 2:
            continue

        xs_path, ys_path = [], []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            xs_path += [pos[u][0], pos[v][0], None]
            ys_path += [pos[u][1], pos[v][1], None]

        path_coords[type_name] = (path_nodes, xs_path, ys_path, color)

        if xs_path:
            ax.plot(xs_path, ys_path, color=color, linewidth=8,
                    alpha=0.18, solid_capstyle='round', solid_joinstyle='round', zorder=3)
            ax.plot(xs_path, ys_path, color=color, linewidth=3.2,
                    alpha=0.95, solid_capstyle='round', solid_joinstyle='round', zorder=4)
            ax.scatter([pos[n][0] for n in path_nodes],
                       [pos[n][1] for n in path_nodes],
                       s=55, c=color, edgecolors='white', linewidths=1.4, zorder=5)

        legend_handles.append(Line2D([], [], color=color, linewidth=3.2,
                                     alpha=0.95, label=label))

    legend_handles.insert(0, Line2D([], [], color='#9E9E9E', linewidth=1,
                                    alpha=0.5, label='背景航道'))

    # ---- 4) 起终点标注 ----
    sx, sy, ex, ey = None, None, None, None
    if src in pos and tgt in pos:
        sx, sy = pos[src]
        ex, ey = pos[tgt]
        ax.scatter([sx], [sy], s=380, c='#D32F2F',
                   edgecolors='black', linewidths=1.6, zorder=7, marker='*')
        ax.scatter([ex], [ey], s=380, c='#FFD600',
                   edgecolors='black', linewidths=1.6, zorder=7, marker='*')
        ax.annotate('起点', (sx, sy), xytext=(-22, -18), textcoords='offset points',
                    fontsize=10.5, fontweight='bold', ha='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#D32F2F',
                              edgecolor='black', linewidth=0.8))
        ax.annotate('终点', (ex, ey), xytext=(-22, 22), textcoords='offset points',
                    fontsize=10.5, fontweight='bold', ha='center', color='#5D4037',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFD600',
                              edgecolor='black', linewidth=0.8))

    # ---- 5) 坐标轴格式化 ----
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}°E'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}°N'))
    ax.set_xlabel('经度 (East)', fontsize=12, fontweight='bold')
    ax.set_ylabel('纬度 (North)', fontsize=12, fontweight='bold')

    # ---- 6) 网格 ----
    ax.grid(True, linestyle='--', alpha=0.4, color='white', linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#455A64')
        spine.set_linewidth(1.2)

    # ---- 7) 图例 ----
    ax.legend(handles=legend_handles, fontsize=11, loc='lower right',
              title='船型路径', title_fontsize=12, framealpha=0.95,
              edgecolor='#37474F', fancybox=True, shadow=True)

    # ---- 8) 指北针 ----
    ax.annotate('N', xy=(0.04, 0.94), xycoords='axes fraction',
                fontsize=18, fontweight='bold', ha='center', color='#1A237E',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor='white',
                          edgecolor='#1A237E', linewidth=1.2))
    ax.annotate('', xy=(0.04, 0.99), xytext=(0.04, 0.87),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#1A237E', lw=2))

    # ---- 9) 比例尺 ----
    all_lons = np.array([p[0] for p in pos.values()])
    all_lats = np.array([p[1] for p in pos.values()])
    lon_range = all_lons.max() - all_lons.min()
    bar_len = lon_range * 0.05
    bar_x0 = all_lons.min() + 0.02 * lon_range
    bar_y0 = all_lats.min() + 0.03 * (all_lats.max() - all_lats.min())
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0],
            color='black', linewidth=3, zorder=10)
    ax.plot([bar_x0, bar_x0], [bar_y0 - 0.003, bar_y0 + 0.003],
            color='black', linewidth=1.5, zorder=10)
    ax.plot([bar_x0 + bar_len, bar_x0 + bar_len],
            [bar_y0 - 0.003, bar_y0 + 0.003],
            color='black', linewidth=1.5, zorder=10)
    km_approx = bar_len * 111
    ax.text(bar_x0 + bar_len / 2, bar_y0 + 0.005,
            f'≈ {km_approx:.1f} km', ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='black', linewidth=0.6))

    # ---- 10) 标题 ----
    ax.set_title(
        f'不同船型路径差异对比（{G.number_of_nodes()} 节点 / '
        f'{topo_edge_count} 边 · 3 种船型差异化路径）',
        fontsize=14, fontweight='bold', pad=14, color='#1A237E')

    # ==== 11) Inset 局部放大窗 ====
    if sx is not None and ex is not None:
        # inset 范围：以起终点为中心，各扩展 0.28°
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        hw = 0.30  # 半宽（经度）
        hh = 0.30  # 半高（纬度）
        inset_xlim = (cx - hw, cx + hw)
        inset_ylim = (cy - hh, cy + hh)

        # 主图上画红色矩形框标记 inset 区域
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (inset_xlim[0], inset_ylim[0]),
            inset_xlim[1] - inset_xlim[0],
            inset_ylim[1] - inset_ylim[0],
            linewidth=1.8, edgecolor='#D32F2F', facecolor='none',
            linestyle='--', zorder=6, alpha=0.7
        )
        ax.add_patch(rect)

        # 创建 inset axes（左下角空白区域，避免遮挡主图路径数据）
        ax_inset = fig.add_axes([0.05, 0.05, 0.35, 0.42])
        ax_inset.set_facecolor('#FAFAFA')

        # inset 内绘制局部背景航道
        for u, v in G.edges():
            if u in pos and v in pos:
                ux, uy = pos[u][0], pos[u][1]
                vx, vy = pos[v][0], pos[v][1]
                if inset_xlim[0] <= ux <= inset_xlim[1] and inset_ylim[0] <= uy <= inset_ylim[1]:
                    ax_inset.plot([ux, vx], [uy, vy],
                                  color='#BDBDBD', linewidth=0.8, alpha=0.35, zorder=1)

        # inset 内绘制三条路径
        for type_name, (path_nodes, xs_path, ys_path, color) in path_coords.items():
            if xs_path:
                # 阴影描边
                ax_inset.plot(xs_path, ys_path, color=color, linewidth=10,
                              alpha=0.15, solid_capstyle='round', solid_joinstyle='round', zorder=3)
                # 主线
                ax_inset.plot(xs_path, ys_path, color=color, linewidth=4.0,
                              alpha=0.95, solid_capstyle='round', solid_joinstyle='round', zorder=4)
                # 路径节点
                ax_inset.scatter([pos[n][0] for n in path_nodes if n in pos],
                                 [pos[n][1] for n in path_nodes if n in pos],
                                 s=70, c=color, edgecolors='white', linewidths=1.6, zorder=5)

        # inset 内起终点标注
        ax_inset.scatter([sx], [sy], s=500, c='#D32F2F',
                         edgecolors='black', linewidths=2, zorder=7, marker='*')
        ax_inset.scatter([ex], [ey], s=500, c='#FFD600',
                         edgecolors='black', linewidths=2, zorder=7, marker='*')
        ax_inset.annotate('起点', (sx, sy), xytext=(-26, -20), textcoords='offset points',
                           fontsize=12, fontweight='bold', ha='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.35', facecolor='#D32F2F',
                                     edgecolor='black', linewidth=1))
        ax_inset.annotate('终点', (ex, ey), xytext=(-26, 24), textcoords='offset points',
                           fontsize=12, fontweight='bold', ha='center', color='#5D4037',
                           bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFD600',
                                     edgecolor='black', linewidth=1))

        # inset 格式化
        ax_inset.set_xlim(inset_xlim)
        ax_inset.set_ylim(inset_ylim)
        ax_inset.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}°E'))
        ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}°N'))
        ax_inset.set_xlabel('经度 (East)', fontsize=10, fontweight='bold')
        ax_inset.set_ylabel('纬度 (North)', fontsize=10, fontweight='bold')
        ax_inset.grid(True, linestyle='--', alpha=0.35, color='#CCC', linewidth=0.8)
        ax_inset.set_axisbelow(True)
        for spine in ax_inset.spines.values():
            spine.set_color('#D32F2F')
            spine.set_linewidth(1.8)
        ax_inset.set_title('路径分叉区域放大', fontsize=11.5, fontweight='bold',
                            pad=8, color='#C62828')

        # inset 图例（简化版）
        inset_legend = [
            Line2D([], [], color='#1E88E5', linewidth=3.5, label='小型货船'),
            Line2D([], [], color='#E53935', linewidth=3.5, label='大型油轮'),
            Line2D([], [], color='#43A047', linewidth=3.5, label='渔船'),
        ]
        ax_inset.legend(handles=inset_legend, fontsize=9, loc='lower right',
                        framealpha=0.92, edgecolor='#D32F2F', fancybox=True)

    # 注意: 不使用 tight_layout，因为 fig.add_axes() 与之不兼容
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info("图片已保存: %s", output_path)


def main():
    from ship_navigator import MultiObjectiveNavigator, PhysicalConstraintChecker, ShipCharacteristics

    G, edge_features, topo_edge_count = load_topology()
    log.info("拓扑网络: %d 节点 / %d 边 (拓扑) / %d 边 (展开)", G.number_of_nodes(), topo_edge_count, G.number_of_edges())

    # 构建节点数据字典 (PhysicalConstraintChecker 需要)
    nodes_data = {}
    for nid, attrs in G.nodes(data=True):
        nodes_data[nid] = attrs

    # 构建 nav_edge_features (与 app.py 一致)
    nav_edge_features = {}
    for (u, v), feat in edge_features.items():
        nav_edge_features[(u, v)] = feat

    constraint_checker = PhysicalConstraintChecker(nav_edge_features, nodes_data, G)
    navigator = MultiObjectiveNavigator(G, nav_edge_features, constraint_checker)

    # 搜索路径差异最大的OD对
    log.info("搜索路径差异最大的OD对...")
    best_od, best_score = find_best_od_pair(navigator, constraint_checker, G, n_trials=30, seed=42)

    if best_od is None or best_score <= 0:
        log.warning("第一轮未找到有差异的OD对，扩大搜索...")
        best_od, best_score = find_best_od_pair(navigator, constraint_checker, G, n_trials=40, seed=123)

    if best_od is None:
        log.error("无法找到多种船型均有路径的OD对，退出")
        return

    src, tgt, paths, path_results = best_od
    log.info("最优OD对: %d → %d, 差异度=%.3f, 船型数=%d", src, tgt, best_score, len(paths))
    for type_name, node_set in paths.items():
        log.info("  %s: %d 个路径节点", type_name, len(node_set))

    # 绘图
    out_path = IMG_DIR / "diff_routes.png"
    plot_diff_routes(G, best_od, out_path, topo_edge_count)
    log.info("完成!")


if __name__ == "__main__":
    main()
