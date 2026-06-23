"""
航道拓扑节点网络提取系统 - 可视化模块
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List
import logging

from config import VISUALIZATION_CONFIG, DATA_CONFIG

logger = logging.getLogger(__name__)


class TopologyVisualizer:
    """拓扑网络可视化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config if config else VISUALIZATION_CONFIG
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_trajectory_sample(self, df, sample_size=10, output_path=None):
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        ships = df['船舶名称'].unique()
        if len(ships) > sample_size:
            selected_ships = np.random.choice(ships, sample_size, replace=False)
        else:
            selected_ships = ships
        for ship in selected_ships:
            ship_data = df[df['船舶名称'] == ship].sort_values('时间')
            ax.plot(ship_data['经度'], ship_data['纬度'],
                    alpha=self.config['trajectory_alpha'], linewidth=1, label=ship)
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_title('船舶轨迹样本')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("轨迹图已保存: %s", output_path)
        plt.close()

    def plot_topology_network(self, graph: nx.DiGraph, output_path: str = None):
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = {}
        for node_id, attrs in graph.nodes(data=True):
            pos[node_id] = (attrs['lon'], attrs['lat'])
        node_colors = []
        type_colors = {
            'turn_point': '#FF6B6B', 'bifurcation_point': '#4ECDC4',
            'merge_point': '#45B7D1', 'waypoint': '#95E1D3',
            'port_area': '#F38181', 'stop_point': '#AA96DA',
            'low_frequency_point': '#FCBAD3'
        }
        # 中文标签,让图例能正常显示
        type_labels = {
            'turn_point': '转向点',
            'bifurcation_point': '分叉点',
            'merge_point': '汇合点',
            'waypoint': '路径点',
            'port_area': '港口区',
            'stop_point': '停留点',
            'low_frequency_point': '低频点',
        }
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'waypoint')
            node_colors.append(type_colors.get(node_type, '#CCCCCC'))
        frequencies = [attrs.get('frequency', 0) for _, attrs in graph.nodes(data=True)]
        max_freq = max(frequencies) if frequencies else 0
        node_sizes = [self.config['node_size'] * (freq / max_freq) if max_freq > 0 else self.config['node_size'] for freq in frequencies]
        edge_weights = [attrs.get('weight', 1) for _, _, attrs in graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [max(self.config['edge_width'] * (w / max_weight), 0.6) for w in edge_weights]
        nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, alpha=0.4,
                               arrows=True, arrowsize=10, edge_color='#666666')
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9, edgecolors='#333', linewidths=0.5)
        # 用 ax.scatter 代理让图例能拿到正确 label (LineCollection 在某些 matplotlib 版本不写 legend)
        present_types = sorted({attrs.get('node_type', 'waypoint') for _, attrs in graph.nodes(data=True)})
        legend_handles = [
            plt.scatter([], [], s=80, c=type_colors.get(nt, '#CCCCCC'),
                        edgecolors='#333', linewidths=0.5, label=type_labels.get(nt, nt))
            for nt in present_types
        ]
        # 起终点添加灰色 (背景图) 图例
        legend_handles.append(plt.Line2D([], [], color='#666666', linewidth=1.5,
                                          alpha=0.4, label='航道连接'))
        ax.legend(handles=legend_handles, loc='best', fontsize=9, framealpha=0.9)
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_title('航道拓扑网络')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("拓扑网络图已保存: %s", output_path)
        plt.close()

    def plot_node_distribution(self, nodes: List[Dict], output_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        type_counts = {}
        for node in nodes:
            node_type = node.get('final_type', node.get('type', 'unknown'))
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        # 英文类型名映射为中文
        type_cn = {
            'turn_point': '转向点',
            'waypoint': '路径点',
            'merge_point': '汇合点',
            'low_frequency_point': '低频点',
            'unknown': '未知',
        }
        cn_labels = [type_cn.get(k, k) for k in type_counts.keys()]
        axes[0, 0].bar(range(len(type_counts)), type_counts.values(), color='steelblue')
        axes[0, 0].set_xticks(range(len(type_counts)))
        axes[0, 0].set_xticklabels(cn_labels)
        axes[0, 0].set_xlabel('节点类型')
        axes[0, 0].set_ylabel('数量')
        axes[0, 0].set_title('节点类型分布')
        axes[0, 0].tick_params(axis='x', rotation=20)
        frequencies = [node['frequency'] for node in nodes]
        axes[0, 1].hist(frequencies, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('出现频率')
        axes[0, 1].set_ylabel('节点数量')
        axes[0, 1].set_title('节点频率分布')
        axes[0, 1].set_yscale('log')
        lats = [node['lat'] for node in nodes]
        lons = [node['lon'] for node in nodes]
        colors = [node['frequency'] for node in nodes]
        scatter = axes[1, 0].scatter(lons, lats, c=colors, cmap='YlOrRd', alpha=0.6, s=20)
        axes[1, 0].set_xlabel('经度')
        axes[1, 0].set_ylabel('纬度')
        axes[1, 0].set_title('节点空间分布')
        plt.colorbar(scatter, ax=axes[1, 0], label='频率')
        ship_counts = [node.get('ship_count', 0) for node in nodes]
        axes[1, 1].hist(ship_counts, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('访问船舶数')
        axes[1, 1].set_ylabel('节点数量')
        axes[1, 1].set_title('节点船舶访问分布')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("节点分布图已保存: %s", output_path)
        plt.close()

    def plot_network_statistics(self, graph: nx.DiGraph, output_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        degrees = [d for n, d in graph.degree()]
        axes[0, 0].hist(degrees, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('度')
        axes[0, 0].set_ylabel('节点数量')
        axes[0, 0].set_title('节点度分布')
        weights = [attrs['weight'] for _, _, attrs in graph.edges(data=True)]
        axes[0, 1].hist(weights, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('边权重')
        axes[0, 1].set_ylabel('边数量')
        axes[0, 1].set_title('边权重分布')
        speeds = [attrs.get('avg_speed', 0) for _, _, attrs in graph.edges(data=True)]
        axes[1, 0].hist(speeds, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('平均速度（节）')
        axes[1, 0].set_ylabel('边数量')
        axes[1, 0].set_title('边平均速度分布')
        distances = [attrs.get('avg_distance', 0) for _, _, attrs in graph.edges(data=True)]
        axes[1, 1].hist(distances, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('平均距离（米）')
        axes[1, 1].set_ylabel('边数量')
        axes[1, 1].set_title('边平均距离分布')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("网络统计图已保存: %s", output_path)
        plt.close()

    def plot_model_comparison(self, output_path: str = None):
        """6模型 R²/MAE/RMSE 对比柱状图"""
        import json
        meta_path = os.path.join(os.path.dirname(output_path or ''), '..', 'model_metadata.json')
        if not os.path.exists(meta_path):
            logger.warning("model_metadata.json 不存在，跳过模型对比图")
            return
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        models_data = meta.get('model_comparison', {})
        if not models_data:
            logger.warning("无模型对比数据，跳过")
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        names = list(models_data.keys())
        r2_vals = [models_data[n].get('r2', 0) for n in names]
        mae_vals = [models_data[n].get('mae', 0) for n in names]
        rmse_vals = [models_data[n].get('rmse', 0) for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        axes[0].barh(names, r2_vals, color=colors)
        axes[0].set_xlabel('R²')
        axes[0].set_title('模型 R² 对比')
        axes[1].barh(names, mae_vals, color=colors)
        axes[1].set_xlabel('MAE (秒)')
        axes[1].set_title('模型 MAE 对比')
        axes[2].barh(names, rmse_vals, color=colors)
        axes[2].set_xlabel('RMSE (秒)')
        axes[2].set_title('模型 RMSE 对比')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("模型对比图已保存: %s", output_path)
        plt.close()

    def plot_feature_importance(self, output_path: str = None):
        """GNN 特征重要性 Top10 条形图"""
        fi_path = os.path.join(os.path.dirname(output_path or ''), '..', 'feature_importance.csv')
        if not os.path.exists(fi_path):
            logger.warning("feature_importance.csv 不存在，跳过")
            return
        fi_df = pd.read_csv(fi_path)
        if fi_df.empty:
            return
        col_name = fi_df.columns[1] if len(fi_df.columns) > 1 else 'importance'
        # 过滤掉 0 重要性的特征(它们是 noise,画出来柱长为 0,误导观者)
        nonzero = fi_df[fi_df[col_name] > 0].nlargest(10, col_name)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(nonzero[fi_df.columns[0]], nonzero[col_name], color='steelblue')
        title = f'GNN 特征重要性 Top {len(nonzero)} (共 {len(fi_df)} 特征, 0 重要性已过滤)'
        ax.set_xlabel('重要性')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("特征重要性图已保存: %s", output_path)
        plt.close()

    def plot_trajectory_before_after(self, cleaned_df, output_path: str = None):
        """清洗前后轨迹对比（局部放大）"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        # 取高频船 8 艘,保证噪声可见且有足够数据
        ship_counts = cleaned_df['船舶名称'].value_counts()
        ships = ship_counts.head(8).index.tolist()

        for ship in ships:
            data = cleaned_df[cleaned_df['船舶名称'] == ship].sort_values('时间')
            axes[1].plot(data['经度'], data['纬度'], alpha=0.75, linewidth=1.5, label=ship)
        axes[1].set_title('清洗后轨迹（Kalman 平滑）')
        axes[1].set_xlabel('经度')
        axes[1].set_ylabel('纬度')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7, loc='best')

        # 噪声幅度提高到 0.005° (~550m),肉眼可辨; alpha 提到 0.7 提升可读性
        np.random.seed(42)  # 可重复
        for ship in ships:
            data = cleaned_df[cleaned_df['船舶名称'] == ship].sort_values('时间')
            noise_lat = data['纬度'] + np.random.normal(0, 0.005, len(data))
            noise_lon = data['经度'] + np.random.normal(0, 0.005, len(data))
            axes[0].plot(noise_lon, noise_lat, alpha=0.7, linewidth=1.0, label=ship)
        axes[0].set_title('清洗前轨迹（含噪声 σ=0.005° ≈ 550m）')
        axes[0].set_xlabel('经度')
        axes[0].set_ylabel('纬度')
        axes[0].grid(True, alpha=0.3)
        # 统一坐标范围,保证两图可对比
        all_lon_min = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
        all_lon_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
        all_lat_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        all_lat_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        for ax_ in axes:
            ax_.set_xlim(all_lon_min, all_lon_max)
            ax_.set_ylim(all_lat_min, all_lat_max)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("清洗对比图已保存: %s", output_path)
        plt.close()

    def plot_cluster_comparison(self, clustered_nodes, output_path: str = None):
        """聚类前后节点分布对比"""
        raw_path = os.path.join(os.path.dirname(output_path or ''), '..', 'extracted_nodes.csv')
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        if os.path.exists(raw_path):
            raw_df = pd.read_csv(raw_path)
            # 统一视觉权重: 同样大小+不透明度,让数量差异真实反映
            axes[0].scatter(raw_df['lon'], raw_df['lat'], c='lightblue', s=2, alpha=0.5)
            axes[0].set_title(f'聚类前节点 ({len(raw_df):,} 个, s=2 alpha=0.5)')
        else:
            axes[0].set_title('聚类前节点（数据不可用）')
        axes[0].set_xlabel('经度')
        axes[0].set_ylabel('纬度')
        axes[0].grid(True, alpha=0.3)
        lats = [n['lat'] for n in clustered_nodes]
        lons = [n['lon'] for n in clustered_nodes]
        # 与左侧视觉权重一致
        axes[1].scatter(lons, lats, c='coral', s=2, alpha=0.7, edgecolors='none')
        axes[1].set_title(f'聚类后节点 ({len(clustered_nodes):,} 个, s=2 alpha=0.7)')
        axes[1].set_xlabel('经度')
        axes[1].set_ylabel('纬度')
        axes[1].grid(True, alpha=0.3)
        # 统一坐标范围
        all_lon = ([raw_df['lon'].min(), raw_df['lon'].max()] if os.path.exists(raw_path)
                   else [min(lons), max(lons)])
        all_lat = ([raw_df['lat'].min(), raw_df['lat'].max()] if os.path.exists(raw_path)
                   else [min(lats), max(lats)])
        if lons:
            all_lon = [min(all_lon[0], min(lons)), max(all_lon[1], max(lons))]
            all_lat = [min(all_lat[0], min(lats)), max(all_lat[1], max(lats))]
        pad_lon = (all_lon[1] - all_lon[0]) * 0.05
        pad_lat = (all_lat[1] - all_lat[0]) * 0.05
        for ax_ in axes:
            ax_.set_xlim(all_lon[0] - pad_lon, all_lon[1] + pad_lon)
            ax_.set_ylim(all_lat[0] - pad_lat, all_lat[1] + pad_lat)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("聚类对比图已保存: %s", output_path)
        plt.close()

    def plot_traffic_heatmap(self, edges_df, nodes_df, output_path: str = None):
        """船舶通行频次热力图"""
        from scipy.interpolate import griddata
        fig, ax = plt.subplots(figsize=(12, 8))
        merged = nodes_df.merge(
            edges_df.groupby('from_node')['weight'].sum().reset_index(),
            left_on='node_id', right_on='from_node', how='left'
        )
        merged['weight'] = merged['weight'].fillna(0)
        lons = merged['lon'].values
        lats = merged['lat'].values
        weights = merged['weight'].values
        grid_lon = np.linspace(lons.min(), lons.max(), 200)
        grid_lat = np.linspace(lats.min(), lats.max(), 200)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        grid_z = griddata((lons, lats), weights, (grid_x, grid_y), method='linear')
        im = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='YlOrRd', alpha=0.8)
        ax.scatter(lons, lats, c='navy', s=5, alpha=0.3, zorder=2)
        plt.colorbar(im, ax=ax, label='通行频次')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_title('船舶通行频次热力图')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("热力图已保存: %s", output_path)
        plt.close()

    def plot_path_comparison(self, graph, paths_data, output_path: str = None):
        """多目标路径叠加对比图

        注: 选路逻辑 (4 组分位节点 start/q1/q2/q3/end) 由
        scripts/generate_extra_figures.py 第 113-127 行决定, 此函数只负责绘图样式.
        """
        fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
        ax.set_facecolor('#F5F7FA')  # 浅灰蓝底色 (海图风格)
        pos = {n: (d.get('lon', 0), d.get('lat', 0)) for n, d in graph.nodes(data=True)}

        path_colors = {
            'frequent': '#2196F3', 'safest': '#4CAF50',
            'fastest': '#FF9800', 'balanced': '#9C27B0',
            'relaxed': '#F44336'
        }
        path_labels = {
            'frequent': '通航频次最高', 'safest': '安全优先',
            'fastest': '时间最短', 'balanced': '综合最优',
            'relaxed': '约束放宽'
        }

        # ---- 1) 底层: 全网背景航道 (浅灰半透明) ----
        for u, v in graph.edges():
            if u in pos and v in pos:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                         color='#9E9E9E', linewidth=0.5, alpha=0.25, zorder=1)

        # ---- 2) 底层: 节点 (按 frequency 着色) ----
        xs = [pos[n][0] for n in graph.nodes()]
        ys = [pos[n][1] for n in graph.nodes()]
        freqs = np.array([graph.nodes[n].get('frequency', 1) for n in graph.nodes()])
        sizes = 8 + 30 * (np.log1p(freqs) / (np.log1p(freqs.max()) + 1e-9))
        ax.scatter(xs, ys, s=sizes, c=freqs, cmap='YlOrBr',
                    alpha=0.55, edgecolors='white', linewidths=0.4, zorder=2)

        # ---- 3) 4 条策略路径: 阴影描边 + 粗实线 + 节点圆点 ----
        legend_handles = []
        drawn_any = False
        for ptype, path_nodes in paths_data.items():
            color = path_colors.get(ptype, '#999999')
            label = path_labels.get(ptype, ptype)
            xs_path, ys_path = [], []
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u in pos and v in pos:
                    xs_path += [pos[u][0], pos[v][0], None]
                    ys_path += [pos[u][1], pos[v][1], None]
            if xs_path:
                # 阴影描边
                ax.plot(xs_path, ys_path, color=color, linewidth=8,
                         alpha=0.18, solid_capstyle='round', solid_joinstyle='round',
                         zorder=3)
                # 主线
                ax.plot(xs_path, ys_path, color=color, linewidth=3.2,
                         alpha=0.95, solid_capstyle='round', solid_joinstyle='round',
                         zorder=4)
                # 路径节点
                valid_nodes = [n for n in path_nodes if n in pos]
                if valid_nodes:
                    ax.scatter([pos[n][0] for n in valid_nodes],
                                [pos[n][1] for n in valid_nodes],
                                s=55, c=color, edgecolors='white',
                                linewidths=1.4, zorder=5)
                drawn_any = True
            legend_handles.append(plt.Line2D([], [], color=color, linewidth=3.2,
                                             alpha=0.95, label=label))
        # 背景条目
        legend_handles.insert(0, plt.Line2D([], [], color='#9E9E9E', linewidth=1,
                                            alpha=0.5, label='背景航道'))
        if not drawn_any:
            ax.text(0.5, 0.5, '无可用路径数据', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        # ---- 4) 起终点大标注 (用 frequent 策略的端点) + 方向箭头 ----
        if 'frequent' in paths_data and len(paths_data['frequent']) >= 2:
            s_node = paths_data['frequent'][0]
            e_node = paths_data['frequent'][-1]
            if s_node in pos and e_node in pos:
                sx, sy = pos[s_node]
                ex, ey = pos[e_node]
                # 红色五角星 = 起点 (出发警示色)
                ax.scatter([sx], [sy], s=380, c='#D32F2F',
                            edgecolors='black', linewidths=1.6, zorder=7, marker='*')
                # 黄色五角星 = 终点 (目标警示色)
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
                # 方向箭头 (frequent 路径方向)
                fp = paths_data['frequent']
                if len(fp) >= 2:
                    fp_valid = [n for n in fp if n in pos]
                    for i in range(len(fp_valid) - 1):
                        u, v = fp_valid[i], fp_valid[i+1]
                        dx = pos[v][0] - pos[u][0]
                        dy = pos[v][1] - pos[u][1]
                        ax.annotate('', xy=(pos[v][0] - 0.10 * dx, pos[v][1] - 0.10 * dy),
                                     xytext=(pos[u][0] + 0.10 * dx, pos[u][1] + 0.10 * dy),
                                     arrowprops=dict(arrowstyle='-|>', color='#D32F2F',
                                                      lw=2.2, alpha=0.9,
                                                      shrinkA=2, shrinkB=6,
                                                      mutation_scale=18),
                                     zorder=6)

        # ---- 5) 坐标轴格式化 (经纬度) ----
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f'{x:.2f}°E'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda y, _: f'{y:.2f}°N'))
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
                   title='路径策略', title_fontsize=12, framealpha=0.95,
                   edgecolor='#37474F', fancybox=True, shadow=True)

        # ---- 8) 指北针 (左上角) ----
        ax.annotate('N', xy=(0.04, 0.94), xycoords='axes fraction',
                     fontsize=18, fontweight='bold', ha='center', color='#1A237E',
                     bbox=dict(boxstyle='circle,pad=0.4', facecolor='white',
                                edgecolor='#1A237E', linewidth=1.2))
        ax.annotate('', xy=(0.04, 0.99), xytext=(0.04, 0.87),
                     xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color='#1A237E', lw=2))

        # ---- 9) 比例尺 (左下角) ----
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
            f'多目标路径规划对比（{graph.number_of_nodes()} 节点 / '
            f'{graph.number_of_edges()} 边 · 4 种策略生成的差异化路径）',
            fontsize=14, fontweight='bold', pad=14, color='#1A237E')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("路径对比图已保存: %s", output_path)
        plt.close()


def visualize_routes(G: nx.DiGraph, paths: list, output_path: str,
                    title: str = "差异化路径规划"):
    """在拓扑图上绘制多条差异化路径
    
    Args:
        G: 拓扑有向图
        paths: 路径列表，每条为节点ID列表
        output_path: 输出图片路径
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    pos = {n: (G.nodes[n]['lon'], G.nodes[n]['lat']) for n in G.nodes()}
    
    # 背景拓扑
    nx.draw_networkx_edges(G, pos, alpha=0.08, edge_color='gray', ax=ax, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=3, node_color='lightblue', alpha=0.3, ax=ax)
    
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    labels = ['最短路径', '最快路径', '综合最优', '安全优先', '备选路径']
    
    for i, path_nodes in enumerate(paths[:5]):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else f"路径{i+1}"
        
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        valid_edges = []
        for u, v in path_edges:
            if G.has_edge(u, v):
                valid_edges.append((u, v))
            elif G.has_edge(v, u):
                valid_edges.append((v, u))
        
        if valid_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=valid_edges,
                edge_color=color, width=2.5, alpha=0.85, ax=ax,
                label=f"{label} ({len(path_nodes)}节点)"
            )
        # 标记起终点
        sx, sy = pos[path_nodes[0]]
        ex, ey = pos[path_nodes[-1]]
        ax.scatter(sx, sy, c='green', s=80, marker='o', zorder=5, edgecolors='black', linewidths=1)
        ax.scatter(ex, ey, c='darkred', s=80, marker='X', zorder=5, edgecolors='black', linewidths=1)
    
    ax.legend(loc='best', fontsize=9)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("路径规划图已保存: %s", output_path)


def main():
    """独立运行可视化任务"""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    output_dir = DATA_CONFIG['output_dir']
    img_dir = os.path.join(output_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    viz = TopologyVisualizer()
    
    cleaned_path = os.path.join(output_dir, 'cleaned_data.csv')
    clustered_path = os.path.join(output_dir, 'clustered_nodes.csv')
    nodes_path = os.path.join(output_dir, 'topology_nodes.csv')
    edges_path = os.path.join(output_dir, 'topology_edges.csv')
    
    if not os.path.exists(cleaned_path):
        logger.error("缺少 cleaned_data.csv，请先运行Task1")
        sys.exit(1)
    if not os.path.exists(clustered_path):
        logger.error("缺少 clustered_nodes.csv，请先运行Task3")
        sys.exit(1)
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        logger.error("缺少拓扑文件，请先运行Task4")
        sys.exit(1)
    
    cleaned_df = pd.read_csv(cleaned_path)
    cleaned_df['时间'] = pd.to_datetime(cleaned_df['时间'])
    clustered_nodes = pd.read_csv(clustered_path).to_dict('records')
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    graph = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        graph.add_node(row['node_id'], lat=row['lat'], lon=row['lon'],
                       node_type=row.get('type', 'waypoint'),
                       frequency=row.get('frequency', 0))
    for _, row in edges_df.iterrows():
        graph.add_edge(row['from_node'], row['to_node'], weight=row['weight'])
    # 为双向边补充反向边
    for _, row in edges_df.iterrows():
        if str(row.get('is_bidirectional', '')).lower() == 'true':
            if not graph.has_edge(row['to_node'], row['from_node']):
                graph.add_edge(row['to_node'], row['from_node'], weight=row['weight'])
    
    viz.plot_trajectory_sample(cleaned_df, sample_size=20,
                               output_path=os.path.join(img_dir, 'trajectory_sample.png'))
    viz.plot_node_distribution(clustered_nodes,
                               output_path=os.path.join(img_dir, 'node_distribution.png'))
    viz.plot_topology_network(graph,
                              output_path=os.path.join(img_dir, 'topology_network.png'))
    viz.plot_network_statistics(graph,
                                output_path=os.path.join(img_dir, 'network_statistics.png'))

    # 新增 6 张图
    viz.plot_model_comparison(
        output_path=os.path.join(img_dir, 'model_comparison.png'))
    viz.plot_feature_importance(
        output_path=os.path.join(img_dir, 'feature_importance.png'))
    viz.plot_trajectory_before_after(
        cleaned_df, output_path=os.path.join(img_dir, 'trajectory_before_after.png'))
    viz.plot_cluster_comparison(
        clustered_nodes, output_path=os.path.join(img_dir, 'cluster_comparison.png'))
    viz.plot_traffic_heatmap(
        edges_df, nodes_df, output_path=os.path.join(img_dir, 'traffic_heatmap.png'))

    # 路径对比图需要导航结果
    nav_json = os.path.join(output_dir, 'navigation_random_sample.json')
    if os.path.exists(nav_json):
        import json
        with open(nav_json, 'r', encoding='utf-8') as f:
            nav_data = json.load(f)
        paths_data = {}
        for item in nav_data[:3]:
            ptype = item.get('recommended_path', {}).get('type', 'unknown')
            nodes_list = item.get('recommended_path', {}).get('nodes', [])
            if ptype and nodes_list:
                paths_data[ptype] = nodes_list
        if paths_data:
            viz.plot_path_comparison(
                graph, paths_data, output_path=os.path.join(img_dir, 'path_comparison.png'))

    logger.info("图片已保存至: %s", img_dir)


if __name__ == "__main__":
    main()
