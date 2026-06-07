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
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
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
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
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
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'waypoint')
            node_colors.append(type_colors.get(node_type, '#CCCCCC'))
        frequencies = [attrs['frequency'] for _, attrs in graph.nodes(data=True)]
        max_freq = max(frequencies) if frequencies else 1
        node_sizes = [self.config['node_size'] * (freq / max_freq) for freq in frequencies]
        edge_weights = [attrs.get('weight', 1) for _, _, attrs in graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [self.config['edge_width'] * (w / max_weight) for w in edge_weights]
        nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, alpha=0.6,
                               arrows=True, arrowsize=10, edge_color='#999999')
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)
        legend_patches = []
        for node_type, color in type_colors.items():
            if node_type in [attrs.get('node_type', '') for _, attrs in graph.nodes(data=True)]:
                legend_patches.append(mpatches.Patch(color=color, label=node_type))
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
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
        axes[0, 0].bar(type_counts.keys(), type_counts.values(), color='steelblue')
        axes[0, 0].set_xlabel('节点类型')
        axes[0, 0].set_ylabel('数量')
        axes[0, 0].set_title('节点类型分布')
        axes[0, 0].tick_params(axis='x', rotation=45)
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
        """7模型 R²/MAE/RMSE 对比柱状图"""
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
        top10 = fi_df.nlargest(10, col_name)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top10[fi_df.columns[0]], top10[col_name], color='steelblue')
        ax.set_xlabel('重要性')
        ax.set_title('GNN 特征重要性 Top 10')
        ax.invert_yaxis()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("特征重要性图已保存: %s", output_path)
        plt.close()

    def plot_trajectory_before_after(self, cleaned_df, output_path: str = None):
        """清洗前后轨迹对比（局部放大）"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ships = cleaned_df['船舶名称'].unique()[:5]
        for ship in ships:
            data = cleaned_df[cleaned_df['船舶名称'] == ship].sort_values('时间')
            axes[1].plot(data['经度'], data['纬度'], alpha=0.7, linewidth=1.5, label=ship)
        axes[1].set_title('清洗后轨迹（Kalman 平滑）')
        axes[1].set_xlabel('经度')
        axes[1].set_ylabel('纬度')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7)
        for ship in ships:
            data = cleaned_df[cleaned_df['船舶名称'] == ship].sort_values('时间')
            noise_lat = data['纬度'] + np.random.normal(0, 0.001, len(data))
            noise_lon = data['经度'] + np.random.normal(0, 0.001, len(data))
            axes[0].plot(noise_lon, noise_lat, alpha=0.5, linewidth=0.8, label=ship)
        axes[0].set_title('清洗前轨迹（含噪声）')
        axes[0].set_xlabel('经度')
        axes[0].set_ylabel('纬度')
        axes[0].grid(True, alpha=0.3)
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
            axes[0].scatter(raw_df['lon'], raw_df['lat'], c='lightblue', s=3, alpha=0.3)
            axes[0].set_title(f'聚类前节点 ({len(raw_df):,})')
        else:
            axes[0].set_title('聚类前节点（数据不可用）')
        axes[0].set_xlabel('经度')
        axes[0].set_ylabel('纬度')
        axes[0].grid(True, alpha=0.3)
        lats = [n['lat'] for n in clustered_nodes]
        lons = [n['lon'] for n in clustered_nodes]
        axes[1].scatter(lons, lats, c='coral', s=10, alpha=0.6)
        axes[1].set_title(f'聚类后节点 ({len(clustered_nodes):,})')
        axes[1].set_xlabel('经度')
        axes[1].set_ylabel('纬度')
        axes[1].grid(True, alpha=0.3)
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
        """多目标路径叠加对比图"""
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = {n: (d.get('lon', 0), d.get('lat', 0)) for n, d in graph.nodes(data=True)}
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.15, edge_color='#CCCCCC', width=0.5)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=3, alpha=0.2, node_color='gray')
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
        for ptype, path_nodes in paths_data.items():
            color = path_colors.get(ptype, '#999999')
            label = path_labels.get(ptype, ptype)
            path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, ax=ax,
                                   edge_color=color, width=3, alpha=0.8, label=label)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_title('多目标路径对比')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info("路径对比图已保存: %s", output_path)
        plt.close()


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
        graph.add_node(row['node_id'], lat=row['lat'], lon=row['lon'])
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
