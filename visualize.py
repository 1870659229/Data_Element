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
    
    viz.plot_trajectory_sample(cleaned_df, sample_size=20,
                               output_path=os.path.join(img_dir, 'trajectory_sample.png'))
    viz.plot_node_distribution(clustered_nodes,
                               output_path=os.path.join(img_dir, 'node_distribution.png'))
    viz.plot_topology_network(graph,
                              output_path=os.path.join(img_dir, 'topology_network.png'))
    viz.plot_network_statistics(graph,
                                output_path=os.path.join(img_dir, 'network_statistics.png'))
    
    logger.info("图片已保存至: %s", img_dir)


if __name__ == "__main__":
    main()
