"""
综合节点中心性分析模块

参考冯润航 2025 论文《基于出租车轨迹数据的城市路网建模及节点影响力分析》的方法论：
- 度中心性（Degree Centrality）：节点连接广度
- 介数中心性（Betweenness Centrality）：节点作为桥梁的重要性
- 接近中心性（Closeness Centrality）：节点到其他节点的平均可达性
- 特征向量中心性（Eigenvector Centrality）：连接其他重要节点的能力

扩展点（相对论文）：
- 论文原方案：3 种中心性 + 熵权法
- 本实现：4 种中心性 + PCA 自动赋权（更稳定的客观赋权）
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List
import logging
import os
import json

logger = logging.getLogger(__name__)


class NodeCentralityAnalyzer:
    """综合节点中心性分析器"""

    DEFAULT_WEIGHTS = {
        'degree': 0.25,
        'betweenness': 0.30,
        'closeness': 0.20,
        'eigenvector': 0.25,
    }

    def __init__(self, graph: nx.Graph = None, nodes_data: Dict = None):
        self.graph = graph
        self.nodes_data = nodes_data or {}
        self.centrality_scores: Dict[int, Dict] = {}
        self.weights = dict(self.DEFAULT_WEIGHTS)

    def compute_all_centrality(self, k_sample: int = 200) -> Dict[int, Dict]:
        """计算所有节点的 4 种中心性指标

        Args:
            k_sample: 介数中心性采样节点数（用于大图近似计算）

        Returns:
            {node_id: {degree, betweenness, closeness, eigenvector, composite}}
        """
        if self.graph is None:
            raise ValueError("图未初始化")

        G_undirected = (
            self.graph.to_undirected()
            if self.graph.is_directed()
            else self.graph
        )

        logger.info("计算度中心性...")
        degree_c = nx.degree_centrality(G_undirected)

        logger.info("计算介数中心性（采样k=%d）...", k_sample)
        n_nodes = G_undirected.number_of_nodes()
        if n_nodes > 500:
            betweenness_c = nx.betweenness_centrality(
                G_undirected, k=min(k_sample, n_nodes)
            )
        else:
            betweenness_c = nx.betweenness_centrality(G_undirected)

        logger.info("计算接近中心性...")
        closeness_c = nx.closeness_centrality(G_undirected)

        logger.info("计算特征向量中心性...")
        try:
            eigenvector_c = nx.eigenvector_centrality(
                G_undirected, max_iter=500
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("特征向量中心性未收敛，使用度中心性近似")
            eigenvector_c = degree_c

        for nid in G_undirected.nodes():
            self.centrality_scores[nid] = {
                'degree': degree_c.get(nid, 0),
                'betweenness': betweenness_c.get(nid, 0),
                'closeness': closeness_c.get(nid, 0),
                'eigenvector': eigenvector_c.get(nid, 0),
            }

        self._compute_composite_centrality()

        logger.info("中心性计算完成: %d 个节点", len(self.centrality_scores))
        return self.centrality_scores

    def _compute_composite_centrality(self):
        """计算综合中心性指标（加权融合）

        先对 4 种中心性按列做 min-max 归一化到 [0,1]，
        再用 self.weights 加权求和，使 composite 值域稳定在 [0,1]。
        """
        if not self.centrality_scores:
            return

        metrics = ['degree', 'betweenness', 'closeness', 'eigenvector']
        raw = {m: [s[m] for s in self.centrality_scores.values()] for m in metrics}
        normalized = {}
        for m in metrics:
            vals = raw[m]
            v_min, v_max = min(vals), max(vals)
            if v_max - v_min < 1e-12:
                normalized[m] = [0.0] * len(vals)
            else:
                normalized[m] = [(v - v_min) / (v_max - v_min) for v in vals]

        for i, (nid, scores) in enumerate(self.centrality_scores.items()):
            scores['composite'] = sum(
                self.weights[m] * normalized[m][i] for m in metrics
            )

    def auto_weight_by_pca(self):
        """使用 PCA 自动确定 4 种中心性的权重

        原理：对 4 种中心性做 PCA，第一主成分的载荷绝对值作为权重
        """
        if not self.centrality_scores:
            return

        from sklearn.decomposition import PCA

        X = np.array([
            [
                s['degree'], s['betweenness'],
                s['closeness'], s['eigenvector']
            ]
            for s in self.centrality_scores.values()
        ])

        pca = PCA(n_components=1)
        pca.fit(X)

        loadings = np.abs(pca.components_[0])
        total = loadings.sum()
        if total == 0:
            logger.warning("PCA 载荷全为 0，使用默认权重")
            return

        self.weights = {
            'degree': float(loadings[0] / total),
            'betweenness': float(loadings[1] / total),
            'closeness': float(loadings[2] / total),
            'eigenvector': float(loadings[3] / total),
        }

        logger.info(
            "PCA 自动权重: %s",
            {k: f"{v:.4f}" for k, v in self.weights.items()}
        )
        self._compute_composite_centrality()

    def identify_key_nodes(self, top_pct: float = 0.1) -> Dict[str, List[int]]:
        """识别关键节点

        Args:
            top_pct: 前百分之几为关键节点

        Returns:
            {'hub': 枢纽节点, 'bridge': 桥梁节点, 'vulnerable': 脆弱节点}
        """
        if not self.centrality_scores:
            return {'hub': [], 'bridge': [], 'vulnerable': []}

        sorted_by_composite = sorted(
            self.centrality_scores.items(),
            key=lambda x: x[1]['composite'],
            reverse=True
        )
        n_total = len(sorted_by_composite)
        n_top = max(1, min(int(n_total * top_pct), n_total // 2))

        hub_nodes = [nid for nid, _ in sorted_by_composite[:n_top]]
        vulnerable_nodes = [
            nid for nid, _ in sorted_by_composite[-n_top:]
        ]

        sorted_by_betweenness = sorted(
            self.centrality_scores.items(),
            key=lambda x: x[1]['betweenness'],
            reverse=True
        )
        bridge_nodes = [nid for nid, _ in sorted_by_betweenness[:n_top]]

        return {
            'hub': hub_nodes,
            'bridge': bridge_nodes,
            'vulnerable': vulnerable_nodes,
        }

    def get_centrality(self, node_id: int) -> float:
        """获取节点的综合中心性"""
        scores = self.centrality_scores.get(node_id, {})
        return scores.get('composite', 0.0)

    def export_results(self, output_dir: str):
        """导出中心性分析结果"""
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for nid, scores in self.centrality_scores.items():
            row = {'node_id': nid}
            row.update(scores)
            node_info = self.nodes_data.get(nid, {})
            row['lat'] = node_info.get('lat', 0)
            row['lon'] = node_info.get('lon', 0)
            row['frequency'] = node_info.get('frequency', 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(
            os.path.join(output_dir, 'node_centrality.csv'),
            index=False
        )

        key_nodes = self.identify_key_nodes()
        with open(
            os.path.join(output_dir, 'key_nodes.json'),
            'w', encoding='utf-8'
        ) as f:
            json.dump(key_nodes, f, ensure_ascii=False, indent=2)

        logger.info("中心性结果已导出: %s", output_dir)
