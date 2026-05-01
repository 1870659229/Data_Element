"""
航道拓扑节点网络提取系统 - 模块化主程序

任务:
├── Task1: 数据预处理（清洗、平滑）
├── Task2: 节点提取（拐点、分岔点、汇合点）
├── Task3: 节点聚类（高频节点识别）
├── Task4: 拓扑网络构建
├── Task5: 动态路段耗时权重建模（多模型对比 XGBoost/LightGBM/RF/GNN）
├── Task6: 可视化
└── Task7: 船舶个性化导航决策（特征检索+物理约束+多目标路径规划）
"""


import os
import sys
import time
import logging
import traceback
import argparse
import pandas as pd
import networkx as nx

from config import DATA_CONFIG
from data_preprocessor import DataPreprocessor
from node_extractor import NodeExtractor
from node_cluster import NodeCluster
from topology_builder import TopologyBuilder
from advanced_weight_model import AdvancedWeightModel
from visualize import TopologyVisualizer
from ship_navigator import ShipNavigationSystem

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TASK_NAMES = {
    1: "数据预处理", 2: "节点提取", 3: "节点聚类",
    4: "拓扑网络构建", 5: "动态路段耗时权重建模",
    6: "可视化", 7: "船舶个性化导航决策",
}


class TaskManager:
    """任务管理器 - 模块化任务调度"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or DATA_CONFIG['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache = {k: None for k in [
            'cleaned_df', 'nodes', 'clustered_nodes', 'graph', 'edge_features'
        ]}
        self.paths = {
            'cleaned_data': os.path.join(self.output_dir, 'cleaned_data.csv'),
            'extracted_nodes': os.path.join(self.output_dir, 'extracted_nodes.csv'),
            'clustered_nodes': os.path.join(self.output_dir, 'clustered_nodes.csv'),
            'topology_json': os.path.join(self.output_dir, 'waterway_topology.json'),
            'topology_nodes': os.path.join(self.output_dir, 'topology_nodes.csv'),
            'topology_edges': os.path.join(self.output_dir, 'topology_edges.csv'),
            'edge_features': os.path.join(self.output_dir, 'edge_features.csv'),
        }
        logger.info("输出目录: %s", self.output_dir)

    # ==================== 通用数据加载方法 ====================

    def _ensure_cleaned_df(self) -> bool:
        """确保 cleaned_df 已加载，失败返回 False"""
        if self.cache['cleaned_df'] is not None:
            return True
        path = self.paths['cleaned_data']
        if not os.path.exists(path):
            logger.error("请先运行任务1（数据预处理）")
            return False
        self.cache['cleaned_df'] = pd.read_csv(path)
        self.cache['cleaned_df']['时间'] = pd.to_datetime(self.cache['cleaned_df']['时间'])
        return True

    def _ensure_nodes(self) -> bool:
        """确保 nodes 已加载"""
        if self.cache['nodes'] is not None:
            return True
        path = self.paths['extracted_nodes']
        if not os.path.exists(path):
            logger.error("请先运行任务2（节点提取）")
            return False
        self.cache['nodes'] = pd.read_csv(path).to_dict('records')
        return True

    def _ensure_clustered_nodes(self) -> bool:
        """确保 clustered_nodes 已加载"""
        if self.cache['clustered_nodes'] is not None:
            return True
        path = self.paths['clustered_nodes']
        if not os.path.exists(path):
            logger.error("请先运行任务3（节点聚类）")
            return False
        self.cache['clustered_nodes'] = pd.read_csv(path).to_dict('records')
        return True

    def _ensure_graph(self) -> bool:
        """确保 graph 已加载（从 CSV 重建）"""
        if self.cache['graph'] is not None:
            return True
        nodes_path = self.paths['topology_nodes']
        edges_path = self.paths['topology_edges']
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            logger.error("请先运行任务4（拓扑网络构建）")
            return False
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
        self.cache['graph'] = nx.DiGraph()
        for _, row in nodes_df.iterrows():
            self.cache['graph'].add_node(row['node_id'], lat=row['lat'], lon=row['lon'])
        for _, row in edges_df.iterrows():
            self.cache['graph'].add_edge(row['from_node'], row['to_node'], weight=row['weight'])
        logger.info("从缓存加载图: %d 节点, %d 边",
                     self.cache['graph'].number_of_nodes(),
                     self.cache['graph'].number_of_edges())
        return True

    # ==================== 任务调度 ====================

    def run_task(self, task_id: int, force: bool = False) -> bool:
        """运行指定任务"""
        if task_id not in TASK_NAMES:
            logger.error("无效的任务编号 %d", task_id)
            return False

        logger.info("=" * 60)
        logger.info("任务%d: %s", task_id, TASK_NAMES[task_id])
        start = time.time()
        success = self._execute_task(task_id, force)
        logger.info("任务%d %s，耗时 %.2fs", task_id,
                     "完成" if success else "失败", time.time() - start)
        return success

    def run_all(self, skip_tasks: list = None):
        """运行所有任务"""
        for tid in range(1, 8):
            if tid in (skip_tasks or []):
                logger.info("跳过任务%d", tid)
                continue
            if not self.run_task(tid):
                logger.error("任务%d 失败，停止执行", tid)
                return False
        self._print_summary()

    def _execute_task(self, task_id: int, force: bool) -> bool:
        dispatch = {
            1: self._task1_preprocess,
            2: self._task2_extract_nodes,
            3: self._task3_cluster_nodes,
            4: self._task4_build_topology,
            5: self._task5_weight_model,
            6: self._task6_visualize,
            7: self._task7_navigation,
        }
        return dispatch[task_id](force)

    # ==================== 任务实现 ====================

    def _task1_preprocess(self, force: bool) -> bool:
        path = self.paths['cleaned_data']
        if not force and os.path.exists(path):
            logger.info("加载缓存: %s", path)
            self.cache['cleaned_df'] = pd.read_csv(path)
            self.cache['cleaned_df']['时间'] = pd.to_datetime(self.cache['cleaned_df']['时间'])
            logger.info("加载完成，%d 条记录", len(self.cache['cleaned_df']))
            return True

        files = [os.path.join(DATA_CONFIG['data_dir'], DATA_CONFIG['file1']),
                 os.path.join(DATA_CONFIG['data_dir'], DATA_CONFIG['file2'])]
        self.cache['cleaned_df'] = DataPreprocessor().process(files)
        self.cache['cleaned_df'].to_csv(path, index=False)
        logger.info("已保存: %s", path)
        return True

    def _task2_extract_nodes(self, force: bool) -> bool:
        if not self._ensure_cleaned_df():
            return False
        path = self.paths['extracted_nodes']
        if not force and os.path.exists(path):
            self.cache['nodes'] = pd.read_csv(path).to_dict('records')
            logger.info("加载缓存: %d 个节点", len(self.cache['nodes']))
            return True

        self.cache['nodes'] = NodeExtractor().extract_nodes(self.cache['cleaned_df'])
        pd.DataFrame(self.cache['nodes']).to_csv(path, index=False)
        logger.info("已保存: %s", path)
        return True

    def _task3_cluster_nodes(self, force: bool) -> bool:
        if not self._ensure_nodes():
            return False
        path = self.paths['clustered_nodes']
        if not force and os.path.exists(path):
            self.cache['clustered_nodes'] = pd.read_csv(path).to_dict('records')
            logger.info("加载缓存: %d 个节点", len(self.cache['clustered_nodes']))
            return True

        cluster = NodeCluster()
        self.cache['clustered_nodes'] = cluster.refine_clusters(
            cluster.cluster_nodes(self.cache['nodes']))
        pd.DataFrame(self.cache['clustered_nodes']).to_csv(path, index=False)
        logger.info("已保存: %s", path)
        return True

    def _task4_build_topology(self, force: bool) -> bool:
        if not self._ensure_clustered_nodes() or not self._ensure_cleaned_df():
            return False

        builder = TopologyBuilder()
        self.cache['graph'] = builder.build_topology(
            self.cache['clustered_nodes'], self.cache['cleaned_df'])
        builder.export_to_json(self.paths['topology_json'])
        builder.export_to_csv(self.paths['topology_nodes'], self.paths['topology_edges'])
        logger.info("图构建完成: %d 节点, %d 边",
                     self.cache['graph'].number_of_nodes(),
                     self.cache['graph'].number_of_edges())
        return True

    def _task5_weight_model(self, force: bool = False, load_model_path: str = None) -> bool:
        """动态路段耗时权重建模（多模型对比）"""
        if not self._ensure_graph() or not self._ensure_cleaned_df():
            return False

        model = AdvancedWeightModel()
        
        # 优先加载已保存的模型（除非 force=True）
        if not force:
            saved_models = [f for f in os.listdir(self.output_dir) if f.startswith('weight_model_') and f.endswith('.pkl')]
            if saved_models and not load_model_path:
                load_model_path = os.path.join(self.output_dir, saved_models[0])
                logger.info("发现已保存的模型: %s", saved_models[0])
        
        if load_model_path and os.path.exists(load_model_path):
            model.load_model(load_model_path)
            self.cache['edge_features'] = model.predict_with_loaded_model(
                self.cache['graph'], self.cache['cleaned_df'])
        else:
            self.cache['edge_features'] = model.build_weights_with_comparison(
                self.cache['graph'], self.cache['cleaned_df'])
            logger.info("开始保存模型...")
            try:
                model.save_model(self.output_dir)
                logger.info("模型保存成功")
            except Exception as e:
                logger.error("模型保存失败: %s", e)
                import traceback; traceback.print_exc()
        
        model.export_results(self.output_dir)
        model.export_model_metadata(self.output_dir)
        return True

    def _task6_visualize(self, force: bool) -> bool:
        if not self._ensure_cleaned_df() or not self._ensure_clustered_nodes():
            return False
        if not self._ensure_graph():
            return False

        img_dir = os.path.join(self.output_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        viz = TopologyVisualizer()

        viz.plot_trajectory_sample(self.cache['cleaned_df'], sample_size=20,
                                   output_path=os.path.join(img_dir, 'trajectory_sample.png'))
        viz.plot_node_distribution(self.cache['clustered_nodes'],
                                  output_path=os.path.join(img_dir, 'node_distribution.png'))
        viz.plot_topology_network(self.cache['graph'],
                                  output_path=os.path.join(img_dir, 'topology_network.png'))
        viz.plot_network_statistics(self.cache['graph'],
                                    output_path=os.path.join(img_dir, 'network_statistics.png'))
        logger.info("图片已保存至: %s", img_dir)
        return True

    def _task7_navigation(self, force: bool) -> bool:
        """船舶个性化导航决策"""
        from datetime import datetime
        
        # 检查前置数据
        required_files = [
            os.path.join(self.output_dir, 'topology_nodes.csv'),
            os.path.join(self.output_dir, 'topology_edges.csv'),
            os.path.join(self.output_dir, 'edge_features_dynamic_weights.csv'),
        ]
        for f in required_files:
            if not os.path.exists(f):
                logger.error("缺少前置数据: %s，请先运行Task1-5", f)
                return False

        nav_system = ShipNavigationSystem(output_dir=self.output_dir)
        nodes = nav_system.get_available_nodes()

        if len(nodes) < 2:
            logger.error("节点数不足，无法进行路径规划")
            return False

        logger.info("可用节点: %d, 可用船舶类型: %s",
                     len(nodes), nav_system.list_ship_types())

        # 演示不同船型的导航决策（为每种船型自动选择可达的远距离起终点）
        demo_ships = [
            ('小型货船', 6),
            ('中型货船', 8),
            ('大型货船', 10),
            ('集装箱船', 9),
            ('大型集装箱船', 11),
            ('油轮', 7),
            ('大型油轮', 13),
            ('客船', 15),
            ('渔船', 5),
            ('拖船', 16),
        ]

        for ship_type, hour in demo_ships:
            start, end = nav_system.find_route_endpoints(ship_type=ship_type)
            if start is None:
                logger.warning(">>> %s: 无可用路径，跳过", ship_type)
                continue

            logger.info(">>> 规划: %s (出发 %d:00), 起点=%s, 终点=%s",
                        ship_type, hour, start, end)
            decision = nav_system.plan_route(
                start=start,
                end=end,
                ship_type=ship_type,
                departure_time=datetime.now().replace(hour=hour, minute=0)
            )
            if decision.get('success'):
                filename = f"navigation_{ship_type}.json"
                nav_system.decision_maker.export_decision(decision, filename)

        logger.info("导航决策任务完成")
        return True

    def _print_summary(self):
        logger.info("=" * 60)
        logger.info("处理完成，输出文件:")
        for name, key in [('清洗数据', 'cleaned_data'), ('提取节点', 'extracted_nodes'),
                          ('聚类节点', 'clustered_nodes'), ('拓扑JSON', 'topology_json'),
                          ('拓扑节点', 'topology_nodes'), ('拓扑边', 'topology_edges')]:
            if os.path.exists(self.paths[key]):
                logger.info("  %s: %s", name, self.paths[key])


def main():
    parser = argparse.ArgumentParser(description='航道拓扑节点网络提取系统')
    parser.add_argument('--task', type=str, default=None, help='运行指定任务，如 "1,2,3"')
    parser.add_argument('--skip', type=str, default=None, help='跳过指定任务，如 "5,6"')
    parser.add_argument('--force', action='store_true', help='强制重新计算')
    args = parser.parse_args()

    try:
        manager = TaskManager()
        if args.task:
            for tid in [int(t.strip()) for t in args.task.split(',')]:
                manager.run_task(tid, force=args.force)
        else:
            skip = [int(t.strip()) for t in args.skip.split(',')] if args.skip else []
            manager.run_all(skip_tasks=skip)
    except Exception as e:
        logger.error("运行错误: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
