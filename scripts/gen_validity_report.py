# -*- coding: utf-8 -*-
"""
路径效度验证脚本

验证系统规划的路径是否与真实AIS轨迹一致，以及物理约束剪枝的合理性。

输出: output/validity_report.json
"""

import sys
import os
import json
import csv
import math
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ship_navigator import (
    MultiObjectiveNavigator,
    PhysicalConstraintChecker,
    ShipCharacteristics,
)
from utils import haversine_distance

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SNAP_THRESHOLD_DEG = 0.01  # 约 1.1 km


# ──────────────────────────────────────────────
# 1. 加载拓扑网络
# ──────────────────────────────────────────────
def load_topology() -> Tuple[Dict, Dict, nx.DiGraph, Set]:
    """从 CSV 构建 NetworkX DiGraph，返回 (nodes_data, graph_edges, G, bidirectional_pairs)"""
    nodes_data = {}
    nodes_path = os.path.join(OUTPUT_DIR, "topology_nodes.csv")
    with open(nodes_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            nid = int(row["node_id"])
            nodes_data[nid] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "type": row.get("type", "unknown"),
                "frequency": int(row.get("frequency", 0)),
                "ship_count": int(row.get("ship_count", 0)),
            }

    graph_edges = {}
    bidirectional_pairs = set()
    edges_path = os.path.join(OUTPUT_DIR, "topology_edges.csv")
    with open(edges_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            u, v = int(row["from_node"]), int(row["to_node"])
            graph_edges[(u, v)] = {"weight": float(row.get("weight", 1))}
            if row.get("is_bidirectional", "").lower() == "true":
                bidirectional_pairs.add((u, v))

    G = nx.DiGraph()
    for nid, attrs in nodes_data.items():
        G.add_node(nid, **attrs)
    for (u, v), attrs in graph_edges.items():
        G.add_edge(u, v, weight=attrs.get("weight", 1))
    # 双向边补反向
    for (u, v) in bidirectional_pairs:
        if not G.has_edge(v, u):
            G.add_edge(v, u, weight=graph_edges[(u, v)].get("weight", 1))
    # 补全单向缺失的反向边（与 app.py _init_navigator 一致）
    for (u, v) in list(graph_edges.keys()):
        if G.has_edge(u, v) and not G.has_edge(v, u):
            G.add_edge(v, u, weight=graph_edges[(u, v)].get("weight", 1))
        elif G.has_edge(v, u) and not G.has_edge(u, v):
            G.add_edge(u, v, weight=graph_edges[(u, v)].get("weight", 1))

    logger.info("拓扑网络: %d 节点, %d 边", G.number_of_nodes(), G.number_of_edges())
    return nodes_data, graph_edges, G, bidirectional_pairs


# ──────────────────────────────────────────────
# 2. 加载边特征
# ──────────────────────────────────────────────
def load_edge_features(bidirectional_pairs: Set) -> Dict:
    """加载 edge_features_dynamic_weights.csv，镜像双向边"""
    ef_path = os.path.join(OUTPUT_DIR, "edge_features_dynamic_weights.csv")
    nav_edge_features = {}
    if not os.path.exists(ef_path):
        logger.warning("edge_features_dynamic_weights.csv 不存在，使用空特征")
        return nav_edge_features

    with open(ef_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            u, v = int(row["from_node"]), int(row["to_node"])
            key = (u, v)
            if key not in nav_edge_features:
                nav_edge_features[key] = {
                    "avg_distance": float(row.get("avg_distance", 100)),
                    "avg_travel_time": float(row.get("avg_travel_time", 30)),
                    "segment_count": int(row.get("segment_count", 0)),
                    "waterway_type": row.get("waterway_type", "open"),
                    "avg_actual_speed": float(row.get("avg_actual_speed", 5)),
                }
    # 镜像双向边
    for (u, v) in bidirectional_pairs:
        if (u, v) in nav_edge_features and (v, u) not in nav_edge_features:
            nav_edge_features[(v, u)] = dict(nav_edge_features[(u, v)])

    logger.info("边特征: %d 条", len(nav_edge_features))
    return nav_edge_features


# ──────────────────────────────────────────────
# 3. 初始化导航系统（参考 app.py _init_navigator）
# ──────────────────────────────────────────────
def init_navigator(G, nav_edge_features, nodes_data):
    constraint_checker = PhysicalConstraintChecker(nav_edge_features, nodes_data, G)
    navigator = MultiObjectiveNavigator(G, nav_edge_features, constraint_checker)
    logger.info("导航器已初始化")
    return navigator, constraint_checker


# ──────────────────────────────────────────────
# 4. 加载 AIS 数据
# ──────────────────────────────────────────────
def load_ais_data() -> pd.DataFrame:
    """加载清洗后 AIS 数据（只读需要的列）"""
    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
    if not os.path.exists(cleaned_path):
        logger.error("cleaned_data.csv 不存在，无法进行路径效度验证")
        return pd.DataFrame()

    usecols = ["船舶名称", "航向", "航速", "纬度", "经度", "时间", "trajectory_segment"]
    df = pd.read_csv(cleaned_path, usecols=usecols)
    df["时间"] = pd.to_datetime(df["时间"])
    # 处理 NaN
    df = df.where(df.notna(), None)
    logger.info("AIS 数据: %d 条记录, %d 艘船", len(df), df["船舶名称"].nunique())
    return df


# ──────────────────────────────────────────────
# 5. 将轨迹点 snap 到最近拓扑节点
# ──────────────────────────────────────────────
def snap_to_node(lat: float, lon: float, nodes_data: Dict) -> Optional[int]:
    """将 (lat, lon) snap 到最近拓扑节点，距离阈值 SNAP_THRESHOLD_DEG 度"""
    best_node = None
    best_dist = float("inf")
    for nid, attrs in nodes_data.items():
        d = math.sqrt((lat - attrs["lat"]) ** 2 + (lon - attrs["lon"]) ** 2)
        if d < best_dist:
            best_dist = d
            best_node = nid
    if best_dist <= SNAP_THRESHOLD_DEG:
        return best_node
    return None


# ──────────────────────────────────────────────
# 6. 从 AIS 数据提取 OD 对
# ──────────────────────────────────────────────
def extract_od_pairs(
    ais_df: pd.DataFrame, nodes_data: Dict, target_count: int = 15
) -> List[Dict]:
    """
    从 AIS 数据中选取有完整轨迹的 OD 对。
    返回 [{"ship": str, "src_node": int, "tgt_node": int, "trajectory_nodes": [int, ...]}, ...]
    """
    if ais_df.empty:
        return []

    # 构建节点坐标的 numpy 数组加速 snap
    node_ids = list(nodes_data.keys())
    node_lats = np.array([nodes_data[nid]["lat"] for nid in node_ids])
    node_lons = np.array([nodes_data[nid]["lon"] for nid in node_ids])

    def fast_snap(lat, lon):
        dists = np.sqrt((node_lats - lat) ** 2 + (node_lons - lon) ** 2)
        idx = np.argmin(dists)
        if dists[idx] <= SNAP_THRESHOLD_DEG:
            return node_ids[idx]
        return None

    # 按船舶+trajectory_segment 分组，提取完整轨迹
    od_pairs = []
    skipped_no_snap = 0
    skipped_same_od = 0
    skipped_short = 0

    grouped = ais_df.groupby(["船舶名称", "trajectory_segment"])
    for (ship, seg), group in grouped:
        if len(group) < 10:
            skipped_short += 1
            continue

        # 按时间排序
        group = group.sort_values("时间")

        # snap 首尾点
        first = group.iloc[0]
        last = group.iloc[-1]
        src = fast_snap(first["纬度"], first["经度"])
        tgt = fast_snap(last["纬度"], last["经度"])

        if src is None or tgt is None:
            skipped_no_snap += 1
            continue
        if src == tgt:
            skipped_same_od += 1
            continue

        # 提取轨迹经过的拓扑节点序列（去重保序）
        traj_nodes = []
        seen = set()
        for _, row in group.iterrows():
            n = fast_snap(row["纬度"], row["经度"])
            if n is not None and n not in seen:
                traj_nodes.append(n)
                seen.add(n)

        if len(traj_nodes) < 3:
            continue

        od_pairs.append({
            "ship": ship,
            "src_node": src,
            "tgt_node": tgt,
            "trajectory_nodes": traj_nodes,
        })

    logger.info(
        "OD 对提取: %d 个有效, 跳过(短=%d, 无snap=%d, 同OD=%d)",
        len(od_pairs), skipped_short, skipped_no_snap, skipped_same_od,
    )

    # 去重：同一 OD 只保留轨迹最长的
    od_best = {}
    for od in od_pairs:
        key = (od["src_node"], od["tgt_node"])
        if key not in od_best or len(od["trajectory_nodes"]) > len(od_best[key]["trajectory_nodes"]):
            od_best[key] = od

    # 选取 target_count 个
    candidates = sorted(od_best.values(), key=lambda x: len(x["trajectory_nodes"]), reverse=True)
    return candidates[:target_count]


# ──────────────────────────────────────────────
# 7. 路径效度计算
# ──────────────────────────────────────────────
def compute_path_validity(
    navigator: MultiObjectiveNavigator,
    od_pairs: List[Dict],
) -> Dict:
    """对每个 OD 对，计算规划路径与实际轨迹的 Jaccard 相似度"""
    details = []
    failed = 0

    for od in od_pairs:
        src, tgt = od["src_node"], od["tgt_node"]
        actual_nodes = set(od["trajectory_nodes"])

        # 用默认策略（大型油轮）规划路径
        ship = ShipCharacteristics(
            ship_name="效度验证_大型油轮",
            ship_type="大型油轮",
            length=96, width=16, draft=5.894,
            height=15, tonnage=3572, max_speed=11,
        )

        try:
            result_paths = navigator.find_paths(src, tgt, ship, hour=None, max_paths=1)
        except Exception as e:
            logger.warning("OD %d->%d 规划失败: %s", src, tgt, e)
            failed += 1
            continue

        if not result_paths:
            failed += 1
            continue

        planned_nodes = set(result_paths[0].nodes)
        intersection = planned_nodes & actual_nodes
        union = planned_nodes | actual_nodes
        jaccard = len(intersection) / len(union) if union else 0.0

        details.append({
            "od": f"{src}->{tgt}",
            "jaccard": round(jaccard, 4),
            "planned_nodes": len(planned_nodes),
            "actual_nodes": len(actual_nodes),
            "intersection": len(intersection),
        })

    if not details:
        return {
            "num_od_pairs": 0,
            "mean_jaccard": 0.0,
            "median_jaccard": 0.0,
            "min_jaccard": 0.0,
            "max_jaccard": 0.0,
            "details": [],
            "failed_count": failed,
        }

    jaccards = [d["jaccard"] for d in details]
    return {
        "num_od_pairs": len(details),
        "mean_jaccard": round(float(np.mean(jaccards)), 4),
        "median_jaccard": round(float(np.median(jaccards)), 4),
        "min_jaccard": round(float(np.min(jaccards)), 4),
        "max_jaccard": round(float(np.max(jaccards)), 4),
        "details": details,
        "failed_count": failed,
    }


# ──────────────────────────────────────────────
# 8. 约束效度计算
# ──────────────────────────────────────────────
def compute_constraint_validity(
    constraint_checker: PhysicalConstraintChecker,
    ais_df: pd.DataFrame,
    nodes_data: Dict,
    graph_edges: Dict,
) -> Dict:
    """
    对大型油轮（吃水 5.894m），获取所有被物理约束剪枝的边。
    从 AIS 数据中统计这些剪枝边上是否有大型船舶通过。
    """
    # 大型油轮
    large_tanker = ShipCharacteristics(
        ship_name="效度验证_大型油轮",
        ship_type="大型油轮",
        length=96, width=16, draft=5.894,
        height=15, tonnage=3572, max_speed=11,
    )
    blocked_edges = constraint_checker.get_blocked_edges(large_tanker)
    logger.info("大型油轮被剪枝边数: %d", len(blocked_edges))

    if ais_df.empty or not blocked_edges:
        return {
            "total_blocked_edges": len(blocked_edges),
            "no_large_ship_edges": len(blocked_edges),
            "constraint_validity_rate": 1.0,  # 无剪枝边 = 约束未误剪，效度为 1
        }

    # 加载船舶特征数据库，判断大型船舶
    ship_db_path = os.path.join(OUTPUT_DIR, "ship_characteristics_db.csv")
    large_ships = set()
    if os.path.exists(ship_db_path):
        ship_db = pd.read_csv(ship_db_path)
        for _, row in ship_db.iterrows():
            length = _safe_float(row.get("length"))
            width = _safe_float(row.get("width"))
            # 大型船舶判定：船长>80m 或 船宽>14m
            if (length is not None and length > 80) or (width is not None and width > 14):
                large_ships.add(row["ship_name"])

    logger.info("大型船舶数: %d", len(large_ships))

    # 构建节点坐标的 numpy 数组加速 snap
    node_ids = list(nodes_data.keys())
    node_lats = np.array([nodes_data[nid]["lat"] for nid in node_ids])
    node_lons = np.array([nodes_data[nid]["lon"] for nid in node_ids])

    # 从 AIS 数据中统计大型船舶经过的边
    # 只看大型船舶的轨迹
    large_ais = ais_df[ais_df["船舶名称"].isin(large_ships)]
    logger.info("大型船舶 AIS 记录数: %d", len(large_ais))

    # 统计大型船舶经过的拓扑边
    large_ship_edges = set()
    for ship, group in large_ais.groupby("船舶名称"):
        for seg, seg_group in group.groupby("trajectory_segment"):
            seg_sorted = seg_group.sort_values("时间")
            prev_node = None
            for _, row in seg_sorted.iterrows():
                lat, lon = row["纬度"], row["经度"]
                if lat is None or lon is None:
                    continue
                curr_node = _fast_snap_single(lat, lon, node_ids, node_lats, node_lons)
                if curr_node is None:
                    continue
                if prev_node is not None and prev_node != curr_node:
                    large_ship_edges.add((prev_node, curr_node))
                prev_node = curr_node

    logger.info("大型船舶经过的拓扑边数: %d", len(large_ship_edges))

    # 计算约束效度：剪枝边中确实无大型船通过的比例
    no_large_ship = 0
    for edge in blocked_edges:
        if edge not in large_ship_edges:
            no_large_ship += 1

    total = len(blocked_edges)
    rate = no_large_ship / total if total > 0 else 0.0

    return {
        "total_blocked_edges": total,
        "no_large_ship_edges": no_large_ship,
        "constraint_validity_rate": round(rate, 4),
    }


def _safe_float(val, fallback=None):
    if val is None:
        return fallback
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return fallback
        return v
    except (ValueError, TypeError):
        return fallback


def _fast_snap_single(lat, lon, node_ids, node_lats, node_lons):
    dists = np.sqrt((node_lats - lat) ** 2 + (node_lons - lon) ** 2)
    idx = np.argmin(dists)
    if dists[idx] <= SNAP_THRESHOLD_DEG:
        return node_ids[idx]
    return None


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("路径效度验证脚本")
    logger.info("=" * 60)

    # 1. 加载拓扑网络
    nodes_data, graph_edges, G, bidirectional_pairs = load_topology()

    # 2. 加载边特征
    nav_edge_features = load_edge_features(bidirectional_pairs)

    # 3. 初始化导航系统
    navigator, constraint_checker = init_navigator(G, nav_edge_features, nodes_data)

    # 4. 加载 AIS 数据
    ais_df = load_ais_data()

    # 5. 提取 OD 对
    od_pairs = extract_od_pairs(ais_df, nodes_data, target_count=15)
    logger.info("选取 %d 个 OD 对用于路径效度验证", len(od_pairs))

    # 6. 路径效度计算
    logger.info("--- 路径效度计算 ---")
    path_validity = compute_path_validity(navigator, od_pairs)

    # 7. 约束效度计算
    logger.info("--- 约束效度计算 ---")
    constraint_validity = compute_constraint_validity(
        constraint_checker, ais_df, nodes_data, graph_edges
    )

    # 8. 输出报告
    report = {
        "path_validity": path_validity,
        "constraint_validity": constraint_validity,
    }

    output_path = os.path.join(OUTPUT_DIR, "validity_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("效度报告已保存: %s", output_path)

    # 打印摘要
    pv = path_validity
    cv = constraint_validity
    print("\n" + "=" * 60)
    print("路径效度验证结果")
    print("=" * 60)
    print(f"  OD 对数: {pv['num_od_pairs']}")
    print(f"  规划失败: {pv.get('failed_count', 0)}")
    print(f"  Jaccard 均值: {pv['mean_jaccard']:.4f}")
    print(f"  Jaccard 中位数: {pv['median_jaccard']:.4f}")
    print(f"  Jaccard 最小值: {pv['min_jaccard']:.4f}")
    print(f"  Jaccard 最大值: {pv['max_jaccard']:.4f}")
    print()
    print("约束效度验证结果")
    print(f"  被剪枝边数: {cv['total_blocked_edges']}")
    print(f"  无大型船通过的边数: {cv['no_large_ship_edges']}")
    print(f"  约束效度率: {cv['constraint_validity_rate']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
