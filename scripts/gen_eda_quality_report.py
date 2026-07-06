# -*- coding: utf-8 -*-
"""数据质量诊断：逐步执行清洗并统计每步去除量

运行: py -3.13 scripts/gen_eda_quality_report.py
输出: output/eda_quality_report.json
"""
import os, sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_CONFIG, CLEANING_CONFIG
from utils import haversine_distance

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("quality")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_raw_data():
    """加载原始AIS数据，优先从xlsx，回退到cleaned_data.csv"""
    data_dir = DATA_CONFIG['data_dir']
    file1 = os.path.join(data_dir, DATA_CONFIG['file1'])
    file2 = os.path.join(data_dir, DATA_CONFIG['file2'])

    if os.path.exists(file1) and os.path.exists(file2):
        log.info("从原始xlsx加载数据...")
        dfs = []
        for fp in [file1, file2]:
            df = pd.read_excel(fp)
            log.info("  加载 %s: %d 条", os.path.basename(fp), len(df))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        log.info("  合计: %d 条", len(df))
        return df, True  # True = 原始数据

    # 回退到 cleaned_data.csv
    cleaned_path = OUTPUT_DIR / "cleaned_data.csv"
    if cleaned_path.exists():
        log.info("原始xlsx不可用，从 cleaned_data.csv 加载（已清洗数据，各步去除量将为0或极小）...")
        df = pd.read_csv(cleaned_path)
        log.info("  加载: %d 条", len(df))
        return df, False  # False = 已清洗数据

    raise FileNotFoundError("找不到原始数据(xlsx)或 cleaned_data.csv")


def step_convert_time(df):
    """时间格式转换（不删除数据）"""
    df['时间'] = pd.to_datetime(df['时间'])
    return df


def step_dedup(df):
    """去重"""
    before = len(df)
    df = df.sort_values(['船舶名称', '时间'])
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['船舶名称', '时间'], keep='first')
    removed = before - len(df)
    return df, removed


def step_speed_filter(df):
    """异常速度过滤"""
    before = len(df)
    max_speed = CLEANING_CONFIG['max_speed']
    df = df[(df['航速'] >= 0) & (df['航速'] <= max_speed)]
    removed = before - len(df)
    return df, removed


def step_drift_detect(df):
    """漂移检测（向量化 + IsolationForest）"""
    before = len(df)
    drift_indices = set()
    max_gap = CLEANING_CONFIG['max_time_gap']
    max_jump = CLEANING_CONFIG['max_distance_jump']
    max_speed = CLEANING_CONFIG['max_speed']

    all_features = []
    all_indices = []

    for ship_name, group in df.groupby('船舶名称'):
        if len(group) < 2:
            continue
        group = group.sort_values('时间').reset_index()

        lats = group['纬度'].values
        lons = group['经度'].values
        times = group['时间'].values
        speeds = group['航速'].values
        courses = group['航向'].values
        orig_idx = group['index'].values

        n = len(group)
        if n < 2:
            continue

        time_diffs = np.zeros(n)
        time_diffs[1:] = [float((times[i] - times[i-1]) / np.timedelta64(1, 's')) for i in range(1, n)]

        dists = np.zeros(n)
        dists[1:] = [haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i]) for i in range(1, n)]

        instant_speeds = np.zeros(n)
        valid_time = time_diffs > 0
        instant_speeds[valid_time] = (dists[valid_time] / time_diffs[valid_time]) * 1.944

        drift_mask = (dists > max_jump) | (instant_speeds > max_speed)
        drift_mask &= ~((time_diffs > max_gap) | (time_diffs <= 0))

        for idx in np.where(drift_mask)[0]:
            drift_indices.add(orig_idx[idx])

        if n >= 5:
            speed_ratios = np.zeros(n)
            course_diffs = np.zeros(n)
            prev_valid = np.roll(valid_time, 1) & valid_time
            calc_speeds = np.zeros(n)
            calc_speeds[prev_valid] = (dists[prev_valid] / time_diffs[prev_valid]) * 1.944

            safe_prev = dists[2:n] / np.maximum(dists[1:n-1], 1e-6)
            speed_ratios[1:-1] = safe_prev

            raw_course_diff = np.abs(np.roll(courses, -1) - np.roll(courses, 1))
            course_diffs[1:-1] = np.where(raw_course_diff[1:-1] > 180, 360 - raw_course_diff[1:-1], raw_course_diff[1:-1])

            for i in range(1, n - 1):
                if valid_time[i] and prev_valid[i]:
                    all_features.append([
                        speeds[i], calc_speeds[i], speed_ratios[i],
                        course_diffs[i], dists[i]
                    ])
                    all_indices.append(orig_idx[i])

    if all_features:
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            X = np.array(all_features)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = IsolationForest(
                contamination=CLEANING_CONFIG.get('iforest_contamination', 0.01),
                random_state=42,
                n_estimators=CLEANING_CONFIG.get('iforest_n_estimators', 100))
            preds = clf.fit_predict(X_scaled)

            for idx, pred in zip(all_indices, preds):
                if pred == -1:
                    drift_indices.add(idx)

            log.info("  IsolationForest: %d 样本, %d 异常",
                     len(all_features), sum(preds == -1))
        except ImportError:
            log.warning("sklearn 未安装，跳过 IsolationForest")

    df = df.drop(list(drift_indices))
    removed = before - len(df)
    return df, removed


def step_short_traj(df):
    """短轨迹过滤"""
    before = len(df)
    min_pts = CLEANING_CONFIG['min_trajectory_points']
    counts = df.groupby('船舶名称').size()
    valid = counts[counts >= min_pts].index
    df = df[df['船舶名称'].isin(valid)]
    removed = before - len(df)
    return df, removed


def step_missing(df):
    """缺失值处理 + 地理围栏"""
    before = len(df)
    missing = df.isnull().sum()
    if missing.any():
        log.info("  缺失值: %s", missing[missing > 0].to_dict())
        df = df.dropna()
    # 地理围栏
    df = df[(df['纬度'] >= 18) & (df['纬度'] <= 42) &
            (df['经度'] >= 105) & (df['经度'] <= 125)]
    removed = before - len(df)
    return df, removed


def main():
    log.info("=" * 50)
    log.info("数据质量诊断：逐步清洗统计")
    log.info("=" * 50)

    df, is_raw = load_raw_data()
    raw_records = len(df)

    # Step 0: 时间转换（不删数据）
    df = step_convert_time(df)

    # Step 1: 去重
    df, removed_dedup = step_dedup(df)
    after_dedup = len(df)
    log.info("去重: %d -> %d (去除 %d)", raw_records, after_dedup, removed_dedup)

    # Step 2: 异常速度过滤
    df, removed_speed = step_speed_filter(df)
    after_speed = len(df)
    log.info("异常速度: %d -> %d (去除 %d)", after_dedup, after_speed, removed_speed)

    # Step 3: 漂移检测
    df, removed_drift = step_drift_detect(df)
    after_drift = len(df)
    log.info("漂移检测: %d -> %d (去除 %d)", after_speed, after_drift, removed_drift)

    # Step 4: 短轨迹过滤
    df, removed_short_traj = step_short_traj(df)
    after_short_traj = len(df)
    log.info("短轨迹: %d -> %d (去除 %d)", after_drift, after_short_traj, removed_short_traj)

    # Step 5: 缺失值 + 地理围栏
    df, removed_missing = step_missing(df)
    after_missing = len(df)
    log.info("缺失值+围栏: %d -> %d (去除 %d)", after_short_traj, after_missing, removed_missing)

    final_records = len(df)
    total_removed = raw_records - final_records
    retention_rate = round(final_records / raw_records * 100, 2) if raw_records > 0 else 0

    report = {
        "raw_records": raw_records,
        "after_dedup": after_dedup,
        "removed_dedup": removed_dedup,
        "after_speed_filter": after_speed,
        "removed_speed": removed_speed,
        "after_drift": after_drift,
        "removed_drift": removed_drift,
        "after_short_traj": after_short_traj,
        "removed_short_traj": removed_short_traj,
        "after_missing": after_missing,
        "removed_missing": removed_missing,
        "final_records": final_records,
        "total_removed": total_removed,
        "retention_rate": retention_rate,
    }

    if not is_raw:
        report["_note"] = "原始xlsx不可用，使用cleaned_data.csv作为输入；各步去除量反映对已清洗数据的重复清洗结果"

    out_path = OUTPUT_DIR / "eda_quality_report.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 50)
    log.info("数据质量报告已保存: %s", out_path)
    log.info("  原始: %d -> 最终: %d (保留率 %.2f%%)", raw_records, final_records, retention_rate)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
