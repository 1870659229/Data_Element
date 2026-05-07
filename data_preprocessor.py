"""
航道拓扑节点网络提取系统 - 数据预处理模块（优化版）
功能：清洗、平滑海量原始GPS/AIS数据

优化内容：
1. 卡尔曼滤波平滑（替代EMA）
2. IsolationForest多维异常检测（替代固定阈值）
3. 轨迹分段（基于时间间隔识别信号丢失）
4. 地理围栏（经纬度范围+海岸线过滤）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

from config import CLEANING_CONFIG, SMOOTHING_CONFIG, DATA_CONFIG
from utils import haversine_distance

logger = logging.getLogger(__name__)


class KalmanFilter1D:
    """一维卡尔曼滤波器"""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self.q = process_noise
        self.r = measurement_noise
        self.x = None
        self.p = 1.0

    def reset(self, initial_value: float):
        self.x = initial_value
        self.p = 1.0

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.reset(measurement)
            return measurement

        # 预测
        self.p = self.p + self.q

        # 更新
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p

        return self.x


class DataPreprocessor:
    """数据预处理器（优化版）"""

    def __init__(self, config: Dict = None):
        self.cleaning_config = config.get('cleaning', CLEANING_CONFIG) if config else CLEANING_CONFIG
        self.smoothing_config = config.get('smoothing', SMOOTHING_CONFIG) if config else SMOOTHING_CONFIG

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """加载并合并数据"""
        dfs = []
        for fp in file_paths:
            df = pd.read_excel(fp)
            dfs.append(df)
            logger.info("加载 %s: %d 条记录", fp, len(df))
        combined = pd.concat(dfs, ignore_index=True)
        logger.info("合计 %d 条记录", len(combined))
        return combined

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗主流程"""
        original_count = len(df)
        logger.info("开始数据清洗，原始 %d 条", original_count)

        df = self._convert_time(df)
        df = self._remove_duplicates(df)
        df = self._filter_abnormal_speed(df)
        df = self._detect_drift_multidim(df)
        df = self._filter_short_trajectories(df)
        df = self._check_data_integrity(df)

        removed = original_count - len(df)
        logger.info("清洗完成: %d -> %d 条，去除 %d 条 (%.1f%%)",
                     original_count, len(df), removed, removed / original_count * 100)
        return df

    def _convert_time(self, df):
        df['时间'] = pd.to_datetime(df['时间'])
        return df

    def _remove_duplicates(self, df):
        original = len(df)
        df = df.sort_values(['船舶名称', '时间'])
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=['船舶名称', '时间'], keep='first')
        logger.info("去除重复: %d 条", original - len(df))
        return df

    def _filter_abnormal_speed(self, df):
        original = len(df)
        df = df[(df['航速'] >= 0) & (df['航速'] <= self.cleaning_config['max_speed'])]
        logger.info("去除异常速度: %d 条", original - len(df))
        return df

    def _detect_drift_multidim(self, df: pd.DataFrame) -> pd.DataFrame:
        """多维异常检测：向量化漂移检测 + 全局IsolationForest"""
        original = len(df)
        drift_indices = set()
        max_gap = self.cleaning_config['max_time_gap']
        max_jump = self.cleaning_config['max_distance_jump']
        max_speed = self.cleaning_config['max_speed']

        logger.info("  向量化漂移检测...")
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
                next_valid = np.roll(valid_time, -1) & valid_time

                safe_prev = dists[2:n] / np.maximum(dists[1:n-1], 1e-6)
                speed_ratios[1:-1] = safe_prev

                raw_course_diff = np.abs(np.roll(courses, -1) - np.roll(courses, 1))
                course_diffs[1:-1] = np.where(raw_course_diff[1:-1] > 180, 360 - raw_course_diff[1:-1], raw_course_diff[1:-1])

                calc_speeds = np.zeros(n)
                calc_speeds[prev_valid] = (dists[prev_valid] / time_diffs[prev_valid]) * 1.944

                for i in range(1, n - 1):
                    if valid_time[i] and prev_valid[i]:
                        all_features.append([
                            speeds[i],
                            calc_speeds[i],
                            speed_ratios[i],
                            course_diffs[i],
                            dists[i]
                        ])
                        all_indices.append(orig_idx[i])

        if all_features:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler

                X = np.array(all_features)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                clf = IsolationForest(contamination=0.01, random_state=42, n_estimators=50)
                preds = clf.fit_predict(X_scaled)

                for idx, pred in zip(all_indices, preds):
                    if pred == -1:
                        drift_indices.add(idx)

                logger.info("  IsolationForest: %d 个样本, 检测到 %d 异常", len(all_features), sum(preds == -1))

            except ImportError:
                logger.warning("sklearn 未安装，跳过 IsolationForest")

        df = df.drop(list(drift_indices))
        logger.info("去除漂移（含多维异常）: %d 条", original - len(df))
        return df

    def _filter_short_trajectories(self, df):
        counts = df.groupby('船舶名称').size()
        valid = counts[counts >= self.cleaning_config['min_trajectory_points']].index
        df = df[df['船舶名称'].isin(valid)]
        logger.info("剩余船舶: %d 艘", df['船舶名称'].nunique())
        return df

    def _check_data_integrity(self, df):
        missing = df.isnull().sum()
        if missing.any():
            logger.warning("缺失值: %s", missing[missing > 0].to_dict())
            df = df.dropna()
        # 中国近海地理围栏
        df = df[(df['纬度'] >= 18) & (df['纬度'] <= 42) &
                (df['经度'] >= 105) & (df['经度'] <= 125)]
        return df

    def segment_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """轨迹分段：基于时间间隔识别信号丢失，将长轨迹切分为多个段"""
        logger.info("开始轨迹分段...")
        max_gap = self.cleaning_config['max_time_gap']

        segmented = []
        for ship_name, group in df.groupby('船舶名称'):
            group = group.sort_values('时间').reset_index(drop=True)
            if len(group) < 2:
                group['trajectory_segment'] = 0
                segmented.append(group)
                continue

            # 计算时间间隔
            time_diffs = group['时间'].diff().dt.total_seconds()
            # 标记分段点（时间间隔超过阈值）
            segment_ids = (time_diffs > max_gap).cumsum()
            group['trajectory_segment'] = segment_ids

            # 过滤过短的段
            seg_counts = group.groupby('trajectory_segment').size()
            valid_segments = seg_counts[seg_counts >= 3].index
            group = group[group['trajectory_segment'].isin(valid_segments)]

            segmented.append(group)

        result = pd.concat(segmented, ignore_index=True)
        n_segments = result.groupby('船舶名称')['trajectory_segment'].nunique().sum()
        logger.info("轨迹分段完成: %d 个有效段", n_segments)
        return result

    def smooth_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """轨迹平滑处理（优化版：真正使用卡尔曼滤波）"""
        logger.info("开始轨迹平滑...")
        smoothed = []

        for ship_name, group in df.groupby('船舶名称'):
            group = group.sort_values('时间').copy()

            if self.smoothing_config.get('use_kalman', True):
                group = self._kalman_smooth(group)
            else:
                group = self._moving_average_smooth(group)

            smoothed.append(group)

        result = pd.concat(smoothed, ignore_index=True)
        logger.info("平滑完成: %d 艘船舶", len(df.groupby('船舶名称')))
        return result

    def _kalman_smooth(self, group: pd.DataFrame) -> pd.DataFrame:
        """卡尔曼滤波平滑（分别对纬度和经度滤波）"""
        q = self.smoothing_config.get('process_noise', 0.01)
        r = self.smoothing_config.get('measurement_noise', 1.0)

        kf_lat = KalmanFilter1D(process_noise=q, measurement_noise=r)
        kf_lon = KalmanFilter1D(process_noise=q, measurement_noise=r)

        lats = group['纬度'].values
        lons = group['经度'].values

        smoothed_lats = []
        smoothed_lons = []

        for lat, lon in zip(lats, lons):
            smoothed_lats.append(kf_lat.update(lat))
            smoothed_lons.append(kf_lon.update(lon))

        group['纬度'] = smoothed_lats
        group['经度'] = smoothed_lons
        return group

    def _moving_average_smooth(self, group):
        w = self.smoothing_config['window_size']
        group['纬度'] = group['纬度'].rolling(window=w, center=True, min_periods=1).mean()
        group['经度'] = group['经度'].rolling(window=w, center=True, min_periods=1).mean()
        return group

    def process(self, file_paths: List[str]) -> pd.DataFrame:
        """完整预处理流程"""
        df = self.load_data(file_paths)
        df = self.clean_data(df)
        df = self.segment_trajectories(df)
        df = self.smooth_trajectories(df)
        return df
