# -*- coding: utf-8 -*-
"""P1-4: 探索性数据分析 (EDA) 图表生成

从 cleaned_data.csv (110.6万条) 生成:
  1. eda_spatial_heatmap.png     轨迹空间密度热力图
  2. eda_speed_distribution.png  航速分布直方图 + 船型分组
  3. eda_temporal_pattern.png    时间维度分析 (小时/月份分布)
  4. eda_ship_statistics.png     船舶统计 (船舶数、轨迹段数)

运行: py -3.13 scripts/gen_eda_figures.py
"""
import os, sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "output"
IMG_DIR = OUTPUT_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_CSV = OUTPUT_DIR / "cleaned_data.csv"

log = logging.getLogger("eda")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_data():
    log.info("正在加载 %s ...", CLEANED_CSV)
    # 只读需要的列以节省内存
    usecols = ['船舶名称', '航向', '航速', '纬度', '经度', '时间', 'trajectory_segment']
    df = pd.read_csv(CLEANED_CSV, usecols=usecols, dtype={
        '船舶名称': 'str', '航向': 'float32', '航速': 'float32',
        '纬度': 'float32', '经度': 'float32', 'trajectory_segment': 'int32'
    })
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    log.info("已加载 %d 条记录, %d 艘船", len(df), df['船舶名称'].nunique())
    return df


def gen_spatial_heatmap(df):
    """轨迹空间密度热力图"""
    fig, ax = plt.subplots(figsize=(12, 9))

    # 使用hexbin做高效空间密度渲染
    hb = ax.hexbin(df['经度'], df['纬度'], gridsize=80, cmap='YlOrRd',
                    mincnt=1, alpha=0.85, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('轨迹点密度', fontsize=11)

    # 叠加拓扑节点位置
    nodes_csv = OUTPUT_DIR / "topology_nodes.csv"
    if nodes_csv.exists():
        nodes = pd.read_csv(nodes_csv)
        ax.scatter(nodes['lon'], nodes['lat'], c='#1E88E5', s=8, alpha=0.6,
                   edgecolors='white', linewidths=0.3, zorder=5, label=f'拓扑节点 ({len(nodes)})')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    ax.set_xlabel('经度 (°E)', fontsize=12)
    ax.set_ylabel('纬度 (°N)', fontsize=12)
    ax.set_title(f'AIS轨迹空间密度热力图\n({len(df):,}条轨迹记录, {df["船舶名称"].nunique()}艘船舶)',
                 fontsize=13, fontweight='bold')
    ax.set_facecolor('#1a1a2e')

    out = IMG_DIR / "eda_spatial_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info("空间热力图已保存: %s", out)


def gen_speed_distribution(df):
    """航速分布直方图 + 基本统计（含零值停泊数据，与论文统计口径一致）"""
    speed = df['航速'].dropna()
    speed = speed[speed >= 0]  # 包含零值（停泊/锚泊），与论文统计口径一致

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图: 航速分布直方图
    ax1 = axes[0]
    ax1.hist(speed, bins=80, color='#1E88E5', edgecolor='white', alpha=0.85, density=True)
    mean_spd = speed.mean()
    median_spd = speed.median()
    ax1.axvline(mean_spd, color='#E53935', linestyle='--', linewidth=2,
                label=f'均值: {mean_spd:.2f} 节')
    ax1.axvline(median_spd, color='#FF6F00', linestyle='--', linewidth=2,
                label=f'中位数: {median_spd:.2f} 节')
    # ±1σ 带
    std_spd = speed.std()
    ax1.axvspan(max(0, mean_spd - std_spd), mean_spd + std_spd, alpha=0.1, color='#E53935',
                label=f'±1σ: {max(0, mean_spd-std_spd):.1f}~{mean_spd+std_spd:.1f} 节')
    # 标注零值占比
    zero_pct = (speed == 0).sum() / len(speed) * 100
    ax1.annotate(f'零速(停泊): {zero_pct:.1f}%',
                 xy=(0, ax1.get_ylim()[1] * 0.7), fontsize=8, color='gray',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.set_xlabel('航速 (节)', fontsize=11)
    ax1.set_ylabel('概率密度', fontsize=11)
    ax1.set_title('船舶航速分布', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右图: 按船舶分组的航速箱线图 (Top-15 高频船舶)
    ax2 = axes[1]
    ship_freq = df['船舶名称'].value_counts().head(15)
    top_ships = ship_freq.index.tolist()
    df_top = df[df['船舶名称'].isin(top_ships)]
    ship_groups = [df_top[df_top['船舶名称'] == s]['航速'].dropna() for s in top_ships]
    bp = ax2.boxplot(ship_groups, labels=[s[:8] for s in top_ships],
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#E3F2FD')
        patch.set_edgecolor('#1E88E5')
    ax2.set_xlabel('船舶名称', fontsize=11)
    ax2.set_ylabel('航速 (节)', fontsize=11)
    ax2.set_title('Top-15高频船舶航速分布', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30, labelsize=8)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('航速探索性分析', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "eda_speed_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info("航速分布图已保存: %s", out)


def gen_temporal_pattern(df):
    """时间维度分析: 小时分布 + 轨迹段时间跨度"""
    df_t = df.dropna(subset=['时间'])
    df_t['hour'] = df_t['时间'].dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图: 24小时分布
    ax1 = axes[0]
    hour_counts = df_t['hour'].value_counts().sort_index()
    bars = ax1.bar(hour_counts.index, hour_counts.values, color='#1E88E5',
                   edgecolor='white', alpha=0.85)
    # 标注峰值
    peak_hour = hour_counts.idxmax()
    peak_val = hour_counts.max()
    ax1.bar([peak_hour], [peak_val], color='#E53935', edgecolor='white', alpha=0.9)
    ax1.axhline(hour_counts.mean(), color='#FF6F00', linestyle='--', linewidth=1.5,
                label=f'均值: {hour_counts.mean():.0f}')
    ax1.set_xlabel('时刻 (小时)', fontsize=11)
    ax1.set_ylabel('轨迹点数', fontsize=11)
    ax1.set_title('AIS数据24小时分布', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 右图: 日期分布
    ax2 = axes[1]
    df_t['date'] = df_t['时间'].dt.date
    date_counts = df_t['date'].value_counts().sort_index()
    # 计算日历跨度（含无数据日）
    date_min, date_max = date_counts.index[0], date_counts.index[-1]
    calendar_days = (date_max - date_min).days + 1
    ax2.fill_between(range(len(date_counts)), date_counts.values,
                     color='#1E88E5', alpha=0.3)
    ax2.plot(range(len(date_counts)), date_counts.values, color='#1E88E5', linewidth=2)
    ax2.set_xlabel('日期序号', fontsize=11)
    ax2.set_ylabel('轨迹点数', fontsize=11)
    ax2.set_title(f'AIS数据时间跨度\n({date_min} ~ {date_max}, 共{calendar_days}天)',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('时间维度探索性分析', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "eda_temporal_pattern.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info("时间分布图已保存: %s", out)


def gen_ship_statistics(df):
    """船舶统计: 轨迹段数分布、船舶活跃度"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图: 每艘船的轨迹段数分布
    ax1 = axes[0]
    seg_per_ship = df.groupby('船舶名称')['trajectory_segment'].nunique()
    ax1.hist(seg_per_ship, bins=30, color='#43A047', edgecolor='white', alpha=0.85)
    mean_seg = seg_per_ship.mean()
    ax1.axvline(mean_seg, color='#E53935', linestyle='--', linewidth=2,
                label=f'均值: {mean_seg:.1f} 段/船')
    ax1.set_xlabel('轨迹段数', fontsize=11)
    ax1.set_ylabel('船舶数', fontsize=11)
    ax1.set_title(f'每艘船舶轨迹段数分布\n(共{df["船舶名称"].nunique()}艘船, {df["trajectory_segment"].nunique()}段)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图: 数据质量概览 (4个指标)
    ax2 = axes[1]
    ax2.axis('off')

    total_records = len(df)
    total_ships = df['船舶名称'].nunique()
    total_segments = df['trajectory_segment'].nunique()
    time_span = df['时间'].dropna()
    days = (time_span.max() - time_span.min()).days if len(time_span) > 0 else 0
    lat_range = f"{df['纬度'].min():.2f}~{df['纬度'].max():.2f}"
    lon_range = f"{df['经度'].min():.2f}~{df['经度'].max():.2f}"

    stats_text = (
        f"📊 AIS数据质量概览\n"
        f"{'─' * 35}\n"
        f"  总记录数:      {total_records:>12,} 条\n"
        f"  船舶数:        {total_ships:>12} 艘\n"
        f"  轨迹段数:      {total_segments:>12} 段\n"
        f"  时间跨度:      {days:>12} 天\n"
        f"  纬度范围:      {lat_range:>12} °N\n"
        f"  经度范围:      {lon_range:>12} °E\n"
        f"  航速均值:      {df['航速'].mean():>12.1f} 节\n"
        f"  航速中位数:    {df['航速'].median():>12.1f} 节\n"
        f"{'─' * 35}\n"
        f"  压缩后拓扑:    416节点 / 506边\n"
        f"  数据压缩率:    99.96%"
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='SimHei',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', linewidth=1.5))
    ax2.set_title('数据质量统计摘要', fontsize=12, fontweight='bold')

    fig.suptitle('船舶与数据统计分析', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "eda_ship_statistics.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info("船舶统计图已保存: %s", out)


def save_eda_summary(df):
    """保存EDA统计摘要JSON"""
    summary = {
        'total_records': int(len(df)),
        'total_ships': int(df['船舶名称'].nunique()),
        'total_segments': int(df['trajectory_segment'].nunique()),
        'speed_mean': float(round(df['航速'].mean(), 2)),
        'speed_median': float(round(df['航速'].median(), 2)),
        'speed_std': float(round(df['航速'].std(), 2)),
        'lat_range': [float(round(df['纬度'].min(), 4)), float(round(df['纬度'].max(), 4))],
        'lon_range': [float(round(df['经度'].min(), 4)), float(round(df['经度'].max(), 4))],
    }
    time_span = df['时间'].dropna()
    if len(time_span) > 0:
        summary['time_start'] = str(time_span.min())
        summary['time_end'] = str(time_span.max())
        summary['time_days'] = int((time_span.max() - time_span.min()).days)

    out = OUTPUT_DIR / "eda_summary.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("EDA摘要已保存: %s", out)


def main():
    log.info("=" * 50)
    log.info("P1-4: 探索性数据分析 (EDA)")
    log.info("=" * 50)

    df = load_data()
    gen_spatial_heatmap(df)
    gen_speed_distribution(df)
    gen_temporal_pattern(df)
    gen_ship_statistics(df)
    save_eda_summary(df)

    log.info("=" * 50)
    log.info("全部完成! 生成 4 张EDA图")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
