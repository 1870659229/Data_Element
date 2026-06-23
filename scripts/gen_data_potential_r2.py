"""数据潜能 R² 上限诊断图
=====================================

数据来源: docs/6段vs2段对比分析.md §2.3
说明: 6 段诊断已废弃,本图只展示 2 段 day/night 版本。
      原诊断脚本 _diagnose_6period.py 已清理(2026-06-10)。

运行: python scripts/gen_data_potential_r2.py
"""
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "output" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# 2 段 (n=1001) — 8 核心特征 + RandomForest, time_ratio 空间
# 数据出处: docs/6段vs2段对比分析.md 第 89-91 行
DATA = {
    '整体 R²': 0.988,
    '同边 R²\n(见过)': 0.990,
    '异边 R²\n(未见)': 0.979,
}

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#1E88E5', '#43A047', '#FB8C00']
bars = ax.bar(DATA.keys(), DATA.values(), color=colors, width=0.55)
ax.bar_label(bars, padding=4, fontsize=12, fmt='%.3f')

ax.set_ylim(0, 1.06)
ax.set_ylabel('R²')
ax.set_title('2 段 (n=1001) — 数据潜能 R² 上限', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
out = IMG_DIR / "data_potential_r2.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {out}")
