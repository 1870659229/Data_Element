"""生成数据融合示意图：shipxy + CCS → ship_characteristics_db"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# --- shipxy box (left) ---
sx_x, sx_y, sx_w, sx_h = 0.5, 3.0, 4.0, 4.5
sx_box = FancyBboxPatch((sx_x, sx_y), sx_w, sx_h,
                         boxstyle="round,pad=0.15", facecolor='#E3F2FD',
                         edgecolor='#1565C0', linewidth=2)
ax.add_patch(sx_box)
ax.text(sx_x + sx_w/2, sx_y + sx_h - 0.35, 'shipxy AIS平台',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#0D47A1')
ax.text(sx_x + sx_w/2, sx_y + sx_h - 0.75, '(309艘完整覆盖)',
        ha='center', va='center', fontsize=10, color='#1565C0')

sx_fields = [
    ('MMSI', '关联键', '#FF8F00'),
    ('船名', '全', '#2E7D32'),
    ('船舶类型', '全', '#2E7D32'),
    ('船长', '100%', '#2E7D32'),
    ('船宽', '100%', '#2E7D32'),
    ('最大航速', '100%', '#2E7D32'),
    ('吃水', '66%', '#F57F17'),
    ('吨位', '部分', '#F57F17'),
]
for i, (field, status, color) in enumerate(sx_fields):
    yy = sx_y + sx_h - 1.2 - i * 0.42
    ax.text(sx_x + 0.4, yy, field, ha='left', va='center', fontsize=10, color='#212121')
    ax.text(sx_x + sx_w - 0.4, yy, status, ha='right', va='center', fontsize=9, color=color, fontweight='bold')

# --- CCS box (right) ---
ccs_x, ccs_y, ccs_w, ccs_h = 7.5, 3.0, 4.0, 4.5
ccs_box = FancyBboxPatch((ccs_x, ccs_y), ccs_w, ccs_h,
                          boxstyle="round,pad=0.15", facecolor='#FFF3E0',
                          edgecolor='#E65100', linewidth=2)
ax.add_patch(ccs_box)
ax.text(ccs_x + ccs_w/2, ccs_y + ccs_h - 0.35, 'CCS 中国船级社',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#BF360C')
ax.text(ccs_x + ccs_w/2, ccs_y + ccs_h - 0.75, '(14艘匹配补充)',
        ha='center', va='center', fontsize=10, color='#E65100')

ccs_fields = [
    ('MMSI', '关联键', '#FF8F00'),
    ('净吨位', '全', '#2E7D32'),
    ('载重吨(DWT)', '全', '#2E7D32'),
    ('型深', '全', '#2E7D32'),
    ('船厂', '全', '#2E7D32'),
    ('建造年份', '全', '#2E7D32'),
    ('船级符号', '全', '#2E7D32'),
    ('主机功率', '全', '#2E7D32'),
]
for i, (field, status, color) in enumerate(ccs_fields):
    yy = ccs_y + ccs_h - 1.2 - i * 0.42
    ax.text(ccs_x + 0.4, yy, field, ha='left', va='center', fontsize=10, color='#212121')
    ax.text(ccs_x + ccs_w - 0.4, yy, status, ha='right', va='center', fontsize=9, color=color, fontweight='bold')

# --- MMSI association arrow ---
ax.annotate('', xy=(ccs_x + 0.1, 6.5), xytext=(sx_x + sx_w - 0.1, 6.5),
            arrowprops=dict(arrowstyle='<->', color='#FF8F00', lw=2.5))
ax.text(6.0, 6.75, 'MMSI 关联', ha='center', va='center', fontsize=10,
        color='#FF8F00', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=1))

# --- Merge arrow (down) ---
merge_x = 6.0
ax.annotate('', xy=(merge_x, 2.2), xytext=(merge_x, 3.0),
            arrowprops=dict(arrowstyle='->', color='#424242', lw=2.5))
ax.text(merge_x, 2.6, '关联合并', ha='center', va='center', fontsize=10,
        color='#424242', fontweight='bold')

# --- Result box (bottom center) ---
res_x, res_y, res_w, res_h = 3.0, 0.3, 6.0, 1.8
res_box = FancyBboxPatch((res_x, res_y), res_w, res_h,
                          boxstyle="round,pad=0.15", facecolor='#E8F5E9',
                          edgecolor='#2E7D32', linewidth=2)
ax.add_patch(res_box)
ax.text(res_x + res_w/2, res_y + res_h - 0.3, 'ship_characteristics_db.csv（309艘融合数据）',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1B5E20')

coverage_items = [
    '船长/船宽/航速: 100%',
    '吃水: 66% (204/309)',
    '型深/DWT: 4.5% (14/309)',
    '船高: 0% → 推断补全',
]
for i, item in enumerate(coverage_items):
    col = i % 2
    row = i // 2
    xx = res_x + 0.5 + col * 3.0
    yy = res_y + res_h - 0.75 - row * 0.38
    color = '#2E7D32' if '100%' in item else '#F57F17' if '66%' in item else '#D84315'
    ax.text(xx, yy, item, ha='left', va='center', fontsize=9.5, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('output/img/data_fusion_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved: output/img/data_fusion_diagram.png')
