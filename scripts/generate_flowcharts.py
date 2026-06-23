"""生成技术路线流程图和改进A*算法流程图（使用matplotlib）"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

def draw_technical_roadmap(output_path):
    """绘制技术路线流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义颜色
    colors = {
        'input': '#e1f5fe',
        'process': '#ffffff',
        'output': '#c8e6c9',
        'final': '#fff3e0',
        'border': '#0288d1'
    }

    # 定义节点位置和内容
    nodes = [
        # (x, y, width, height, text, color_key)
        (7, 9.2, 3.5, 0.7, '原始AIS数据\n110.6万条', 'input'),
        (7, 8.0, 3.5, 0.7, '数据预处理\nIsolationForest + 卡尔曼滤波', 'process'),
        (7, 6.8, 3.5, 0.7, '节点提取\nDouglas-Peucker + 曲率检测', 'process'),
        (7, 5.6, 3.5, 0.7, '节点聚类\nHDBSCAN + KDE', 'process'),
        (7, 4.4, 3.5, 0.7, '拓扑网络构建\nHMM + Viterbi解码', 'process'),
        (3.5, 3.2, 3.2, 0.7, '416节点/506边\n航道拓扑网络', 'output'),
        (3.5, 2.0, 3.2, 0.7, '动态权重建模\n28维特征 + PNA', 'process'),
        (10.5, 2.0, 3.2, 0.7, '船舶特征检索\n309艘船舶数据库', 'process'),
        (7, 0.8, 3.5, 0.7, '改进A*路径规划\n物理约束 + 风险感知', 'process'),
        (7, -0.4, 3.5, 0.7, '多目标路径输出\n6种策略 × 10种船型', 'output'),
        (7, -1.6, 3.5, 0.7, 'Web可视化导航\nFlask + Leaflet.js', 'final'),
    ]

    # 绘制节点
    for x, y, w, h, text, color_key in nodes:
        fancy_box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=colors[color_key],
            edgecolor=colors['border'],
            linewidth=2
        )
        ax.add_patch(fancy_box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

    # 定义箭头连接
    arrows = [
        (7, 8.85, 7, 8.35),   # 原始数据 -> 预处理
        (7, 7.65, 7, 7.15),   # 预处理 -> 节点提取
        (7, 6.45, 7, 5.95),   # 节点提取 -> 聚类
        (7, 5.25, 7, 4.75),   # 聚类 -> 拓扑构建
        (7, 4.05, 3.5, 3.55), # 拓扑构建 -> 拓扑网络
        (3.5, 2.85, 3.5, 2.35), # 拓扑网络 -> 动态权重
        (7, 4.05, 10.5, 2.35), # 拓扑构建 -> 船舶检索
        (3.5, 1.65, 7, 1.15),  # 动态权重 -> 路径规划
        (10.5, 1.65, 7, 1.15), # 船舶检索 -> 路径规划
        (7, 0.45, 7, 0.05),   # 路径规划 -> 多目标输出
        (7, -0.75, 7, -1.25), # 多目标输出 -> Web可视化
    ]

    # 绘制箭头
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#616161', lw=1.5))

    # 添加标题
    ax.text(7, 9.8, '技术路线流程图', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_path}")


def draw_astar_flowchart(output_path):
    """绘制改进A*算法流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # 定义颜色
    colors = {
        'input': '#e1f5fe',
        'process': '#ffffff',
        'output': '#c8e6c9',
        'decision': '#fff9c4',
        'prune': '#ffcdd2',
        'final': '#fff3e0',
        'border': '#0288d1'
    }

    # 定义节点
    nodes = [
        # 输入
        (6, 13.2, 4, 0.7, '输入: 起点S, 终点T, 船舶信息', 'input'),

        # 船舶检索
        (6, 12.0, 3.5, 0.7, '船舶特征检索', 'decision'),

        # 检索方式
        (2, 10.8, 2.5, 0.6, '船名精确查询', 'process'),
        (5, 10.8, 2.5, 0.6, '关键词匹配', 'process'),
        (8.5, 10.8, 2.5, 0.6, '航速推断', 'process'),
        (11, 10.8, 2, 0.6, '模板兜底', 'process'),

        # 获取参数
        (6, 9.6, 3.5, 0.7, '获取物理参数\n吃水/船高/船宽', 'process'),

        # 初始化
        (6, 8.4, 3.5, 0.7, '初始化A*搜索\nOpenList, ClosedList', 'process'),

        # 取出节点
        (6, 7.2, 3.5, 0.7, '取出f(n)最小节点n', 'process'),

        # 判断是否终点
        (6, 6.0, 2.5, 0.7, 'n == 终点T?', 'decision'),

        # 扩展邻居
        (6, 4.8, 3.5, 0.7, '扩展邻居节点', 'process'),

        # 物理约束校验
        (6, 3.6, 3.5, 0.7, '物理约束校验', 'decision'),

        # 剪枝
        (10, 3.6, 2.5, 0.7, '剪枝: 不可行', 'prune'),

        # 计算代价
        (6, 2.4, 3.5, 0.7, '计算综合代价\nf(n) = g(n) + h(n)', 'process'),

        # 更新OpenList
        (6, 1.2, 3.5, 0.7, '更新OpenList', 'process'),

        # 回溯输出
        (2, 5.0, 2.5, 0.7, '回溯路径输出', 'output'),

        # 6种策略
        (2, 3.8, 2.5, 0.7, '6种策略输出', 'final'),
    ]

    # 绘制节点
    for x, y, w, h, text, color_key in nodes:
        if color_key == 'decision':
            # 菱形判断框
            diamond = plt.Polygon(
                [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                facecolor=colors[color_key],
                edgecolor=colors['border'],
                linewidth=2
            )
            ax.add_patch(diamond)
        else:
            fancy_box = FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.1",
                facecolor=colors[color_key],
                edgecolor=colors['border'],
                linewidth=2
            )
            ax.add_patch(fancy_box)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold')

    # 定义箭头连接
    arrows = [
        # 主流程
        (6, 12.85, 6, 12.35),   # 输入 -> 检索
        (6, 11.65, 6, 9.95),    # 检索 -> 获取参数
        (6, 9.25, 6, 8.75),     # 获取参数 -> 初始化
        (6, 8.05, 6, 7.55),     # 初始化 -> 取出节点
        (6, 6.85, 6, 6.35),     # 取出节点 -> 判断终点
        (6, 5.65, 6, 5.15),     # 判断终点 -> 扩展邻居
        (6, 4.45, 6, 3.95),     # 扩展邻居 -> 约束校验
        (6, 3.25, 6, 2.75),     # 约束校验 -> 计算代价
        (6, 2.05, 6, 1.55),     # 计算代价 -> 更新OpenList
        (6, 0.85, 6, 0.55),     # 更新OpenList -> 循环回取出节点（这里简化）

        # 终点判断到输出
        (4.75, 6.0, 2, 5.0),    # 判断终点 -> 回溯输出
        (2, 4.65, 2, 4.15),     # 回溯输出 -> 策略输出

        # 约束剪枝
        (7.75, 3.6, 8.75, 3.6), # 约束校验 -> 剪枝

        # 检索方式到获取参数
        (2, 10.5, 4.25, 9.95),
        (5, 10.5, 5.25, 9.95),
        (8.5, 10.5, 6.75, 9.95),
        (11, 10.5, 7.75, 9.95),
    ]

    # 绘制箭头
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#616161', lw=1.5))

    # 添加"是"/"否"标签
    ax.text(4.75, 6.2, '是', fontsize=8, color='green')
    ax.text(6.5, 5.7, '否', fontsize=8, color='red')

    # 添加标题
    ax.text(6, 13.8, '改进A*算法流程图', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'output', 'img')

    # 确保输出目录存在
    os.makedirs(img_dir, exist_ok=True)

    # 生成技术路线图
    draw_technical_roadmap(os.path.join(img_dir, 'technical_roadmap.png'))

    # 生成改进A*流程图
    draw_astar_flowchart(os.path.join(img_dir, 'improved_astar_flowchart.png'))

    print("All flowcharts generated successfully!")
"""生成技术路线流程图和改进A*算法流程图"""
import subprocess
import os

def generate_mermaid_png(mmd_file, output_file):
    """使用mermaid-cli生成PNG"""
    cmd = [
        'mmdc',
        '-i', mmd_file,
        '-o', output_file,
        '-w', '1200',
        '-H', '800',
        '--backgroundColor', 'white'
    ]
    subprocess.run(cmd, check=True)
    print(f"Generated: {output_file}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'output', 'img')
    scripts_dir = os.path.join(base_dir, 'scripts')

    # 确保输出目录存在
    os.makedirs(img_dir, exist_ok=True)

    # 生成技术路线图
    generate_mermaid_png(
        os.path.join(scripts_dir, 'technical_roadmap.mmd'),
        os.path.join(img_dir, 'technical_roadmap.png')
    )

    # 生成改进A*流程图
    generate_mermaid_png(
        os.path.join(scripts_dir, 'improved_astar.mmd'),
        os.path.join(img_dir, 'improved_astar_flowchart.png')
    )

    print("All flowcharts generated successfully!")
