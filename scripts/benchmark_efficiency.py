# -*- coding: utf-8 -*-
"""
算法效率基准测试脚本
记录每个Task的运行时间和峰值内存，输出到 output/efficiency_report.txt
"""

import time
import tracemalloc
import os
import sys
import json
import platform
import psutil

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TaskManager


def get_system_info():
    """获取系统信息"""
    return {
        'cpu': platform.processor() or platform.machine(),
        'memory_gb': round(psutil.virtual_memory().total / 1024**3, 1),
        'python': platform.python_version(),
        'os': f"{platform.system()} {platform.release()}",
    }


def benchmark_task(task_func, task_name):
    """运行单个任务并记录时间和内存"""
    tracemalloc.start()
    start = time.time()
    try:
        result = task_func()
        success = result if result is not None else True
        error = None
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()
    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'task': task_name,
        'time_s': round(elapsed, 2),
        'peak_memory_mb': round(peak / 1024 / 1024, 2),
        'success': success,
        'error': error,
    }


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    sys_info = get_system_info()
    print(f"[INFO] 系统: {sys_info['os']}, CPU: {sys_info['cpu']}, "
          f"内存: {sys_info['memory_gb']}GB, Python: {sys_info['python']}")

    tm = TaskManager(output_dir=output_dir)
    results = []

    # Task 1-6 逐个测试
    task_map = {
        1: lambda: tm._task1_preprocess(force=False),
        2: lambda: tm._task2_extract_nodes(force=False),
        3: lambda: tm._task3_cluster_nodes(force=False),
        4: lambda: tm._task4_build_topology(force=False),
        5: lambda: tm._task5_weight_model(force=False),
        6: lambda: tm._task6_visualize(force=False),
    }

    for task_id, func in task_map.items():
        name = f"Task{task_id}"
        print(f"[BENCHMARK] Running {name}...")
        result = benchmark_task(func, name)
        results.append(result)
        status = "OK" if result['success'] else f"FAIL: {result['error']}"
        print(f"  -> {result['time_s']}s, {result['peak_memory_mb']}MB, {status}")

    # Task 7: 单次导航决策
    print("[BENCHMARK] Running Task7 (10 ship types)...")
    result = benchmark_task(lambda: tm._task7_navigation(force=False), 'Task7(10船型)')
    results.append(result)
    status = "OK" if result['success'] else f"FAIL: {result['error']}"
    print(f"  -> {result['time_s']}s, {result['peak_memory_mb']}MB, {status}")

    # 输出报告
    report_path = os.path.join(output_dir, 'efficiency_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("算法效率基准测试报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"测试环境:\n")
        f.write(f"  CPU: {sys_info['cpu']}\n")
        f.write(f"  内存: {sys_info['memory_gb']}GB\n")
        f.write(f"  Python: {sys_info['python']}\n")
        f.write(f"  OS: {sys_info['os']}\n")
        f.write(f"  数据量: 1,100,000条AIS轨迹\n\n")
        f.write(f"{'任务':<25} {'运行时间(s)':<15} {'峰值内存(MB)':<15} {'状态'}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            status = "成功" if r['success'] else f"失败: {r['error']}"
            f.write(f"{r['task']:<25} {r['time_s']:<15.2f} {r['peak_memory_mb']:<15.2f} {status}\n")
        f.write("\n")
        total_time = sum(r['time_s'] for r in results)
        max_memory = max(r['peak_memory_mb'] for r in results)
        f.write(f"全流程总运行时间: {total_time:.2f}s ({total_time/60:.1f}min)\n")
        f.write(f"单任务最大内存: {max_memory:.2f}MB\n")

    # 同时输出 JSON 便于程序读取
    json_path = os.path.join(output_dir, 'efficiency_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'system': sys_info, 'results': results}, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {report_path}")
    print(f"JSON已保存: {json_path}")


if __name__ == '__main__':
    main()
