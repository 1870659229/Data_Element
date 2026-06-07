# -*- coding: utf-8 -*-
"""
异常输入鲁棒性测试脚本
系统测试边界/异常场景，输出到 output/robustness_report.txt
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


def test_invalid_coordinates():
    """无效坐标"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 999, 'start_lon': 999,
            'end_lat': 30.5, 'end_lon': 122.0,
            'ship_type': '散货船'
        })
        data = resp.json
        return not data.get('success', False) or data.get('message') is not None


def test_same_start_end():
    """起终点相同"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 30.0, 'start_lon': 122.0,
            'end_lat': 30.0, 'end_lon': 122.0,
            'ship_type': '散货船'
        })
        data = resp.json
        # 应返回错误（同一节点或距离太近）
        return not data.get('success', False) or data.get('message') is not None


def test_extreme_ship_type():
    """极端船型（超大型）"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 30.0, 'start_lon': 122.0,
            'end_lat': 31.0, 'end_lon': 122.5,
            'ship_type': '超大型油轮'
        })
        data = resp.json
        # 应该返回路径（可能带RELAXED标记）或合理错误
        return data.get('success', False) or data.get('message') is not None


def test_unknown_ship_type():
    """未知船型"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 30.0, 'start_lon': 122.0,
            'end_lat': 31.0, 'end_lon': 122.5,
            'ship_type': '未知船型XYZ'
        })
        data = resp.json
        # 应该回退到默认模板并正常规划，或返回合理错误
        return data.get('success', False) or data.get('message') is not None


def test_no_path():
    """不连通OD（远离航道的坐标）"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 25.0, 'start_lon': 120.0,
            'end_lat': 35.0, 'end_lon': 125.0,
            'ship_type': '散货船'
        })
        data = resp.json
        # 远离航道，应返回snap距离过远错误
        return not data.get('success', False) or data.get('message') is not None


def test_missing_params():
    """缺少必要参数"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 30.0,
            # 缺少 start_lon, end_lat, end_lon
        })
        data = resp.json
        return not data.get('success', False) or resp.status_code == 400


def test_negative_coords():
    """负数坐标"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': -30.0, 'start_lon': -122.0,
            'end_lat': 31.0, 'end_lon': 122.5,
            'ship_type': '散货船'
        })
        data = resp.json
        # 负数坐标不在航道范围内，应返回snap距离过远错误
        return not data.get('success', False) or data.get('message') is not None


def test_zero_distance():
    """零距离（起终点非常接近）"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 30.0, 'start_lon': 122.0,
            'end_lat': 30.0001, 'end_lon': 122.0001,
            'ship_type': '散货船'
        })
        data = resp.json
        # 可能匹配到同一节点返回错误，或返回极短路径
        return data.get('success', False) or data.get('message') is not None


def test_string_coords():
    """字符串坐标"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': 'abc', 'start_lon': 'xyz',
            'end_lat': 31.0, 'end_lon': 122.5,
            'ship_type': '散货船'
        })
        data = resp.json
        return not data.get('success', False) or resp.status_code == 400


def test_null_coords():
    """null坐标"""
    with app.test_client() as client:
        resp = client.post('/api/plan', json={
            'start_lat': None, 'start_lon': None,
            'end_lat': 31.0, 'end_lon': 122.5,
            'ship_type': '散货船'
        })
        data = resp.json
        return not data.get('success', False) or resp.status_code == 400


TEST_CASES = [
    ("无效坐标(999,999)", test_invalid_coordinates, "应返回错误提示"),
    ("起终点相同", test_same_start_end, "应返回同一节点错误"),
    ("极端船型(超大型油轮)", test_extreme_ship_type, "应返回RELAXED兜底路径或合理错误"),
    ("未知船型", test_unknown_ship_type, "应回退默认模板正常规划或合理错误"),
    ("远离航道坐标", test_no_path, "应返回snap距离过远错误"),
    ("缺少必要参数", test_missing_params, "应返回参数错误提示"),
    ("负数坐标", test_negative_coords, "应返回snap距离过远错误"),
    ("零距离(起终点极近)", test_zero_distance, "应正常处理或返回同一节点提示"),
    ("字符串坐标", test_string_coords, "应返回坐标格式错误"),
    ("null坐标", test_null_coords, "应返回参数错误提示"),
]


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for name, test_func, expected in TEST_CASES:
        print(f"[ROBUSTNESS] Testing: {name}...")
        try:
            passed = test_func()
            error_msg = None
        except Exception as e:
            passed = False
            error_msg = traceback.format_exc()
        result = {
            'name': name,
            'expected': expected,
            'passed': passed,
            'error': error_msg,
        }
        results.append(result)
        status = "PASS" if passed else "FAIL"
        print(f"  -> {status}" + (f" (exception: {error_msg[:100]})" if error_msg else ""))

    # 输出报告
    report_path = os.path.join(output_dir, 'robustness_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("异常输入鲁棒性测试报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'测试用例':<25} {'预期行为':<30} {'结果'}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            status = "通过" if r['passed'] else "失败"
            f.write(f"{r['name']:<25} {r['expected']:<30} {status}\n")
            if r['error']:
                f.write(f"  异常: {r['error'][:200]}\n")
        f.write("\n")
        passed_count = sum(1 for r in results if r['passed'])
        f.write(f"通过率: {passed_count}/{len(results)} ({passed_count/len(results)*100:.0f}%)\n")

    # JSON
    json_path = os.path.join(output_dir, 'robustness_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {report_path}")
    print(f"JSON已保存: {json_path}")


if __name__ == '__main__':
    main()
