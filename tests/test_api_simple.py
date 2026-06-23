# -*- coding: utf-8 -*-
"""
简化版 API 测试脚本
只测试基本的 HTTP 请求，不依赖复杂的应用启动
"""

import requests
import json
import time

def test_api_endpoints():
    """测试 API 端点"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("=== 开始测试 API 端点 ===")
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(2)
    
    # 测试用例
    test_cases = [
        ("首页", "GET", "/"),
        ("船舶类型", "GET", "/api/ship_types"),
        ("船舶列表", "GET", "/api/ships"),
        ("拓扑节点", "GET", "/api/topology_nodes"),
        ("轨迹采样", "GET", "/api/trajectory_sample"),
    ]
    
    results = []
    
    for name, method, endpoint in test_cases:
        print(f"\n测试 {name} ({method} {endpoint})...")
        try:
            url = f"{base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, timeout=10)
            
            print(f"  状态码: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        if data.get('success'):
                            print(f"  OK 成功: {data.get('message', 'OK')}")
                            if 'data' in data:
                                if isinstance(data['data'], list):
                                    print(f"  数据量: {len(data['data'])} 条")
                                else:
                                    print(f"  数据类型: {type(data['data']).__name__}")
                        else:
                            print(f"  FAIL 业务错误: {data.get('message', '未知错误')}")
                    else:
                        print(f"  ✓ 响应格式: {type(data).__name__}")
                except json.JSONDecodeError:
                    print(f"  ✓ 响应内容: {response.text[:100]}...")
            else:
                print(f"  FAIL HTTP 错误: {response.status_code}")
                
            results.append((name, response.status_code, True))
            
        except requests.exceptions.ConnectionError:
            print(f"  FAIL 连接失败: 服务器可能未启动")
            results.append((name, 0, False))
        except requests.exceptions.Timeout:
            print(f"  FAIL 请求超时")
            results.append((name, 0, False))
        except Exception as e:
            print(f"  FAIL 测试失败: {e}")
            results.append((name, 0, False))
    
    # 测试路径规划 API
    print(f"\n测试 路径规划 (POST /api/plan)...")
    try:
        # 先获取节点数据
        nodes_response = requests.get(f"{base_url}/api/topology_nodes", timeout=10)
        if nodes_response.status_code == 200:
            nodes_data = nodes_response.json()
            if nodes_data.get('success') and nodes_data.get('data'):
                nodes = nodes_data['data']
                if len(nodes) >= 2:
                    start_node = nodes[0]
                    end_node = nodes[1]
                    
                    plan_data = {
                        'start_lat': start_node['lat'],
                        'start_lon': start_node['lon'],
                        'end_lat': end_node['lat'],
                        'end_lon': end_node['lon'],
                        'ship_type': '中型货船',
                        'ship_name': ''
                    }
                    
                    print(f"  测试路径规划: 从节点 {start_node['node_id']} 到节点 {end_node['node_id']}")
                    
                    plan_response = requests.post(
                        f"{base_url}/api/plan",
                        json=plan_data,
                        timeout=30
                    )
                    
                    print(f"  状态码: {plan_response.status_code}")
                    
                    if plan_response.status_code == 200:
                        plan_result = plan_response.json()
                        if plan_result.get('success'):
                            routes = plan_result.get('data', {}).get('routes', [])
                            print(f"  OK 路径规划成功，生成 {len(routes)} 条路线")
                            
                            recommended = plan_result.get('data', {}).get('recommended_path', {})
                            if recommended:
                                print(f"  推荐路线: 距离 {recommended.get('total_distance_km', 0)} km, "
                                      f"时间 {recommended.get('total_time_min', 0)} 分钟")
                        else:
                            print(f"  FAIL 路径规划失败: {plan_result.get('message', '未知错误')}")
                    else:
                        print(f"  FAIL HTTP 错误: {plan_response.status_code}")
                    
                    results.append(("路径规划", plan_response.status_code, plan_response.status_code == 200))
                else:
                    print("  FAIL 节点数据不足")
                    results.append(("路径规划", 0, False))
            else:
                print("  FAIL 无法获取节点数据")
                results.append(("路径规划", 0, False))
        else:
            print("  FAIL 无法获取节点数据")
            results.append(("路径规划", 0, False))
            
    except Exception as e:
        print(f"  FAIL 路径规划测试失败: {e}")
        results.append(("路径规划", 0, False))
    
    # 测试错误处理
    print(f"\n测试 错误处理 (POST /api/plan 无效数据)...")
    try:
        invalid_data = {
            'start_lat': 91,  # 无效纬度
            'start_lon': 181,  # 无效经度
            'end_lat': 0,
            'end_lon': 0,
            'ship_type': '中型货船'
        }
        
        error_response = requests.post(
            f"{base_url}/api/plan",
            json=invalid_data,
            timeout=10
        )
        
        print(f"  状态码: {error_response.status_code}")
        
        if error_response.status_code == 400:
            error_data = error_response.json()
            print(f"  OK 错误处理正常: {error_data.get('message', 'N/A')}")
            results.append(("错误处理", 400, True))
        else:
            print(f"  FAIL 错误处理可能有问题")
            results.append(("错误处理", error_response.status_code, False))
            
    except Exception as e:
        print(f"  FAIL 错误处理测试失败: {e}")
        results.append(("错误处理", 0, False))
    
    # 总结
    print("\n=== 测试总结 ===")
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("OK 所有测试通过！")
    else:
        print("FAIL 部分测试失败")
        for name, status, success in results:
            if not success:
                print(f"  - {name}: 状态码 {status}")
    
    return passed == total

if __name__ == '__main__':
    test_api_endpoints()