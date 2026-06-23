# -*- coding: utf-8 -*-
"""
水上航道智能路径规划系统 - 浏览器自动化测试
使用 Playwright 测试 Web 应用的主要功能
"""

from playwright.sync_api import sync_playwright
import json
import time

def test_app():
    """测试应用的主要功能"""
    
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        print("=== 开始测试水上航道智能路径规划系统 ===")
        
        # 1. 测试首页加载
        print("\n1. 测试首页加载...")
        try:
            page.goto('http://127.0.0.1:5000')
            page.wait_for_load_state('networkidle')
            
            # 检查页面标题
            title = page.title()
            print(f"   页面标题: {title}")
            
            # 截图保存
            page.screenshot(path='output/test_homepage.png', full_page=True)
            print("   首页截图已保存: output/test_homepage.png")
            
            # 检查页面内容
            content = page.content()
            if "水上航道智能路径规划" in content:
                print("   ✓ 首页加载成功")
            else:
                print("   ✗ 首页内容可能有问题")
                
        except Exception as e:
            print(f"   ✗ 首页加载失败: {e}")
        
        # 2. 测试船舶类型 API
        print("\n2. 测试船舶类型 API...")
        try:
            response = page.goto('http://127.0.0.1:5000/api/ship_types')
            if response and response.ok:
                data = response.json()
                if data.get('success'):
                    ship_types = data.get('data', [])
                    print(f"   ✓ 船舶类型 API 正常，返回 {len(ship_types)} 种类型")
                    print(f"   船舶类型: {ship_types[:5]}...")  # 显示前5个
                else:
                    print(f"   ✗ API 返回错误: {data.get('message')}")
            else:
                print(f"   ✗ API 请求失败: {response.status if response else '无响应'}")
        except Exception as e:
            print(f"   ✗ 船舶类型 API 测试失败: {e}")
        
        # 3. 测试船舶列表 API
        print("\n3. 测试船舶列表 API...")
        try:
            response = page.goto('http://127.0.0.1:5000/api/ships')
            if response and response.ok:
                data = response.json()
                if data.get('success'):
                    ships = data.get('data', [])
                    print(f"   ✓ 船舶列表 API 正常，返回 {len(ships)} 艘船舶")
                    if ships:
                        print(f"   示例船舶: {ships[0].get('ship_name', 'N/A')}")
                else:
                    print(f"   ✗ API 返回错误: {data.get('message')}")
            else:
                print(f"   ✗ API 请求失败: {response.status if response else '无响应'}")
        except Exception as e:
            print(f"   ✗ 船舶列表 API 测试失败: {e}")
        
        # 4. 测试拓扑节点 API
        print("\n4. 测试拓扑节点 API...")
        try:
            response = page.goto('http://127.0.0.1:5000/api/topology_nodes')
            if response and response.ok:
                data = response.json()
                if data.get('success'):
                    nodes = data.get('data', [])
                    print(f"   ✓ 拓扑节点 API 正常，返回 {len(nodes)} 个节点")
                    if nodes:
                        print(f"   示例节点: ID={nodes[0].get('node_id')}, 位置=({nodes[0].get('lat')}, {nodes[0].get('lon')})")
                else:
                    print(f"   ✗ API 返回错误: {data.get('message')}")
            else:
                print(f"   ✗ API 请求失败: {response.status if response else '无响应'}")
        except Exception as e:
            print(f"   ✗ 拓扑节点 API 测试失败: {e}")
        
        # 5. 测试路径规划 API（使用模拟数据）
        print("\n5. 测试路径规划 API...")
        try:
            # 先获取一些节点数据用于测试
            page.goto('http://127.0.0.1:5000/api/topology_nodes')
            nodes_response = page.evaluate('() => document.body.innerText')
            nodes_data = json.loads(nodes_response) if nodes_response else {}
            
            if nodes_data.get('success') and nodes_data.get('data'):
                nodes = nodes_data['data']
                if len(nodes) >= 2:
                    # 使用前两个节点作为起终点
                    start_node = nodes[0]
                    end_node = nodes[1]
                    
                    # 构建路径规划请求
                    plan_data = {
                        'start_lat': start_node['lat'],
                        'start_lon': start_node['lon'],
                        'end_lat': end_node['lat'],
                        'end_lon': end_node['lon'],
                        'ship_type': '中型货船',
                        'ship_name': ''
                    }
                    
                    print(f"   测试路径规划: 从节点 {start_node['node_id']} 到节点 {end_node['node_id']}")
                    
                    # 发送 POST 请求
                    page.goto('http://127.0.0.1:5000/api/plan', wait_until='networkidle')
                    
                    # 使用 JavaScript 发送 POST 请求
                    plan_result = page.evaluate('''async (planData) => {
                        const response = await fetch('/api/plan', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(planData)
                        });
                        return await response.json();
                    }''', plan_data)
                    
                    if plan_result and plan_result.get('success'):
                        routes = plan_result.get('data', {}).get('routes', [])
                        print(f"   ✓ 路径规划成功，生成 {len(routes)} 条路线")
                        
                        # 显示推荐路线信息
                        recommended = plan_result.get('data', {}).get('recommended_path', {})
                        if recommended:
                            print(f"   推荐路线: 距离 {recommended.get('total_distance_km', 0)} km, "
                                  f"时间 {recommended.get('total_time_min', 0)} 分钟")
                    else:
                        print(f"   ✗ 路径规划失败: {plan_result.get('message', '未知错误')}")
                else:
                    print("   ✗ 节点数据不足，无法测试路径规划")
            else:
                print("   ✗ 无法获取节点数据进行路径规划测试")
                
        except Exception as e:
            print(f"   ✗ 路径规划 API 测试失败: {e}")
        
        # 6. 测试轨迹采样 API
        print("\n6. 测试轨迹采样 API...")
        try:
            response = page.goto('http://127.0.0.1:5000/api/trajectory_sample')
            if response and response.ok:
                data = response.json()
                if data.get('success'):
                    trajectories = data.get('trajectories', [])
                    print(f"   ✓ 轨迹采样 API 正常，返回 {len(trajectories)} 条轨迹")
                else:
                    print(f"   ✗ API 返回错误: {data.get('error')}")
            else:
                print(f"   ✗ API 请求失败: {response.status if response else '无响应'}")
        except Exception as e:
            print(f"   ✗ 轨迹采样 API 测试失败: {e}")
        
        # 7. 测试交互功能（点击地图）
        print("\n7. 测试前端交互功能...")
        try:
            page.goto('http://127.0.0.1:5000')
            page.wait_for_load_state('networkidle')
            
            # 截图保存当前状态
            page.screenshot(path='output/test_interaction_before.png')
            
            # 尝试查找并点击地图元素
            map_element = page.locator('#map')
            if map_element.count() > 0:
                print("   ✓ 找到地图元素")
                
                # 尝试点击地图中心位置
                map_element.click(position={'x': 400, 'y': 300})
                page.wait_for_timeout(1000)  # 等待响应
                
                # 截图保存点击后状态
                page.screenshot(path='output/test_interaction_after.png')
                print("   ✓ 地图交互测试完成")
            else:
                print("   ✗ 未找到地图元素")
                
        except Exception as e:
            print(f"   ✗ 前端交互测试失败: {e}")
        
        # 8. 测试错误处理
        print("\n8. 测试错误处理...")
        try:
            # 测试无效的路径规划请求
            invalid_plan = {
                'start_lat': 91,  # 无效纬度
                'start_lon': 181,  # 无效经度
                'end_lat': 0,
                'end_lon': 0,
                'ship_type': '中型货船'
            }
            
            page.goto('http://127.0.0.1:5000/api/plan', wait_until='networkidle')
            
            error_result = page.evaluate('''async (planData) => {
                const response = await fetch('/api/plan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(planData)
                });
                return {
                    status: response.status,
                    data: await response.json()
                };
            }''', invalid_plan)
            
            if error_result and error_result.get('status') == 400:
                print("   ✓ 错误处理正常，返回 400 状态码")
                print(f"   错误信息: {error_result.get('data', {}).get('message', 'N/A')}")
            else:
                print("   ✗ 错误处理可能有问题")
                
        except Exception as e:
            print(f"   ✗ 错误处理测试失败: {e}")
        
        # 关闭浏览器
        browser.close()
        
        print("\n=== 测试完成 ===")
        print("测试结果已保存到 output/ 目录")
        
        return True

if __name__ == '__main__':
    test_app()