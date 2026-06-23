"""
航运在线 ZC 船舶（其他内贸船舶）批量采集 - Playwright 版

用法:
    python scripts/fetch_sol_zc_batch.py                  # 采集所有船
    python scripts/fetch_sol_zc_batch.py --limit 10       # 只采集前 10 艘
    python scripts/fetch_sol_zc_batch.py --resume         # 断点续采（跳过已采集）
    python scripts/fetch_sol_zc_batch.py --headless false # 显示浏览器

逻辑:
    1. 读取 output/ship_characteristics_db.csv 获取船名列表
    2. Playwright 打开航运在线 ZC 船舶页面
    3. 逐个搜索船名，提取：船名、载重吨、建造时间
    4. 结果写入 output/sol_zc_results.csv，支持断点续采

免费数据字段:
    - ship_name: 船名
    - tonnage: 载重吨（从搜索结果列表提取）
    - build_year: 建造时间
    - detail_url: 详细页面链接
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from playwright.sync_api import sync_playwright

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
RESULTS_CSV = OUTPUT_DIR / 'sol_zc_results.csv'

# 日志文件
LOG_FILE = OUTPUT_DIR / 'sol_zc_batch.log'
LOG_FILE.parent.mkdir(exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
        return len(text)
    def flush(self):
        for f in self.files:
            f.flush()

_log_fh = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)
sys.stdout = Tee(sys.stdout, _log_fh)

# Chrome 路径
CHROME_PATH = r'D:\18706\chrome-win64\chrome.exe'

# 限速参数
SEARCH_DELAY = 3.0       # 搜索间隔（秒）- 航运在线可能有反爬
BATCH_PAUSE = 30         # 每 N 艘暂停（秒）
BATCH_SIZE = 50          # 每 N 艘暂停一次

# 目标 URL
BASE_URL = 'https://tool.sol.com.cn/chinaships_other.asp'


def find_chrome() -> str | None:
    """查找 Chrome 可执行文件"""
    if Path(CHROME_PATH).exists():
        return CHROME_PATH
    candidates = [
        r'D:\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def load_ship_names(csv_path: Path) -> list[str]:
    """从 ship_characteristics_db.csv 读取船名列表"""
    df = pd.read_csv(csv_path)
    return df['ship_name'].dropna().unique().tolist()


def load_existing_results() -> set[str]:
    """加载已采集的船名（用于断点续采）"""
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(df['ship_name'].dropna().tolist())


def _init_browser(playwright, chrome_path: str, headless: bool):
    """启动浏览器并打开航运在线 ZC 船舶页面"""
    try:
        browser = playwright.chromium.launch(
            headless=headless,
            executable_path=chrome_path,
            args=['--no-sandbox'],
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800},
        )
        page = context.new_page()

        print(f'打开 {BASE_URL} ...', flush=True)
        page.goto(BASE_URL, timeout=60000, wait_until='domcontentloaded')
        time.sleep(3)
        print(f'页面标题: {page.title()}', flush=True)

        return browser, page
    except Exception as e:
        print(f'浏览器初始化失败: {e}', flush=True)
        return None, None


def search_ship(page, ship_name: str) -> dict | None:
    """
    搜索船名，返回搜索结果
    
    Returns:
        dict with keys: ship_name, tonnage, build_year, detail_url
        or None if not found
    """
    try:
        # 清空船名输入框并输入新船名
        name_input = page.locator('#shipname')
        name_input.fill('')
        name_input.fill(ship_name)
        
        # 点击精确查询按钮
        page.get_by_role('button', name='精确查询').click()
        
        # 等待页面加载
        time.sleep(2)
        
        # 检查是否有"暂无信息" - 表示没有找到
        page_text = page.content()
        if '暂无信息' in page_text:
            return {'ship_name': ship_name, 'tonnage': None, 'build_year': '', 'detail_url': '', 'found': False}
        
        # 查找结果表格中的行
        # 结果表格结构: 船名 | 载重吨 | 建造时间 | 详细
        result_rows = page.locator('table:has-text("船 名") tr').all()
        
        for row in result_rows:
            cells = row.locator('td').all()
            if len(cells) >= 3:
                # 检查第一列是否是船名
                name_cell = cells[0].inner_text().strip()
                if name_cell == ship_name:
                    # 找到匹配的船
                    tonnage_text = cells[1].inner_text().strip()
                    build_year = cells[2].inner_text().strip()
                    
                    # 提取详细链接
                    detail_link = ''
                    link = row.locator('a[href*="detail"]')
                    if link.count() > 0:
                        detail_link = link.first.get_attribute('href')
                        if detail_link and not detail_link.startswith('http'):
                            detail_link = f'https://tool.sol.com.cn/{detail_link}'
                    
                    # 解析载重吨
                    tonnage = None
                    if tonnage_text and tonnage_text != '--' and tonnage_text != '':
                        try:
                            tonnage = float(tonnage_text)
                        except ValueError:
                            pass
                    
                    return {
                        'ship_name': ship_name,
                        'tonnage': tonnage,
                        'build_year': build_year if build_year != '--' else '',
                        'detail_url': detail_link,
                        'found': True,
                    }
        
        # 没有找到匹配的船
        return {'ship_name': ship_name, 'tonnage': None, 'build_year': '', 'detail_url': '', 'found': False}
        
    except Exception as e:
        print(f'  搜索异常: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return None


def _save_results(results: list[dict]):
    """保存结果到 CSV"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
    print(f'已保存 {len(df)} 条记录到 {RESULTS_CSV}', flush=True)


def run_batch(ship_names: list[str], resume: bool, headless: bool, limit: int):
    """主采集流程"""
    chrome = find_chrome()
    if not chrome:
        print('未找到 Chrome，请安装或修改 CHROME_PATH', flush=True)
        sys.exit(1)

    # 断点续采：跳过已采集的
    done = load_existing_results() if resume else set()
    todo = [n for n in ship_names if n not in done]
    if limit > 0:
        todo = todo[:limit]

    print(f'总船名: {len(ship_names)}, 已采集: {len(done)}, 待采集: {len(todo)}', flush=True)
    if not todo:
        print('无待采集船，退出', flush=True)
        return

    results = []
    # 如果续采，先加载已有结果
    if resume and RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        results = existing_df.to_dict('records')

    with sync_playwright() as p:
        browser, page = _init_browser(p, chrome, headless)
        if not browser:
            return

        success = 0
        not_found = 0
        error_count = 0

        for i, name in enumerate(todo):
            print(f'\n[{i+1}/{len(todo)}] {name}', flush=True)

            try:
                # 检查浏览器是否还活着
                try:
                    page.title()
                except Exception:
                    print('  浏览器崩溃，重启中...', flush=True)
                    try:
                        browser.close()
                    except Exception:
                        pass
                    browser, page = _init_browser(p, chrome, headless)
                    if not browser:
                        print('  重启失败，保存已有结果后退出', flush=True)
                        _save_results(results)
                        return

                # 搜索
                result = search_ship(page, name)
                
                if result is None:
                    print(f'  搜索异常', flush=True)
                    results.append({
                        'ship_name': name,
                        'tonnage': None,
                        'build_year': '',
                        'detail_url': '',
                        'found': False,
                        'data_source': 'sol_zc_error',
                    })
                    error_count += 1
                    
                elif not result.get('found'):
                    print(f'  未找到', flush=True)
                    results.append({
                        'ship_name': name,
                        'tonnage': None,
                        'build_year': '',
                        'detail_url': '',
                        'found': False,
                        'data_source': 'sol_zc_not_found',
                    })
                    not_found += 1
                    
                else:
                    tonnage = result.get('tonnage')
                    build_year = result.get('build_year', '')
                    detail_url = result.get('detail_url', '')
                    
                    print(f'  找到! 载重吨={tonnage}, 建造年份={build_year}', flush=True)
                    results.append({
                        'ship_name': name,
                        'tonnage': tonnage,
                        'build_year': build_year,
                        'detail_url': detail_url,
                        'found': True,
                        'data_source': 'sol_zc',
                    })
                    success += 1

            except Exception as e:
                print(f'  处理异常: {e}', flush=True)
                results.append({
                    'ship_name': name,
                    'tonnage': None,
                    'build_year': '',
                    'detail_url': '',
                    'found': False,
                    'data_source': 'sol_zc_error',
                })
                error_count += 1

            # 每 10 艘保存一次断点
            if (i + 1) % 10 == 0:
                _save_results(results)

            # 每 BATCH_SIZE 艘暂停
            if (i + 1) % BATCH_SIZE == 0:
                print(f'\n--- 暂停 {BATCH_PAUSE}s (已采集 {i+1} 艘) ---', flush=True)
                time.sleep(BATCH_PAUSE)

            time.sleep(SEARCH_DELAY)

        try:
            browser.close()
        except Exception:
            pass

    # 最终保存
    _save_results(results)

    # 统计
    total = len(todo)
    print(f'\n========== 采集完成 ==========', flush=True)
    print(f'总计: {total}, 成功: {success}, 未找到: {not_found}, 异常: {error_count}', flush=True)
    print(f'结果: {RESULTS_CSV}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='航运在线 ZC 船舶批量采集（Playwright）')
    parser.add_argument('--limit', type=int, default=0, help='只采集前 N 艘（0=全部）')
    parser.add_argument('--resume', action='store_true', help='断点续采（跳过已采集）')
    parser.add_argument('--headless', type=str, default='true', help='是否无头模式 (true/false)')
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / 'output' / 'ship_characteristics_db.csv'
    if not csv_path.exists():
        print(f'船名文件不存在: {csv_path}', flush=True)
        sys.exit(1)

    ship_names = load_ship_names(csv_path)
    print(f'从 CSV 读取 {len(ship_names)} 个船名', flush=True)

    headless = args.headless.lower() != 'false'
    run_batch(ship_names, resume=args.resume, headless=headless, limit=args.limit)


if __name__ == '__main__':
    main()
