# 船舶选择器 NaN 崩溃修复与船型规范化 & 搜索下拉框遮挡修复

> 日期：2026-06-17
> 影响文件：`app.py`、`templates/index.html`
> 备份：`app.py` 修改前备份于 QoderWork 工作目录

---

## 1. 问题现象

前端"全部船型"下拉框展开后无任何船型选项，只有默认的"全部船型"一项。用户无法按船型筛选船舶，也无法选择具体船舶进行个性化路径规划。

## 2. 根因分析

### 2.1 直接原因：JSON 中出现非法 NaN

`output/ship_characteristics_db.csv` 中约 104 艘船的 `draft`（吃水深度）字段为空值。pandas 读取后变为 `NaN`，Flask 的 `jsonify()` 将其序列化为 JavaScript 原生 `NaN` 字面量。但 `NaN` 不是合法的 JSON 值（JSON 规范只允许数字、字符串、布尔、null、数组、对象），浏览器的 `JSON.parse()` 直接抛出：

```
SyntaxError: Unexpected token 'N', ..."","draft":NaN,"lengt"... is not valid JSON
```

`loadShips()` 函数未编写 `.catch()` 错误处理，Promise rejection 被静默吞掉，导致 `ship-type-filter` 下拉框永远停留在初始状态。

**验证方式：** 启动 app 后在浏览器控制台执行：

```javascript
fetch('/api/ships').then(r => r.json()).catch(e => console.error(e));
// 输出: SyntaxError: Unexpected token 'N' ...
```

### 2.2 结构性问题：船型名称不匹配

即便修复 NaN，还存在 CSV 船型与 `SHIP_TEMPLATES` 模板名称不匹配的问题：

| CSV ship_type | 数量 | SHIP_TEMPLATES 对应 | 匹配状态 |
|---|---|---|---|
| 货船 | 267 | 小型货船 / 中型货船 / 大型货船 | 名称不匹配 |
| 执法船 | 9 | （无） | 模板缺失 |
| 客船 | 8 | 客船 | 正常 |
| 油轮 | 7 | 油轮 | 正常 |
| 集装箱船 | 5 | 集装箱船 | 正常 |
| 渔船 | 5 | 渔船 | 正常 |
| 拖轮 | 1 | 拖船 | 一字之差 |
| 挖泥船 | 1 | （无） | 模板缺失 |
| 液体散货 | 1 | （无） | 模板缺失 |
| 游艇 | 1 | （无） | 模板缺失 |
| 翼船 | 1 | （无） | 模板缺失 |
| 其他 | 3 | （无） | 模板缺失 |

后果：用户选择"执法船"类型的船后，后端 `SHIP_TEMPLATES.get('执法船')` 返回 None，fallback 到"中型货船"模板，物理约束参数（吃水、船宽等）与真实船舶完全不符。

### 2.3 NaN 数据来源

shipxy 批量采集脚本（`scripts/fetch_shipxy_batch.py`）采集了 309 艘船的数据，其中 204 艘有真实吃水数据，但 105 艘在船讯网上也没有吃水信息。合并脚本（`scripts/merge_shipxy_results.py`）只覆盖有值的字段，因此这 105 艘船的 draft 在 CSV 中保持为空。

## 3. 修复方案

### 3.1 新增 `_safe_float()` 工具函数（app.py）

```python
def _safe_float(val, fallback=None):
    """安全的浮点数转换：NaN/None/空值 → fallback"""
    import math
    if val is None:
        return fallback
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return fallback
        return v
    except (ValueError, TypeError):
        return fallback
```

作用：统一拦截所有非法数值，确保 JSON 输出中不会出现 NaN/Inf。

### 3.2 新增 `_normalize_ship_type()` 船型规范化函数（app.py）

```python
def _normalize_ship_type(raw_type, length_m=None):
    if not raw_type or not isinstance(raw_type, str):
        return '中型货船'
    t = raw_type.strip()
    if t == '拖轮':
        return '拖船'          # 同义词统一
    if t == '货船':
        L = _safe_float(length_m, 63)
        if L < 55:
            return '小型货船'
        elif L < 100:
            return '中型货船'
        else:
            return '大型货船'
    return t
```

作用：把 CSV 中泛化的"货船"按船长细分为三种，把"拖轮"统一为"拖船"。

### 3.3 SHIP_TEMPLATES 扩充（app.py）

从 10 种增至 16 种，新增 6 种模板：

| 新增船型 | length | width | draft | height | tonnage | max_speed |
|---|---|---|---|---|---|---|
| 执法船 | 50 | 8 | 2.0 | 15 | 500 | 18 |
| 挖泥船 | 80 | 16 | 4.5 | 20 | 3000 | 6 |
| 液体散货 | 70 | 12 | 3.5 | 15 | 2000 | 8 |
| 游艇 | 30 | 6 | 1.5 | 10 | 100 | 20 |
| 翼船 | 25 | 8 | 1.0 | 8 | 50 | 30 |
| 其他 | 50 | 10 | 2.5 | 12 | 1000 | 8 |

参数来源：参考同类船舶的典型物理特征。

### 3.4 ship_db 加载逻辑重写（app.py `load_data()`）

改造前：

```python
ship_db[name] = {
    'draft': row.get('draft'),       # 可能是 NaN
    'ship_type': row.get('ship_type', '货船'),  # 可能是 '货船'（不匹配模板）
}
```

改造后：

```python
raw_length = _safe_float(row.get('length'))
normalized_type = _normalize_ship_type(raw_type, raw_length)
tpl = SHIP_TEMPLATES.get(normalized_type, SHIP_TEMPLATES['中型货船'])

ship_db[name] = {
    'draft': _safe_float(row.get('draft'), tpl['draft']),  # NaN → 模板兜底
    'ship_type': normalized_type,                            # 已规范化
}
```

### 3.5 `plan_paths()` 函数修复（app.py）

改造前：

```python
length=real.get('length') or SHIP_TEMPLATES.get(ship_type, {}).get('length', 100)
```

问题：`or` 对 0 值也会跳过（0 是 falsy），对 NaN 不会跳过（NaN 是 truthy）。

改造后：

```python
length=_safe_float(real.get('length'), tpl['length'])
```

### 3.6 `/api/ships` 端点输出清理（app.py）

所有数值字段改用 `_safe_float`，确保 JSON 合法。

### 3.7 `/api/ship_types` 端点更新（app.py）

改为返回模板类型与 ship_db 实际类型的并集，确保前端隐藏 select 能匹配所有船型。

### 3.8 前端优化（templates/index.html）

- `loadShips()` 增加 `.catch()` 错误处理，异常不再被静默吞掉
- 船型下拉框每项显示船舶数量（如"油轮 (7艘)"）
- 搜索结果列表顶部显示匹配数量摘要（如"共 309 艘"）
- 船舶列表项增加吨位显示
- 选中船舶信息栏样式优化，字段更清晰

## 4. 验证结果

| 检查项 | 结果 |
|---|---|
| ship_db 中 NaN 值数量 | 0 |
| `/api/ships` JSON 中 NaN 出现次数 | 0 |
| `/api/ships` 返回船舶总数 | 309 |
| `/api/ships` 返回船型种类 | 14 种（规范化后） |
| `/api/ship_types` 返回模板种类 | 16 种 |
| 前端下拉框选项数 | 15（含"全部船型 (309艘)"） |
| 船型过滤功能 | 正常（选"油轮"→显示 7 艘） |
| 选船后 ship_type 同步 | 正常（隐藏 select 值 = 选中船的规范化类型） |
| 路径规划 API（选中具体船） | 成功，使用真实船舶参数 |

## 5. 影响范围

| 文件 | 修改内容 |
|---|---|
| `app.py` | 新增 `_safe_float`、`_normalize_ship_type`；SHIP_TEMPLATES +6 种；`load_data()` 重写船舶加载；`plan_paths()` NaN 安全；`/api/ships` 和 `/api/ship_types` 端点更新 |
| `templates/index.html` | `loadShips()` 错误处理+数量显示；`filterShips()` 摘要行+吨位；`selectShip()` 样式 |

## 6. 经验教训

1. **pandas NaN ≠ JSON null**：pandas 的 NaN 经 Flask jsonify 变成 JavaScript 非法值，且前端 fetch 的 `.then()` 不会触发，错误完全静默。所有从 pandas 读取的数值字段在输出 JSON 前必须做 NaN 检查。

2. **前端 fetch 必须加 `.catch()`**：否则任何解析错误都会被静默吞掉，表现为"功能莫名其妙不工作"。

3. **`or` 不是万能的 fallback**：`value or default` 对 0（合法值）会错误跳过，对 NaN（非法值）不会跳过。应使用显式检查函数。

4. **数据源之间的命名规范必须对齐**：shipxy API 返回的 AIS 标准船型码（70-79=货船）与项目自定义的细分模板（小/中/大型货船）存在粒度差异，需要在加载时做规范化映射。

---

# 船舶搜索下拉框被按钮遮挡修复

> 日期：2026-06-17
> 影响文件：`templates/index.html`
> 类型：前端 CSS 定位 + 层叠上下文问题

---

## 7. 问题现象

左侧面板的"搜索船名"输入框展开下拉列表后，下拉内容被下方的"开始路径规划"按钮和"播放轨迹"按钮遮挡，用户无法正常浏览和选择船舶。

## 8. 根因分析

问题由两层 CSS 机制叠加导致：

### 8.1 第一层：overflow 裁剪

`.input-section` 设置了 `overflow-y: auto`（用于侧边栏内容溢出时滚动），形成滚动裁剪容器。原本 `position: absolute` 的下拉框作为 `.input-section` 的子孙元素，会被裁剪在滚动容器的可视区域内，无法延伸到下方的按钮区域之上。

### 8.2 第二层：CSS transform 破坏 fixed 定位（核心原因）

`.input-section > div` 有入场动画：

```css
.input-section > div {
  animation: fadeUp 0.5s ease both;
}
@keyframes fadeUp {
  0%   { transform: translateY(12px); opacity: 0; }
  100% { transform: translateY(0);    opacity: 1; }
}
```

`animation-fill-mode: both` 使动画结束后最终帧的样式保留。因此每个子 div 都残留了 `transform: matrix(1, 0, 0, 1, 0, 0)`（恒等变换）。

虽然视觉上无位移，但 CSS 规范规定：**任何带 `transform`（即使是恒等值）的元素都会为 `position: fixed` 后代创建新的包含块**。这导致下拉框的 `position: fixed` 不再相对于视口定位，而是相对于这个带 transform 的 div——实际偏移量达到 428px。

此外，`.sidebar` 自身的滑入动画 `sidebarSlideIn` 也有同样问题（`animation-fill-mode: both` 残留 `transform`），即使修复了子 div 的问题，sidebar 层级的 transform 仍然会干扰 fixed 定位。

### 8.3 验证方式

在浏览器控制台执行：

```javascript
// 检查下拉框祖先链的 transform
(function(){
  var el = document.getElementById('ship-dropdown');
  while(el && el !== document.documentElement) {
    var cs = getComputedStyle(el);
    console.log(el.tagName, el.id||'', el.className, 'transform:', cs.transform);
    el = el.parentElement;
  }
})();
// 修复前：至少一个祖先 div 有 matrix(1,0,0,1,0,0)
// 修复后：所有祖先 transform 为 none
```

## 9. 修复方案

### 9.1 将 `#ship-dropdown` 移出 `.input-section`（HTML）

把下拉框从 `.input-section` 内部的搜索框包裹 div 中移出，放到 `.sidebar` 的直接子级。这样彻底脱离了 `.input-section` 的 `overflow-y: auto` 裁剪和所有子 div 的 transform 干扰。

修改前：

```html
<div class="input-section">
  ...
  <div style="position:relative;">
    <input id="ship-search" ...>
    <div id="ship-dropdown"></div>  <!-- 被 overflow 和 transform 双重影响 -->
  </div>
  ...
</div>
```

修改后：

```html
<div class="input-section">
  ...
  <div style="position:relative;">
    <input id="ship-search" ...>
    <!-- dropdown 已从此处移出 -->
  </div>
  ...
</div>  <!-- end .input-section -->

<!-- 放在 sidebar 直接子级，脱离 input-section 的 overflow/transform -->
<div id="ship-dropdown"></div>
```

### 9.2 添加 fixed 定位 CSS

```css
#ship-dropdown {
  position: fixed;
  z-index: 10001;
  max-height: 200px;
  overflow-y: auto;
  background: var(--bg-elevated);
  border: 1px solid var(--border-light);
  border-radius: 0 0 var(--radius-sm) var(--radius-sm);
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  display: none;
}
```

### 9.3 新增 `positionShipDropdown()` 动态定位函数（JS）

由于 `position: fixed` 脱离文档流，需要用 JS 动态计算下拉框位置：

```javascript
function positionShipDropdown() {
  const search = document.getElementById('ship-search');
  const dropdown = document.getElementById('ship-dropdown');
  if (!search || !dropdown) return;
  const rect = search.getBoundingClientRect();
  dropdown.style.left = rect.left + 'px';
  dropdown.style.top = (rect.bottom + 1) + 'px';
  dropdown.style.width = rect.width + 'px';
}
```

### 9.4 `filterShips()` 调用定位

当下拉框显示时调用定位：

```javascript
if (filtered.length > 0) {
  dropdown.style.display = 'block';
  positionShipDropdown();
} else {
  dropdown.style.display = 'none';
}
```

### 9.5 滚动/resize 事件监听

```javascript
// input-section 滚动时重新定位
var inputSection = document.querySelector('.input-section');
if (inputSection) {
  inputSection.addEventListener('scroll', function() {
    var dd = document.getElementById('ship-dropdown');
    if (dd && dd.style.display === 'block') positionShipDropdown();
  });
}
// 窗口 resize 时重新定位
window.addEventListener('resize', function() {
  var dd = document.getElementById('ship-dropdown');
  if (dd && dd.style.display === 'block') positionShipDropdown();
});
```

### 9.6 sidebar 动画结束后清除 transform

```javascript
var sidebarEl = document.querySelector('.sidebar');
if (sidebarEl) {
  sidebarEl.addEventListener('animationend', function() {
    sidebarEl.style.animation = 'none';
    sidebarEl.style.transform = 'none';
  });
}
```

## 10. 验证结果

| 检查项 | 修复前 | 修复后 |
|---|---|---|
| 下拉框 position | absolute（被 overflow 裁剪） | fixed（脱离裁剪） |
| 祖先链 transform | matrix(1,0,0,1,0,0)（破坏 fixed） | none |
| 下拉框 top vs 搜索框 bottom | 偏移 428px | 间距 1px |
| 滚动后位置跟踪 | 不适用 | 自动跟随（gap=1px） |
| 选船后关闭 | 正常 | 正常 |
| 点击外部关闭 | 正常 | 正常 |
| "开始路径规划"按钮遮挡 | 遮挡 | 不遮挡 |
| "播放轨迹"按钮遮挡 | 遮挡 | 不遮挡 |

## 11. 影响范围

| 文件 | 修改内容 |
|---|---|
| `templates/index.html` | `#ship-dropdown` HTML 位置迁移；新增 CSS fixed 样式；新增 `positionShipDropdown()` 函数；`filterShips()` 调用定位；scroll/resize 监听；sidebar animationend 清理 |

## 12. 经验教训

1. **`animation-fill-mode: both` 的副作用**：CSS 动画结束后残留的 `transform`（即使是恒等值 `matrix(1,0,0,1,0,0)`）会创建新的包含块，导致 `position: fixed` 子孙元素定位异常。这类问题非常隐蔽，因为视觉上看不出 transform 的存在。

2. **`position: fixed` 不是万能的**：虽然 fixed 通常相对于视口定位，但任何祖先元素的 `transform`、`perspective`、`filter`、`will-change` 等属性都会改变其行为。在复杂布局中使用 fixed 定位时，必须检查完整祖先链。

3. **overflow + fixed 的组合**：将 fixed 元素放在 `overflow: auto/scroll` 容器内虽然不会被裁剪（fixed 脱离滚动容器），但如果祖先链上有 transform，fixed 退化为相对于 transform 祖先定位，仍然可能被裁剪。

4. **动态定位是可靠方案**：对于需要"浮在一切之上"的下拉框/弹出框，`position: fixed` + JS 动态计算位置（`getBoundingClientRect()`）是最可靠的方案，但必须同时处理滚动和 resize 事件。
