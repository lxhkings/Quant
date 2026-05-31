# Web UI 体验重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `quant/web/` 从 6 页平铺的单文件 `app.py` 重构成「漏斗流程 + 引导」的 streamlit 原生多页应用，计算逻辑零改动。

**Architecture:** 拆 `app.py` 为首页（总览）+ `pages/` 目录 6 个步骤页 + `_shared.py` 共用组件（步骤导航、术语 help、数据模式选择器）。每页只渲染 UI 并调用既有 `viewmodel`/`select` 函数，不改任何计算逻辑。

**Tech Stack:** streamlit 1.58（原生多页 `pages/`、`st.page_link`、`st.metric`）、altair 6.1（既有条形图）、pytest（仅防回归，不新增 UI 测试）。

---

## 背景与约束

参见设计文档 `docs/superpowers/specs/2026-05-31-web-ux-redesign-design.md`。要点：

- **定位**：自用研究工具。重点是流程清晰 + 黑话翻译，不是多人协作或生产信号。
- **零逻辑改动**：`quant/web/viewmodel.py`、`quant/select/screen.py`、`quant/report/`、`quant/validate/` 全部不动。现有 128 个 pytest 用例必须保持全绿。
- **不新增 UI 测试**：streamlit UI 层本就无自动化测试，自用工具遵循 YAGNI。每个任务的验证 = 跑既有 pytest 防回归 + 手动启动目测。
- **mode 不跨页**：streamlit 原生多页下全局状态不自动保留，每页各自渲染数据模式选择器，默认 `research`，与现状行为一致。

## 既有事实（实现时依赖，已核实）

- streamlit 版本 1.58.0，支持 `st.page_link(page, label=...)`、`st.metric`、`st.columns`、`st.Page`/`pages/` 目录两种多页机制。本计划用 **`pages/` 目录约定**（文件名数字前缀自动排序、自动生成 sidebar）。
- 因子构造参数（`viewmodel._factor_params` 与 `select.screen._factor_params` 一致）：
  - `momentum` → 用 `lookback`(默认252) / `skip`(默认21)
  - 其余因子（`ma_bias`/`amihud`/`volatility`/`rel_volume`/`short_reversal`）→ 用 `window`(默认200/20)
- `viewmodel` 现有函数签名（不改）：
  - `available_factors() -> list[str]`
  - `workshop(name, lookback, skip, window, horizon, quantiles, mode, neutralize) -> str` (markdown)
  - `combine(names, weighting, quantiles, mode) -> dict`，返回 `{"weights": dict, "warnings": list, "metrics": {"annual_return","sharpe","max_drawdown","monthly_win_rate"}, "nav": Series}`
  - `holdout_run(name) -> str` (markdown)，已消耗时抛 `RuntimeError`
  - `history(Path) -> list[dict]`
  - `selector(names, weights, top_n, neutralize, mode) -> dict`，返回 `{"as_of": str, "weights": dict, "table": DataFrame}`，table 列 `instrument_id/score/rank/zone/sector`
  - `leaderboard(mode) -> DataFrame`
- 启动命令不变：`uv run streamlit run quant/web/app.py`。streamlit 自动发现同目录 `pages/`。
- `docs/superpowers/` 在 `.gitignore` 中，但既有 spec/plan 均以 `git add -f` 强制跟踪。本计划文件同样需 `-f`（已由写计划步骤处理）。

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `quant/web/_shared.py` | 创建 | `GLOSSARY` 术语表、`mode_selector()`、`step_header()` |
| `quant/web/app.py` | 重写 | 首页「🏠 总览」：简介 + 漏斗图 + 6 张入口卡 |
| `quant/web/pages/1_①_因子扫描.py` | 创建 | 原批量排行榜分支 + 步骤骨架 |
| `quant/web/pages/2_②_因子工坊.py` | 创建 | 原因子工坊分支 + 参数显隐 + help |
| `quant/web/pages/3_③_多因子合成.py` | 创建 | 原合成分支，去 JSON dump 改 metric 卡 |
| `quant/web/pages/4_④_holdout闸门.py` | 创建 | 原 holdout 分支 + 红色风险条 |
| `quant/web/pages/5_⑤_选股器.py` | 创建 | 原选股器分支 + 顶部摘要 |
| `quant/web/pages/6_📒_台账.py` | 创建 | 原历史分支（不编号） |
| `tests/test_web_shared.py` | 创建 | `_shared.GLOSSARY` 纯数据断言（唯一新增测试） |

> 注：旧 `quant/web/app.py` 的 `if/elif` 分支被拆分到 `pages/`，旧 `__pycache__` 无需处理（git 已忽略）。

---

## Task 1: 创建共用模块 `_shared.py`

**Files:**
- Create: `quant/web/_shared.py`
- Test: `tests/test_web_shared.py`

`_shared.py` 含三部分：`GLOSSARY`（纯 dict，可测）、`mode_selector()`、`step_header()`（调 streamlit，不单测）。

- [ ] **Step 1: 写失败测试（GLOSSARY 覆盖关键术语）**

`tests/test_web_shared.py`:

```python
from quant.web._shared import GLOSSARY


def test_glossary_covers_core_terms():
    required = {
        "lookback", "skip", "window", "horizon",
        "IC", "ICIR", "分位档数", "共线性", "DSR", "行业中性化",
    }
    assert required <= set(GLOSSARY)


def test_glossary_values_are_nonempty_strings():
    assert all(isinstance(v, str) and v.strip() for v in GLOSSARY.values())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_web_shared.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'quant.web._shared'`

- [ ] **Step 3: 写 `_shared.py`**

`quant/web/_shared.py`:

```python
"""Web 多页共用组件：术语表、数据模式选择器、步骤导航头部。

只渲染 UI，不含计算逻辑。被 app.py 与 pages/ 下各页引用。
"""

import streamlit as st

# 术语 → 白话，给各页 number_input/checkbox 的 help= 用
GLOSSARY = {
    "lookback": "回看多少天算动量，常用 252（约 1 年）",
    "skip": "跳过最近几天，避开短期反转，常用 21",
    "window": "计算窗口天数，常用 200",
    "horizon": "预测未来多少天的收益，常用 21",
    "quantiles": "按因子值把股票分几组看单调性，常用 5",
    "IC": "因子值与未来收益的相关性，越高预测力越强",
    "ICIR": "IC 均值 / IC 波动，>0.5 算不错",
    "分位档数": "按因子值把股票分几组看单调性，常用 5",
    "共线性": "因子间太像（相关>0.7）会重复下注",
    "DSR": "抗过拟合指标，扣掉多次试验的运气成分",
    "行业中性化": "去掉行业影响，只看因子本身",
}

_MODE_HELP = (
    "research：留出近 2 年数据不看，防偷看（研究阶段用）。\n\n"
    "full：用全部数据，仅最终选股时用。"
)


def mode_selector() -> str:
    """在 sidebar 渲染数据模式选择器，返回选中的 mode。"""
    return st.sidebar.selectbox(
        "数据模式", ["research", "full"], index=0, help=_MODE_HELP
    )


def step_header(step_no: int, title: str, what: str,
                prev: tuple[str, str] | None,
                next: tuple[str, str] | None) -> None:
    """渲染步骤页统一头部：第 N 步 / 5 + 上下步链接 + 这步干啥。

    prev/next 为 (page_path, label) 或 None。page_path 相对入口文件目录（quant/web/）。
    """
    st.caption(f"第 {step_no} 步 / 5 · {title}")
    c1, c2 = st.columns(2)
    if prev:
        c1.page_link(prev[0], label=f"◀ {prev[1]}")
    if next:
        c2.page_link(next[0], label=f"{next[1]} ▶")
    st.info(f"**这步干啥**：{what}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_web_shared.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add -f quant/web/_shared.py tests/test_web_shared.py
git commit -m "feat(web): shared module — glossary, mode selector, step header

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 重写 `app.py` 为首页总览

**Files:**
- Modify(重写): `quant/web/app.py`

首页纯展示：简介 + 漏斗图 + 6 张入口卡。无计算调用，故无单元测试，靠手动目测 + import 不报错验证。

> 注意 `st.page_link` 的 page 路径相对于主入口文件所在目录，写 `pages/<文件名>.py`。文件名含中文与圆圈数字，需与 Task 3-8 创建的文件名**逐字一致**。

- [ ] **Step 1: 重写 `app.py`**

`quant/web/app.py`（整文件替换）:

```python
"""Quant 因子研究 Web —— 首页总览。

启动：uv run streamlit run quant/web/app.py
streamlit 自动发现同目录 pages/ 下的步骤页并生成 sidebar。
"""

import streamlit as st

st.set_page_config(page_title="Quant 因子研究", layout="wide")

st.title("🏠 Quant 因子研究 · 总览")
st.write(
    "多因子量化选股研究平台。下面是一条完整研究流水线，像漏斗一样："
    "想法多 → 层层筛 → 活下来的少数 → 组合产出。"
)

st.markdown(
    """
```
想法多 ┌─────────────────────────────────────────────┐ 信号少
       │ ①扫描  →  ②工坊  →  ③合成  →  ④闸门  → ⑤选股│
       │  粗筛     精测      增强      终验锁定   产出 │
       └─────────────────────────────────────────────┘
                 📒 台账：全程自动记录
```
"""
)

st.divider()

_CARDS = [
    ("pages/1_①_因子扫描.py", "① 因子扫描",
     "一键体检全库因子，按 ICIR 排序，找出哪些值得深挖。", "开局粗筛", "进入 ①"),
    ("pages/2_②_因子工坊.py", "② 因子工坊",
     "单因子深度体检：预测力 / 单调性 / 多空三关红绿灯。", "盯一个因子", "进入 ②"),
    ("pages/3_③_多因子合成.py", "③ 多因子合成",
     "几个因子加权合成，查共线性，回测。", "组合增强", "进入 ③"),
    ("pages/4_④_holdout闸门.py", "④ holdout 闸门",
     "⚠ 定稿因子最终验证，仅一次，跑后锁定。", "即将上线", "进入 ④"),
    ("pages/5_⑤_选股器.py", "⑤ 选股器",
     "多因子加权打分，产出今日买入池。", "要选股清单", "进入 ⑤"),
    ("pages/6_📒_台账.py", "📒 台账",
     "看历史每次试验记录。", "回溯", "查看 📒"),
]

cols = st.columns(2)
for i, (path, title, what, when, btn) in enumerate(_CARDS):
    with cols[i % 2].container(border=True):
        st.subheader(title)
        st.write(what)
        st.caption(f"何时用：{when}")
        st.page_link(path, label=btn)
```

- [ ] **Step 2: 验证 import 不报错**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/app.py').read()); print('ok')"`
Expected: 输出 `ok`（语法正确；page_link 目标页将在 Task 3-8 创建，启动目测留到 Task 8 末）

- [ ] **Step 3: 提交**

```bash
git add -f quant/web/app.py
git commit -m "feat(web): home overview page with funnel diagram and entry cards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 步骤页 ① 因子扫描

**Files:**
- Create: `quant/web/pages/1_①_因子扫描.py`

源自旧 `app.py` 批量排行榜分支。第 1 步，无上一步，下一步→工坊。

- [ ] **Step 1: 创建页面**

`quant/web/pages/1_①_因子扫描.py`:

```python
"""第 1 步 · 因子扫描（批量排行榜）。"""

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import step_header

st.set_page_config(page_title="① 因子扫描", layout="wide")

step_header(
    1, "因子扫描",
    "一键体检全库因子，按 ICIR 排序，先看哪些有预测力、值得深挖。",
    prev=None, next=("pages/2_②_因子工坊.py", "② 因子工坊"),
)
from quant.web._shared import mode_selector
mode = mode_selector()

st.caption("批量扫描写独立 scan 台账，不影响 DSR 主台账。")
if st.button("批量体检全部因子"):
    df = viewmodel.leaderboard(mode=mode)
    st.dataframe(df)
    st.success("ICIR 高的几个，去 ②因子工坊 深挖。")
```

- [ ] **Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/pages/1_①_因子扫描.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: 提交**

```bash
git add -f "quant/web/pages/1_①_因子扫描.py"
git commit -m "feat(web): step 1 page — factor scan leaderboard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 步骤页 ② 因子工坊（参数显隐 + help）

**Files:**
- Create: `quant/web/pages/2_②_因子工坊.py`

源自旧工坊分支。**实质改动**：参数随因子显隐（momentum 显 lookback/skip，其余显 window），全部参数加 `help`。

- [ ] **Step 1: 创建页面**

`quant/web/pages/2_②_因子工坊.py`:

```python
"""第 2 步 · 因子工坊（单因子端到端体检）。"""

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import GLOSSARY, mode_selector, step_header

st.set_page_config(page_title="② 因子工坊", layout="wide")

step_header(
    2, "因子工坊",
    "对单个因子做 IC + 分位 + 多空体检，看它有没有预测力。"
    "跑完看报告卡顶部红绿灯三关是否通过。",
    prev=("pages/1_①_因子扫描.py", "① 因子扫描"),
    next=("pages/3_③_多因子合成.py", "③ 多因子合成"),
)
mode = mode_selector()

name = st.selectbox("因子", viewmodel.available_factors())

# 参数随因子显隐：momentum 用 lookback/skip，其余用 window
lookback = skip = window = None
c1, c2, c3 = st.columns(3)
if name == "momentum":
    lookback = c1.number_input("lookback", 1, 1000, 252, help=GLOSSARY["lookback"])
    skip = c2.number_input("skip", 0, 250, 21, help=GLOSSARY["skip"])
else:
    window = c1.number_input("window", 1, 1000, 200, help=GLOSSARY["window"])

horizon = c2.number_input("horizon", 1, 120, 21, help=GLOSSARY["horizon"])
quantiles = c3.number_input("分位档数", 2, 10, 5, help=GLOSSARY["分位档数"])
neutralize = c3.checkbox("行业中性化", help=GLOSSARY["行业中性化"])

if st.button("跑检验"):
    md = viewmodel.workshop(
        name,
        lookback=lookback if lookback is not None else 252,
        skip=skip if skip is not None else 21,
        window=window if window is not None else 200,
        horizon=horizon, quantiles=quantiles, mode=mode, neutralize=neutralize,
    )
    st.markdown(md)
```

> 说明：`workshop` 签名要求 lookback/skip/window 三参数都传；非当前因子的参数传默认值即可（viewmodel 内部按因子名只取需要的）。

- [ ] **Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/pages/2_②_因子工坊.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: 提交**

```bash
git add -f "quant/web/pages/2_②_因子工坊.py"
git commit -m "feat(web): step 2 page — factor workshop with param show/hide and help

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 步骤页 ③ 多因子合成（去 JSON dump → metric 卡）

**Files:**
- Create: `quant/web/pages/3_③_多因子合成.py`

源自旧合成分支。**实质改动**：`st.json(weights)`→表格；`st.json(metrics)`→4 个 `st.metric`；顶部加通俗结论。

- [ ] **Step 1: 创建页面**

`quant/web/pages/3_③_多因子合成.py`:

```python
"""第 3 步 · 多因子合成（加权合成 + 共线性预警 + 回测）。"""

import pandas as pd
import streamlit as st

from quant.web import viewmodel
from quant.web._shared import GLOSSARY, mode_selector, step_header

st.set_page_config(page_title="③ 多因子合成", layout="wide")

step_header(
    3, "多因子合成",
    "把几个因子加权合成一个综合分，查因子间共线性，再回测组合表现。",
    prev=("pages/2_②_因子工坊.py", "② 因子工坊"),
    next=("pages/4_④_holdout闸门.py", "④ holdout 闸门"),
)
mode = mode_selector()

names = st.multiselect("因子（多选）", viewmodel.available_factors(),
                       default=["momentum", "ma_bias"])
weighting = st.radio("加权法", ["equal", "ic"], horizontal=True)
quantiles = st.number_input("分位档数", 2, 10, 5, help=GLOSSARY["分位档数"])

if st.button("合成回测") and names:
    out = viewmodel.combine(names, weighting=weighting, quantiles=quantiles, mode=mode)

    # 通俗结论
    warns = out["warnings"]
    sharpe = out["metrics"]["sharpe"]
    col_msg = "因子间无明显共线性" if not warns else f"⚠ 共线性预警 {len(warns)} 处（相关≥0.7，可能重复下注）"
    sharpe_msg = "Sharpe 达标（≥1）" if sharpe >= 1 else "Sharpe 偏低（<1）"
    st.info(f"**通俗结论**：{col_msg}；{sharpe_msg}。")

    st.subheader("权重")
    st.dataframe(pd.DataFrame(
        {"因子": list(out["weights"]), "权重": list(out["weights"].values())}
    ))

    st.subheader("共线性预警（|相关|≥0.7）")
    st.write(warns or "无")

    st.subheader("绩效")
    m = out["metrics"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("年化收益", f"{m['annual_return']:.1%}")
    g2.metric("Sharpe", f"{m['sharpe']:.2f}")
    g3.metric("最大回撤", f"{m['max_drawdown']:.1%}")
    g4.metric("月胜率", f"{m['monthly_win_rate']:.1%}")

    st.line_chart(out["nav"])
```

- [ ] **Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/pages/3_③_多因子合成.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: 提交**

```bash
git add -f "quant/web/pages/3_③_多因子合成.py"
git commit -m "feat(web): step 3 page — combine with metric cards, drop raw JSON dump

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 步骤页 ④ holdout 闸门（红色风险条）

**Files:**
- Create: `quant/web/pages/4_④_holdout闸门.py`

源自旧 holdout 分支。**实质改动**：顶部加红色 `st.error` 风险条强化破坏性提示；现有二次 checkbox 保留。

- [ ] **Step 1: 创建页面**

`quant/web/pages/4_④_holdout闸门.py`:

```python
"""第 4 步 · holdout 闸门（定稿因子最终验证，仅一次，跑后锁定）。"""

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import mode_selector, step_header

st.set_page_config(page_title="④ holdout 闸门", layout="wide")

step_header(
    4, "holdout 闸门",
    "对定稿因子在留出的 holdout 数据上做最终验证。这是不可逆操作。",
    prev=("pages/3_③_多因子合成.py", "③ 多因子合成"),
    next=("pages/5_⑤_选股器.py", "⑤ 选股器"),
)
mode_selector()  # 渲染以保持各页一致；holdout 固定用 holdout 数据，mode 不参与

st.error("⚠ 破坏性操作：每个因子仅可在 holdout 跑一次，跑后永久锁定，不可重来。")
st.warning("务必确认这是最终定稿因子，再继续。")

name = st.selectbox("定稿因子", viewmodel.available_factors())
confirm = st.checkbox("我确认这是最终定稿，仅跑一次")
if st.button("在 holdout 跑最终验证") and confirm:
    try:
        md = viewmodel.holdout_run(name)
        st.markdown(md)
        st.success("holdout 已消耗，因子已锁定。")
    except RuntimeError as e:
        st.error(str(e))
```

- [ ] **Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/pages/4_④_holdout闸门.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: 提交**

```bash
git add -f "quant/web/pages/4_④_holdout闸门.py"
git commit -m "feat(web): step 4 page — holdout gate with red risk banner

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 步骤页 ⑤ 选股器（顶部摘要）

**Files:**
- Create: `quant/web/pages/5_⑤_选股器.py`

源自旧选股器分支。第 5 步，上一步→闸门，无下一步（指回台账）。**改动**：顶部补「今日买入池 N 只，截面日」摘要，归一权重从 caption 提为小表。

- [ ] **Step 1: 创建页面**

`quant/web/pages/5_⑤_选股器.py`:

```python
"""第 5 步 · 选股器（多因子加权打分 → 今日买入池）。"""

import altair as alt
import pandas as pd
import streamlit as st

from quant.web import viewmodel
from quant.web._shared import mode_selector, step_header

st.set_page_config(page_title="⑤ 选股器", layout="wide")

step_header(
    5, "选股器",
    "用多因子加权对最新截面打分，产出今日买入池。跑完可去 📒台账 回溯。",
    prev=("pages/4_④_holdout闸门.py", "④ holdout 闸门"),
    next=("pages/6_📒_台账.py", "📒 台账"),
)
mode = mode_selector()

names = st.multiselect("因子", viewmodel.available_factors(), default=["momentum"])
weights = {}
if names:
    cols = st.columns(len(names))
    for i, nm in enumerate(names):
        weights[nm] = cols[i].slider(f"{nm} 权重", 0, 100, 100 // len(names))
    st.caption(f"总权重 {sum(weights.values())}（提交时自动归一为 100%）")
top_n = st.number_input("买入池数量 top-N", 1, 100, 3)
neutralize = st.checkbox("行业中性化")

if st.button("选股") and names and sum(weights.values()) > 0:
    out = viewmodel.selector(names, weights, top_n=int(top_n),
                             neutralize=neutralize, mode=mode)
    t = out["table"]
    n_buy = int((t["zone"] == "buy").sum())
    st.success(f"今日买入池 {n_buy} 只 · 截面日 {out['as_of']}")

    st.subheader("归一权重")
    st.dataframe(pd.DataFrame(
        {"因子": list(out["weights"]), "权重": list(out["weights"].values())}
    ))

    chart = alt.Chart(t).mark_bar().encode(
        x=alt.X("score:Q", title="综合得分 (0-100)"),
        y=alt.Y("instrument_id:N", sort="-x", title=None),
        color=alt.Color(
            "zone:N",
            scale=alt.Scale(domain=["buy", "candidate"], range=["#2e8b2e", "#3b6fe0"]),
            legend=alt.Legend(title="池"),
        ),
        tooltip=["instrument_id", "score", "rank", "sector", "zone"],
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(t)
```

- [ ] **Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/pages/5_⑤_选股器.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: 提交**

```bash
git add -f "quant/web/pages/5_⑤_选股器.py"
git commit -m "feat(web): step 5 page — selector with buy-pool summary

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: 步骤页 📒 台账（不编号）+ 整体启动验收

**Files:**
- Create: `quant/web/pages/6_📒_台账.py`

源自旧历史分支。不属漏斗步骤，头部简化（仅标题 + 一句说明，无编号/无上下步）。

- [ ] **Step 1: 创建页面**

`quant/web/pages/6_📒_台账.py`:

```python
"""台账 · 全程试验记录（不属漏斗某一步）。"""

from pathlib import Path

import streamlit as st

from quant.web import viewmodel
from quant.web._shared import mode_selector

st.set_page_config(page_title="📒 台账", layout="wide")

st.title("📒 台账")
st.caption("全程试验记录：工坊体检、回测、holdout 都会写在这里。")
mode_selector()

rows = viewmodel.history(Path("quant_out/ledger.jsonl"))
if rows:
    st.dataframe(rows)
else:
    st.info("暂无记录。先在 ②因子工坊 或 ④holdout闸门 里跑一次。")
```

- [ ] **Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('quant/web/pages/6_📒_台账.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: 跑全量 pytest 确认零回归**

Run: `uv run pytest -q`
Expected: 全绿（原 128 + 新增 2 = 130 passed），无 failure

- [ ] **Step 4: 启动 Web 目测（手动验收）**

Run（后台启动）:
```bash
uv run streamlit run quant/web/app.py --server.headless true > /tmp/st.log 2>&1 &
```
然后打开浏览器 `http://localhost:8501`，逐项核对：
1. sidebar 顺序为 总览 / ①因子扫描 / ②因子工坊 / ③多因子合成 / ④holdout闸门 / ⑤选股器 / 📒台账
2. 首页显示漏斗图 + 6 张卡，点卡片按钮能跳到对应页
3. 每个步骤页顶部显示「第 N 步 / 5 + 上下步链接 + 这步干啥」蓝框
4. ②工坊：选 momentum 显示 lookback/skip，选其他因子显示 window；参数 hover 出现 help 白话
5. ③合成：跑后显示 4 个 metric 卡 + 通俗结论，无裸 JSON
6. ④闸门：顶部红色风险条

核对完停止：`kill %1`（或 `kill $!`）

Expected: 6 项全部符合

- [ ] **Step 5: 提交**

```bash
git add -f "quant/web/pages/6_📒_台账.py"
git commit -m "feat(web): ledger page (unnumbered) and finalize multipage redesign

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review（已执行）

**1. Spec 覆盖**：

| Spec 要求 | 对应任务 |
|---|---|
| §4 `_shared.py`（导航/help/mode） | Task 1 |
| §5.1 首页漏斗图 + 6 卡 | Task 2 |
| §5.2 step_header / mode_selector / GLOSSARY | Task 1 |
| §5.3 ① 因子扫描 + 结论引导 | Task 3 |
| §5.3 ② 参数显隐 + help | Task 4 |
| §5.3 ③ 去 JSON dump → metric 卡 + 通俗结论 | Task 5 |
| §5.3 ④ 红色风险条 | Task 6 |
| §5.3 ⑤ 顶部摘要 + 权重小表 | Task 7 |
| §5.3 📒 台账简化头部 | Task 8 |
| §7 现有 128 用例不回归 | Task 8 Step 3 |
| §7 手动验收 6 项 | Task 8 Step 4 |

无遗漏。

**2. 占位扫描**：无 TBD/TODO；每个代码步骤含完整可运行代码。

**3. 类型一致**：`step_header(step_no, title, what, prev, next)`、`mode_selector()`、`GLOSSARY` 在 Task 1 定义，Task 2-8 引用签名一致。page_link 路径字符串与各页文件名逐字对应（`pages/1_①_因子扫描.py` 等，相对于入口文件目录 quant/web/）。`viewmodel.combine` 返回的 `metrics` 键名（annual_return/sharpe/max_drawdown/monthly_win_rate）与 Task 5 使用一致。`selector` 返回 table 的 `zone` 列与 Task 7 `(t["zone"]=="buy")` 一致。
