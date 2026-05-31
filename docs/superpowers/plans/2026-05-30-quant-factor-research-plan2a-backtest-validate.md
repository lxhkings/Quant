# Quant 多因子合成 + 真实回测 + 抗过拟合 Implementation Plan (Plan 2a/2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Plan 1（数据/因子/单因子检验/报告卡）之上，建多因子合成、含成本真实回测引擎、抗过拟合三件套（walk-forward / Deflated Sharpe / 试验台账）与 holdout 闸门，并用 CLI 串通。

**Architecture:** 横截面引擎，全程 pandas 宽矩阵 `[date × instrument_id]`。回测从因子矩阵分位选股、按调仓频率构权重、用日度收益滚动净值、按换手扣成本。抗过拟合层读回测产出的每期 Sharpe，配合试验台账（`ledger.jsonl`）算 Deflated Sharpe；walk-forward 滚动切窗在样本内选参、样本外验证。holdout 闸门用持久化状态文件强制"仅一次"。依赖方向 `cli → {report, validate, combine, backtest} → {eval, process, factor, data}`，`validate → backtest`，无环。

**Tech Stack:** Python 3.11、pandas 2.2、numpy、scipy（`scipy.stats.norm/skew/kurtosis`）、typer、pytest。沿用 Plan 1 既有依赖，无新增第三方库。

**Plan 2b（不在本计划）：** Streamlit web 四页、其余因子族、行业中性化、缓存层、因子正交化。本计划只产研究闭环（合成 + 回测 + 抗过拟合 + holdout 闸门）。

---

## 前置依赖（Plan 1 既有接口，本计划直接调用，勿改动）

| 接口 | 文件 | 签名 |
|---|---|---|
| 价格面板 | `quant/data/panel.py` | `load_price_matrix(field="close", market="us", root=None) -> DataFrame` |
| holdout 截断 | `quant/data/holdout.py` | `apply_holdout(matrix, mode="research", holdout_years=2) -> DataFrame` |
| 前瞻收益 | `quant/data/returns.py` | `forward_returns(close, horizon=1) -> DataFrame` |
| 因子 | `quant/factor/library/{momentum,ma_bias}.py` | `Momentum(lookback,skip).compute(close)` / `MABias(window).compute(close)` |
| 处理链 | `quant/process/pipeline.py` | `winsorize(df)`, `zscore(df)`, `Pipeline([...])(df)` |
| IC | `quant/eval/ic.py` | `ic_series(factor, fwd_ret, method="spearman") -> Series`, `ic_summary(ic) -> dict` |
| 合成 fixture | `tests/conftest.py` | `fake_lake` → `(root, instruments[3], days[60])` |

---

## File Structure

| 文件 | 职责 |
|---|---|
| `quant/data/returns.py`（修改） | 追加 `simple_returns()` 日度简单收益 |
| `quant/backtest/__init__.py` | 包标记 |
| `quant/backtest/metrics.py` | `annualized_return/sharpe/max_drawdown/calmar/monthly_win_rate` + `BacktestMetrics` + `compute_metrics()` |
| `quant/backtest/engine.py` | `rebalance_dates()` / `target_weights()` / `backtest()` + `BacktestResult` |
| `quant/combine/__init__.py` | 包标记 |
| `quant/combine/synth.py` | `zscore_factors/equal_weight/ic_weight/combine_score/factor_correlation/high_correlation_warnings` |
| `quant/validate/__init__.py` | 包标记 |
| `quant/validate/ledger.py` | `Ledger` 试验台账（jsonl 追加 + 计数 + 取 Sharpe 列表） |
| `quant/validate/dsr.py` | `expected_max_sharpe()` / `deflated_sharpe()` |
| `quant/validate/walkforward.py` | `walk_forward()` / `oos_is_ratio()` |
| `quant/validate/gate.py` | `is_consumed/mark_consumed/assert_not_consumed` holdout 闸门 |
| `quant/report/backtest_card.py` | `BacktestReport` dataclass + `to_markdown()`（红绿灯） |
| `quant/cli.py`（修改） | 追加 `quant backtest` / `quant combine` / `quant holdout` 命令 |

每文件单一职责；回测拆"组合构建/模拟"（engine）与"绩效统计"（metrics）两文件，边界清晰。

---

## Task 1: 日度简单收益

回测净值需要日度简单收益（区别于 Plan 1 的前瞻收益）。在既有 `returns.py` 追加。

**Files:**
- Modify: `quant/data/returns.py`
- Create: `tests/test_simple_returns.py`

- [ ] **Step 1: 写失败测试**

`tests/test_simple_returns.py`:
```python
import numpy as np
import pandas as pd

from quant.data.returns import simple_returns


def test_simple_returns_value():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]}, index=idx)
    r = simple_returns(close)
    assert np.isnan(r["AAA"].iloc[0])          # 首日无前值
    assert np.isclose(r["AAA"].iloc[1], 0.10)  # 110/100-1
    assert np.isclose(r["AAA"].iloc[2], -0.10) # 99/110-1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_simple_returns.py -v`
Expected: FAIL（ImportError: cannot import name 'simple_returns'）

- [ ] **Step 3: 实现 simple_returns**

在 `quant/data/returns.py` 末尾追加：
```python
def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """日度简单收益 = close[t] / close[t-1] - 1。首行为 NaN（无前值）。"""
    return close.pct_change()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_simple_returns.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/data/returns.py tests/test_simple_returns.py
git commit -m "feat: daily simple returns"
```

---

## Task 2: 回测绩效指标

净值/收益 → 年化、Sharpe、最大回撤、Calmar、月胜率。纯函数 + 汇总 dataclass。

**Files:**
- Create: `quant/backtest/__init__.py`
- Create: `quant/backtest/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: 写失败测试**

`tests/test_metrics.py`:
```python
import numpy as np
import pandas as pd

from quant.backtest.metrics import (
    BacktestMetrics,
    annualized_return,
    calmar,
    compute_metrics,
    max_drawdown,
    monthly_win_rate,
    sharpe,
)


def test_annualized_return_doubling_in_one_year():
    nav = pd.Series([1.0, 2.0])
    # 2 个点、每年 2 期 → 跨 1 年 → 翻倍 = 100%
    assert np.isclose(annualized_return(nav, periods_per_year=2), 1.0)


def test_max_drawdown_known_path():
    nav = pd.Series([1.0, 1.2, 0.9, 1.0])
    # 峰值 1.2 → 谷 0.9 → 回撤 -0.25
    assert np.isclose(max_drawdown(nav), -0.25)


def test_sharpe_nan_when_flat():
    r = pd.Series([0.01, 0.01, 0.01])  # 标准差 0
    assert np.isnan(sharpe(r))


def test_sharpe_positive_when_mostly_up():
    r = pd.Series([0.02, -0.01, 0.03, 0.01])
    assert sharpe(r, periods_per_year=1) > 0


def test_calmar():
    assert np.isclose(calmar(0.2, -0.1), 2.0)
    assert np.isnan(calmar(0.2, 0.0))


def test_monthly_win_rate_half():
    # 1 月全涨、2 月全跌 → 月胜率 0.5
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    vals = [0.01 if d.month == 1 else -0.01 for d in idx]
    r = pd.Series(vals, index=idx)
    assert np.isclose(monthly_win_rate(r), 0.5)


def test_compute_metrics_bundle():
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.01] * 10, index=idx)
    nav = (1 + r).cumprod()
    m = compute_metrics(nav, r)
    assert isinstance(m, BacktestMetrics)
    assert m.annual_return > 0
    assert m.max_drawdown <= 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL（ModuleNotFoundError: quant.backtest）

- [ ] **Step 3: 实现 metrics**

`quant/backtest/__init__.py`:
```python
```

`quant/backtest/metrics.py`:
```python
"""回测绩效指标：年化、Sharpe、最大回撤、Calmar、月胜率。

net 收益序列驱动一切。Sharpe 默认按 periods_per_year 年化（×sqrt(N)）；
传 periods_per_year=1 得每期（非年化）Sharpe，供 Deflated Sharpe 使用。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(nav: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """复合年化收益 = (末值/首值)^(年频/期数) - 1。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return float("nan")
    total = nav.iloc[-1] / nav.iloc[0]
    years = len(nav) / periods_per_year
    return total ** (1 / years) - 1


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sharpe = 均值/标准差 × sqrt(年频)。标准差为 0 或样本不足返回 NaN。"""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return r.mean() / sd * np.sqrt(periods_per_year)


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤 = min(nav/历史峰值 - 1)，为非正数。"""
    nav = nav.dropna()
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def calmar(annual: float, mdd: float) -> float:
    """Calmar = 年化收益 / |最大回撤|。回撤为 0 或 NaN 时返回 NaN。"""
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annual / abs(mdd)


def monthly_win_rate(returns: pd.Series) -> float:
    """月度收益为正的比例。"""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    monthly = (1 + r).resample("ME").prod() - 1
    if monthly.empty:
        return float("nan")
    return float((monthly > 0).mean())


@dataclass
class BacktestMetrics:
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    monthly_win_rate: float


def compute_metrics(
    nav: pd.Series, returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> BacktestMetrics:
    """从净值 + net 收益序列汇出全套指标。"""
    ann = annualized_return(nav, periods_per_year)
    mdd = max_drawdown(nav)
    return BacktestMetrics(
        annual_return=ann,
        sharpe=sharpe(returns, periods_per_year),
        max_drawdown=mdd,
        calmar=calmar(ann, mdd),
        monthly_win_rate=monthly_win_rate(returns),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/__init__.py quant/backtest/metrics.py tests/test_metrics.py
git commit -m "feat: backtest performance metrics"
```

---

## Task 3: 回测组合构建（调仓日 + 目标权重）

调仓日 = 每个周期首个交易日；目标权重 = 顶档等权（多头）或顶档多 + 底档空（多空）。

**Files:**
- Create: `quant/backtest/engine.py`
- Create: `tests/test_portfolio.py`

- [ ] **Step 1: 写失败测试**

`tests/test_portfolio.py`:
```python
import numpy as np
import pandas as pd

from quant.backtest.engine import rebalance_dates, target_weights


def test_rebalance_dates_monthly_first_trading_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    reb = rebalance_dates(idx, freq="M")
    # 3 个月 → 3 个调仓日，各为当月首个交易日
    assert len(reb) == 3
    assert reb[0] == idx[idx.month == 1][0]
    assert reb[1] == idx[idx.month == 2][0]


def test_target_weights_long_top_quantile_equal():
    row = pd.Series({c: v for c, v in zip("ABCDEFGHIJ", range(1, 11))}, dtype=float)
    w = target_weights(row, n=5, side="long")
    # 10 标的 5 档 → 每档 2 个；顶档 = 两个最高因子 {I,J}，各 0.5
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["J"], 0.5)
    assert np.isclose(w["I"], 0.5)
    assert np.isclose(w["A"], 0.0)


def test_target_weights_long_short_nets_zero():
    row = pd.Series({c: v for c, v in zip("ABCDEFGHIJ", range(1, 11))}, dtype=float)
    w = target_weights(row, n=5, side="long_short")
    # 顶档 +0.5/+0.5，底档 -0.5/-0.5 → 净敞口 0，毛敞口 2
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(w.abs().sum(), 2.0)
    assert w["J"] > 0 and w["A"] < 0


def test_target_weights_all_zero_when_insufficient():
    row = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan}, dtype=float)
    w = target_weights(row, n=5, side="long")
    # 有效标的 < 档数 → 不建仓
    assert np.isclose(w.abs().sum(), 0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_portfolio.py -v`
Expected: FAIL（ImportError: rebalance_dates）

- [ ] **Step 3: 实现 engine 的组合构建部分**

`quant/backtest/engine.py`:
```python
"""回测引擎：分位选股 → 构权重 → 滚动净值 → 扣成本。

约定（防前视）：
- 第 t 个调仓日用截至 t 的因子值构目标权重；
- 该权重经 shift(1) 应用到 t+1 起的日度收益；
- 换手成本在调仓日 t 当天扣减。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

_PERIOD_CODE = {"M": "M", "W": "W"}


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """每个周期的首个交易日作为调仓日。freq: 'M' 月度 / 'W' 周度。"""
    if freq not in _PERIOD_CODE:
        raise ValueError(f"未知 freq: {freq!r}")
    s = pd.Series(index, index=index)
    first = s.groupby(index.to_period(_PERIOD_CODE[freq])).first()
    return pd.DatetimeIndex(first.values)


def target_weights(factor_row: pd.Series, n: int = 5, side: str = "long") -> pd.Series:
    """单个截面的目标权重。

    side="long"：顶档（第 n 档）等权，权重和 1。
    side="long_short"：顶档等权做多、底档（第 1 档）等权做空，净 0、毛 2。
    有效标的不足 n 档返回全 0。
    """
    w = pd.Series(0.0, index=factor_row.index)
    valid = factor_row.dropna()
    if len(valid) < n:
        return w
    buckets = pd.qcut(valid.rank(method="first"), n, labels=False, duplicates="drop") + 1
    top = buckets.index[buckets == n]
    if len(top):
        w[top] = 1.0 / len(top)
    if side == "long_short":
        bottom = buckets.index[buckets == 1]
        if len(bottom):
            w[bottom] = -1.0 / len(bottom)
    elif side != "long":
        raise ValueError(f"未知 side: {side!r}")
    return w
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_portfolio.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/engine.py tests/test_portfolio.py
git commit -m "feat: backtest rebalance dates + target weights"
```

---

## Task 4: 回测模拟（净值 + 换手）

把目标权重沿时间持有，乘日度收益累净值，调仓日按换手扣成本。

**Files:**
- Modify: `quant/backtest/engine.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: 写失败测试**

`tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult, backtest


def _setup():
    """10 标的、5 个交易日，单月（单次调仓）。

    因子恒定排序 A<B<...<J；n=5 → 顶档 = {I,J} 各 0.5。
    收益：每标的每日固定，J=+2%、I=+1%、其余 0 → 顶档篮子日收益 1.5%。
    """
    idx = pd.bdate_range("2020-01-06", periods=5)  # 同一周/月内
    cols = list("ABCDEFGHIJ")
    factor = pd.DataFrame([list(range(1, 11))] * 5, index=idx, columns=cols, dtype=float)
    drift = {c: 0.0 for c in cols}
    drift["I"], drift["J"] = 0.01, 0.02
    close = pd.DataFrame(
        {c: [100.0 * (1 + drift[c]) ** i for i in range(5)] for c in cols}, index=idx
    )
    return factor, close


def test_zero_cost_nav_matches_basket():
    factor, close = _setup()
    res = backtest(factor, close, n=5, side="long", freq="M", cost_bps=0.0)
    assert isinstance(res, BacktestResult)
    # 顶档篮子（I,J 等权）日收益 = (1% + 2%)/2 = 1.5%；
    # 调仓日 idx[0] 建仓，权重 shift(1) 从 idx[1] 起生效 → 净值含 4 个 1.5% 复利
    expected = (1 + 0.015) ** 4
    assert np.isclose(res.nav.iloc[-1], expected, rtol=1e-6)


def test_cost_drags_nav_below_zero_cost():
    factor, close = _setup()
    free = backtest(factor, close, n=5, side="long", freq="M", cost_bps=0.0)
    paid = backtest(factor, close, n=5, side="long", freq="M", cost_bps=50.0)
    # 有成本净值 < 无成本净值
    assert paid.nav.iloc[-1] < free.nav.iloc[-1]


def test_turnover_full_on_first_rebalance():
    factor, close = _setup()
    res = backtest(factor, close, n=5, side="long", freq="M", cost_bps=0.0)
    # 首次建仓从空仓换到满仓 → 换手 = 1（顶档总权重 1 的一半 symmetric diff）
    reb_turnover = res.turnover[res.turnover > 0]
    assert np.isclose(reb_turnover.iloc[0], 1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL（ImportError: backtest）

- [ ] **Step 3: 在 engine.py 追加 backtest**

在 `quant/backtest/engine.py` 末尾追加：
```python
@dataclass
class BacktestResult:
    nav: pd.Series       # 累计净值（起点 1）
    returns: pd.Series   # 每期 net 收益（已扣成本）
    turnover: pd.Series  # 各日换手（仅调仓日非 0）


def backtest(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    n: int = 5,
    side: str = "long",
    freq: str = "M",
    cost_bps: float = 10.0,
) -> BacktestResult:
    """按调仓频率分位选股、等权持有、扣单边成本，返回净值序列。"""
    daily_ret = close.pct_change()
    reb = rebalance_dates(close.index, freq)

    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    turnover = pd.Series(0.0, index=close.index)
    prev_w = pd.Series(0.0, index=close.columns)
    for d in reb:
        if d not in factor.index:
            continue
        tw = target_weights(factor.loc[d], n=n, side=side).reindex(close.columns).fillna(0.0)
        weights.loc[d] = tw.values
        turnover.loc[d] = float((tw - prev_w).abs().sum() / 2)
        prev_w = tw

    # 调仓日之间持有不变；非调仓日 NaN → 前向填充
    weights = weights.ffill().fillna(0.0)
    # 防前视：t 日权重作用于 t+1 收益
    port_ret = (weights.shift(1) * daily_ret).sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = port_ret - cost.reindex(port_ret.index).fillna(0.0)
    nav = (1 + net).cumprod()
    return BacktestResult(nav=nav, returns=net, turnover=turnover)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/engine.py tests/test_backtest.py
git commit -m "feat: backtest simulation (nav + turnover + cost)"
```

---

## Task 5: 多因子合成

每因子先 zscore → 等权 / IC 加权 → 加权合成总分；并出共线性相关矩阵与高相关预警。

**Files:**
- Create: `quant/combine/__init__.py`
- Create: `quant/combine/synth.py`
- Create: `tests/test_combine.py`

- [ ] **Step 1: 写失败测试**

`tests/test_combine.py`:
```python
import numpy as np
import pandas as pd

from quant.combine.synth import (
    combine_score,
    equal_weight,
    factor_correlation,
    high_correlation_warnings,
    ic_weight,
    zscore_factors,
)


def _factors():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = list("ABCD")
    a = pd.DataFrame([[1, 2, 3, 4]] * 3, index=idx, columns=cols, dtype=float)
    b = pd.DataFrame([[4, 3, 2, 1]] * 3, index=idx, columns=cols, dtype=float)
    return {"a": a, "b": b}


def test_zscore_factors_row_mean_zero():
    z = zscore_factors(_factors())
    assert np.isclose(z["a"].iloc[0].mean(), 0.0)


def test_equal_weight():
    assert equal_weight(["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_ic_weight_proportional_to_ic():
    w = ic_weight({"a": 0.04, "b": 0.02})
    assert np.isclose(w["a"], 2 / 3)
    assert np.isclose(w["b"], 1 / 3)


def test_ic_weight_negative_falls_back_to_equal():
    w = ic_weight({"a": -0.01, "b": -0.02})
    assert np.isclose(w["a"], 0.5)


def test_combine_score_weighted_sum():
    f = _factors()
    z = zscore_factors(f)
    score = combine_score(z, {"a": 0.5, "b": 0.5})
    # a 与 b 互为反向 zscore → 等权合成在每个截面应抵消为约 0
    assert np.allclose(score.iloc[0].values, 0.0, atol=1e-9)


def test_correlation_and_warnings():
    z = zscore_factors(_factors())
    corr = factor_correlation(z)
    # a、b 完全反向 → 相关 -1
    assert np.isclose(corr.loc["a", "b"], -1.0)
    warns = high_correlation_warnings(corr, threshold=0.7)
    assert warns and warns[0][2] < 0  # 触发预警，相关为负
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_combine.py -v`
Expected: FAIL（ModuleNotFoundError: quant.combine）

- [ ] **Step 3: 实现 synth**

`quant/combine/__init__.py`:
```python
```

`quant/combine/synth.py`:
```python
"""多因子合成：逐因子 zscore → 加权 → 合成总分；含共线性相关矩阵与预警。

正交化（去共线）属 Plan 2b/ v2，本模块不实现。
"""

import pandas as pd

from quant.process.pipeline import zscore


def zscore_factors(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """逐因子做横截面标准化。"""
    return {name: zscore(f) for name, f in factors.items()}


def equal_weight(names: list[str]) -> dict[str, float]:
    """等权：每因子 1/N。"""
    w = 1.0 / len(names)
    return {n: w for n in names}


def ic_weight(ic_means: dict[str, float]) -> dict[str, float]:
    """IC 加权：权重 ∝ max(IC, 0)，归一。全为非正时退回等权。"""
    pos = {n: max(v, 0.0) for n, v in ic_means.items()}
    total = sum(pos.values())
    if total == 0:
        return equal_weight(list(ic_means))
    return {n: v / total for n, v in pos.items()}


def combine_score(
    factors: dict[str, pd.DataFrame], weights: dict[str, float]
) -> pd.DataFrame:
    """加权求和合成总分（因子应已标准化）。"""
    score = None
    for name, f in factors.items():
        term = f * weights[name]
        score = term if score is None else score.add(term, fill_value=0.0)
    return score


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """因子值两两相关矩阵（按对齐后的全体格点配对，逐对剔除 NaN）。"""
    names = list(factors)
    base = factors[names[0]]
    flat = {
        name: f.reindex(index=base.index, columns=base.columns).to_numpy().ravel()
        for name, f in factors.items()
    }
    return pd.DataFrame(flat).corr()


def high_correlation_warnings(
    corr: pd.DataFrame, threshold: float = 0.7
) -> list[tuple[str, str, float]]:
    """返回 |相关| ≥ 阈值的因子对列表。"""
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) >= threshold:
                out.append((cols[i], cols[j], float(v)))
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_combine.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/combine/ tests/test_combine.py
git commit -m "feat: multi-factor synthesis + collinearity"
```

---

## Task 6: 试验台账 Ledger

每次回测/扫参追加一行 JSON（factor/params/sharpe），供 DSR 用试验次数与 Sharpe 方差折减。

**Files:**
- Create: `quant/validate/__init__.py`
- Create: `quant/validate/ledger.py`
- Create: `tests/test_ledger.py`

- [ ] **Step 1: 写失败测试**

`tests/test_ledger.py`:
```python
from quant.validate.ledger import Ledger


def test_record_and_count(tmp_path):
    led = Ledger(tmp_path / "sub" / "ledger.jsonl")  # 自动建父目录
    led.record({"factor": "momentum", "sharpe": 1.2})
    led.record({"factor": "momentum", "sharpe": 0.8})
    assert led.count() == 2


def test_sharpes_skips_missing_and_nan(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.record({"factor": "a", "sharpe": 1.0})
    led.record({"factor": "b"})                     # 无 sharpe
    led.record({"factor": "c", "sharpe": float("nan")})
    assert led.sharpes() == [1.0]


def test_entries_empty_when_no_file(tmp_path):
    led = Ledger(tmp_path / "missing.jsonl")
    assert led.entries() == []
    assert led.count() == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ledger.py -v`
Expected: FAIL（ModuleNotFoundError: quant.validate）

- [ ] **Step 3: 实现 ledger**

`quant/validate/__init__.py`:
```python
```

`quant/validate/ledger.py`:
```python
"""试验台账：每次回测/扫参追加一行 JSON，记录 factor/params/sharpe。

纪律——扫了多少组参数必须记账，Deflated Sharpe 用该次数 N 折减。
"""

import json
import math
from pathlib import Path


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict) -> None:
        """追加一条试验记录。"""
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self) -> list[dict]:
        """读全部记录（文件不存在返回空）。"""
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def count(self) -> int:
        """试验次数 N。"""
        return len(self.entries())

    def sharpes(self) -> list[float]:
        """所有有效（非 NaN）Sharpe 值，供 DSR 估方差。"""
        return [
            e["sharpe"]
            for e in self.entries()
            if "sharpe" in e and not math.isnan(e["sharpe"])
        ]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_ledger.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/validate/__init__.py quant/validate/ledger.py tests/test_ledger.py
git commit -m "feat: trial ledger for multiple-testing accounting"
```

---

## Task 7: Deflated Sharpe Ratio

多次试验后 Sharpe 会虚高。DSR 按试验次数 N + 收益偏度/峰度折减，出真实显著性概率。

公式（Bailey & López de Prado）：
- 期望最大 Sharpe（零假设基准）：`E[max] = sqrt(V) · [(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))]`，γ 为 Euler-Mascheroni 常数 ≈0.5772，V 为各试验 Sharpe 的方差。
- `DSR = Φ( (SR - SR0)·sqrt(T-1) / sqrt(1 - skew·SR + (kurt-1)/4·SR²) )`，SR/SR0 为每期（非年化）Sharpe，T 为收益样本数，kurt 为非超额峰度（正态=3），Φ 为标准正态 CDF。

**Files:**
- Create: `quant/validate/dsr.py`
- Create: `tests/test_dsr.py`

- [ ] **Step 1: 写失败测试**

`tests/test_dsr.py`:
```python
import numpy as np

from quant.validate.dsr import deflated_sharpe, expected_max_sharpe


def test_expected_max_zero_for_single_trial():
    # N=1 无多重检验惩罚
    assert expected_max_sharpe(sr_variance=1.0, n_trials=1) == 0.0


def test_expected_max_grows_with_trials():
    e10 = expected_max_sharpe(sr_variance=1.0, n_trials=10)
    e100 = expected_max_sharpe(sr_variance=1.0, n_trials=100)
    assert 0 < e10 < e100


def test_dsr_high_when_sr_beats_benchmark():
    # 强 SR、零基准、长样本、正态 → DSR≈1
    assert deflated_sharpe(sr=0.2, sr0=0.0, n_obs=1000) > 0.99


def test_dsr_low_when_sr_below_benchmark():
    assert deflated_sharpe(sr=0.0, sr0=0.1, n_obs=1000) < 0.01


def test_dsr_half_at_benchmark():
    assert np.isclose(deflated_sharpe(sr=0.1, sr0=0.1, n_obs=500), 0.5)


def test_dsr_nan_when_too_few_obs():
    assert np.isnan(deflated_sharpe(sr=0.2, sr0=0.0, n_obs=1))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_dsr.py -v`
Expected: FAIL（ModuleNotFoundError: quant.validate.dsr）

- [ ] **Step 3: 实现 dsr**

`quant/validate/dsr.py`:
```python
"""Deflated Sharpe Ratio：多重检验折减后的真实显著性。"""

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """N 次独立试验下零假设的期望最大 Sharpe（多重检验基准 SR0）。

    sr_variance：各试验 Sharpe 的方差 V。N≤1 或 V≤0 时无惩罚，返回 0。
    """
    if n_trials <= 1 or sr_variance <= 0 or np.isnan(sr_variance):
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(
    sr: float, sr0: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """观测 Sharpe 扣减基准 SR0 后的真实显著性概率。

    sr/sr0：每期（非年化）Sharpe。n_obs：收益样本数 T。
    skew/kurt：收益的偏度与非超额峰度（正态=3）。返回 [0,1] 概率。
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr**2)
    stat = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(stat))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_dsr.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/validate/dsr.py tests/test_dsr.py
git commit -m "feat: deflated sharpe ratio"
```

---

## Task 8: Walk-forward

滚动切窗：样本内（IS）选最优参数，样本外（OOS）验证，看 OOS 衰减。OOS/IS Sharpe 比值 <0.5 预警过拟合。

**Files:**
- Create: `quant/validate/walkforward.py`
- Create: `tests/test_walkforward.py`

- [ ] **Step 1: 写失败测试**

`tests/test_walkforward.py`:
```python
import numpy as np
import pandas as pd

from quant.validate.walkforward import oos_is_ratio, walk_forward


def _world():
    """10 标的、120 交易日。标的按编号有固定日漂移：J 最强、A 最弱。

    构造两个候选因子：'good'=升序排名（追强者）、'bad'=降序（追弱者）。
    good 在 IS/OOS 都应正 Sharpe；bad 负。
    """
    idx = pd.bdate_range("2020-01-01", periods=120)
    cols = list("ABCDEFGHIJ")
    drift = {c: 0.0005 * i for i, c in enumerate(cols)}  # A=0 .. J=0.0045
    close = pd.DataFrame(
        {c: [100.0 * (1 + drift[c]) ** t for t in range(120)] for c in cols}, index=idx
    )
    asc = pd.DataFrame([list(range(1, 11))] * 120, index=idx, columns=cols, dtype=float)
    desc = pd.DataFrame([list(range(10, 0, -1))] * 120, index=idx, columns=cols, dtype=float)
    factors = {"good": asc, "bad": desc}
    return close, factors


def test_walk_forward_picks_good_param():
    close, factors = _world()
    wf = walk_forward(
        build=lambda p: factors[p],
        close=close,
        params=["good", "bad"],
        is_days=60,
        oos_days=30,
        n=5,
        side="long",
    )
    assert not wf.empty
    # 每个窗口 IS 都应选中 'good'
    assert (wf["best_param"] == "good").all()
    # OOS Sharpe 为正（信号持续）
    assert (wf["oos_sharpe"] > 0).all()


def test_oos_is_ratio_finite():
    close, factors = _world()
    wf = walk_forward(
        build=lambda p: factors[p],
        close=close,
        params=["good", "bad"],
        is_days=60,
        oos_days=30,
        n=5,
        side="long",
    )
    ratio = oos_is_ratio(wf)
    assert np.isfinite(ratio)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: FAIL（ModuleNotFoundError: quant.validate.walkforward）

- [ ] **Step 3: 实现 walkforward**

`quant/validate/walkforward.py`:
```python
"""Walk-forward：滚动窗口样本内选参、样本外验证。

参数只在 IS 调，OOS 仅验证，量化过拟合（OOS/IS Sharpe 比）。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from quant.backtest.engine import backtest
from quant.backtest.metrics import sharpe as _sharpe


def walk_forward(
    build: Callable[[object], pd.DataFrame],
    close: pd.DataFrame,
    params: list,
    is_days: int,
    oos_days: int,
    step: int | None = None,
    n: int = 5,
    side: str = "long",
    freq: str = "M",
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """滚动切窗。build(param) 返回因子矩阵。

    返回 DataFrame[is_start, oos_start, best_param, is_sharpe, oos_sharpe]。
    """
    index = close.index
    step = step or oos_days
    rows = []
    start = 0
    while start + is_days + oos_days <= len(index):
        is_slice = index[start : start + is_days]
        oos_slice = index[start + is_days : start + is_days + oos_days]

        best_param, best_is = None, -np.inf
        for p in params:
            f = build(p)
            res = backtest(
                f.loc[is_slice], close.loc[is_slice],
                n=n, side=side, freq=freq, cost_bps=cost_bps,
            )
            s = _sharpe(res.returns)
            if not np.isnan(s) and s > best_is:
                best_is, best_param = s, p

        if best_param is None:
            start += step
            continue

        f = build(best_param)
        oos_res = backtest(
            f.loc[oos_slice], close.loc[oos_slice],
            n=n, side=side, freq=freq, cost_bps=cost_bps,
        )
        rows.append({
            "is_start": is_slice[0],
            "oos_start": oos_slice[0],
            "best_param": best_param,
            "is_sharpe": best_is,
            "oos_sharpe": _sharpe(oos_res.returns),
        })
        start += step
    return pd.DataFrame(rows)


def oos_is_ratio(wf: pd.DataFrame) -> float:
    """OOS/IS 平均 Sharpe 比值，<0.5 预警过拟合。"""
    is_mean = wf["is_sharpe"].mean()
    if is_mean == 0 or np.isnan(is_mean):
        return float("nan")
    return float(wf["oos_sharpe"].mean() / is_mean)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/validate/walkforward.py tests/test_walkforward.py
git commit -m "feat: walk-forward validation"
```

---

## Task 9: Holdout 闸门

定稿因子仅一次在 holdout 跑最终报告。持久化状态文件记录"已消耗"，再跑即作弊 → 拒绝。

**Files:**
- Create: `quant/validate/gate.py`
- Create: `tests/test_gate.py`

- [ ] **Step 1: 写失败测试**

`tests/test_gate.py`:
```python
import pytest

from quant.validate.gate import assert_not_consumed, is_consumed, mark_consumed


def test_unconsumed_by_default(tmp_path):
    state = tmp_path / "holdout_state.json"
    assert is_consumed("momentum", state) is False


def test_mark_then_consumed(tmp_path):
    state = tmp_path / "sub" / "holdout_state.json"  # 父目录自动建
    mark_consumed("momentum", state)
    assert is_consumed("momentum", state) is True
    # 其他因子不受影响
    assert is_consumed("ma_bias", state) is False


def test_assert_raises_after_consume(tmp_path):
    state = tmp_path / "holdout_state.json"
    assert_not_consumed("momentum", state)  # 未消耗 → 不抛
    mark_consumed("momentum", state)
    with pytest.raises(RuntimeError, match="holdout 已消耗"):
        assert_not_consumed("momentum", state)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_gate.py -v`
Expected: FAIL（ModuleNotFoundError: quant.validate.gate）

- [ ] **Step 3: 实现 gate**

`quant/validate/gate.py`:
```python
"""Holdout 闸门：强制每个因子仅在 holdout 上验证一次。

状态文件 {factor_name: 消耗时间戳}。消耗后再跑 = 作弊 → 拒绝。
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _load(state_path: Path) -> dict:
    p = Path(state_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def is_consumed(factor_name: str, state_path: Path) -> bool:
    """该因子的 holdout 是否已消耗。"""
    return factor_name in _load(state_path)


def mark_consumed(factor_name: str, state_path: Path) -> None:
    """标记该因子 holdout 已消耗（写时间戳）。"""
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _load(p)
    data[factor_name] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def assert_not_consumed(factor_name: str, state_path: Path) -> None:
    """已消耗则抛 RuntimeError。"""
    if is_consumed(factor_name, state_path):
        raise RuntimeError(f"holdout 已消耗：{factor_name}（再跑=作弊）")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_gate.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/validate/gate.py tests/test_gate.py
git commit -m "feat: holdout gate (run-once enforcement)"
```

---

## Task 10: 回测报告卡

把回测指标 + DSR + holdout 状态汇成 `BacktestReport`，渲染 Markdown（红绿灯）。与 Plan 1 的 `FactorReport` 并列，不改动后者。

**Files:**
- Create: `quant/report/backtest_card.py`
- Create: `tests/test_backtest_card.py`

- [ ] **Step 1: 写失败测试**

`tests/test_backtest_card.py`:
```python
from quant.report.backtest_card import BacktestReport


def _report(sharpe=1.5, dsr=0.97, annual=0.18):
    return BacktestReport(
        factor_name="momentum",
        params={"lookback": 252, "skip": 21},
        annual_return=annual,
        sharpe=sharpe,
        max_drawdown=-0.12,
        calmar=1.5,
        monthly_win_rate=0.58,
        avg_turnover=0.20,
        deflated_sharpe=dsr,
        n_trials=8,
        holdout_consumed=False,
    )


def test_markdown_contains_key_fields():
    md = _report().to_markdown()
    assert "momentum" in md
    assert "Sharpe" in md
    assert "DSR" in md
    assert "lookback" in md


def test_green_light_when_strong():
    md = _report(sharpe=1.5, dsr=0.97, annual=0.18).to_markdown()
    assert "🟢" in md


def test_red_light_when_weak():
    md = _report(sharpe=0.3, dsr=0.40, annual=-0.05).to_markdown()
    assert "🔴" in md


def test_holdout_status_shown():
    md = _report().to_markdown()
    assert "holdout" in md.lower()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_backtest_card.py -v`
Expected: FAIL（ModuleNotFoundError: quant.report.backtest_card）

- [ ] **Step 3: 实现 backtest_card**

`quant/report/backtest_card.py`:
```python
"""回测报告卡：BacktestReport dataclass + Markdown 渲染（含红绿灯）。"""

from dataclasses import dataclass


def _light(ok: bool) -> str:
    return "🟢" if ok else "🔴"


@dataclass
class BacktestReport:
    factor_name: str
    params: dict
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    monthly_win_rate: float
    avg_turnover: float
    deflated_sharpe: float
    n_trials: int
    holdout_consumed: bool

    def to_markdown(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        ret_ok = self.annual_return > 0
        sharpe_ok = self.sharpe >= 1.0
        dsr_ok = self.deflated_sharpe >= 0.95
        return f"""# 回测体检报告：{self.factor_name}

参数：{params}

## 红绿灯结论

| 关卡 | 指标 | 判定 |
|---|---|---|
| 含成本收益 | 年化 {self.annual_return:.2%} | {_light(ret_ok)} |
| 风险调整 | Sharpe = {self.sharpe:.2f} | {_light(sharpe_ok)} |
| 抗过拟合 | DSR = {self.deflated_sharpe:.3f}（{self.n_trials} 次试验） | {_light(dsr_ok)} |

## 绩效

- 年化收益：{self.annual_return:.2%}
- Sharpe：{self.sharpe:.2f}
- 最大回撤：{self.max_drawdown:.2%}
- Calmar：{self.calmar:.2f}
- 月胜率：{self.monthly_win_rate:.2%}
- 平均换手：{self.avg_turnover:.2%}

## Holdout 状态

- holdout 已消耗：{"是" if self.holdout_consumed else "否"}
"""
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_backtest_card.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/report/backtest_card.py tests/test_backtest_card.py
git commit -m "feat: backtest scorecard report"
```

---

## Task 11: CLI — `quant backtest`

串通：数据 → 因子 → 处理 → 含成本回测 → 试验台账 → DSR → 报告卡。在 Plan 1 既有 `quant/cli.py` 追加命令与共享 `_make_factor` 助手。

**Files:**
- Modify: `quant/cli.py`
- Create: `tests/test_cli_backtest.py`

- [ ] **Step 1: 写失败测试（用合成 lake，3 标的 → quantiles=3）**

`tests/test_cli_backtest.py`:
```python
from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_backtest_runs_end_to_end(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["backtest", "momentum", "--lookback", "20", "--skip", "1",
         "--quantiles", "3", "--freq", "M", "--mode", "full",
         "--ledger-path", str(tmp_path / "ledger.jsonl"),
         "--state-path", str(tmp_path / "state.json")],
    )
    assert result.exit_code == 0, result.output
    assert "回测体检报告" in result.output
    assert "DSR" in result.output


def test_backtest_unknown_factor_errors(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["backtest", "nope", "--mode", "full",
         "--ledger-path", str(tmp_path / "ledger.jsonl"),
         "--state-path", str(tmp_path / "state.json")],
    )
    assert result.exit_code != 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli_backtest.py -v`
Expected: FAIL（No such command 'backtest'，exit_code != 0 但首个断言 `"回测体检报告" in output` 失败）

- [ ] **Step 3: 在 cli.py 追加 import、共享助手与 backtest 命令**

在 `quant/cli.py` 顶部 import 区追加：
```python
from scipy.stats import kurtosis, skew

from quant.backtest.engine import backtest as run_backtest
from quant.backtest.metrics import compute_metrics, sharpe
from quant.report.backtest_card import BacktestReport
from quant.validate.dsr import deflated_sharpe, expected_max_sharpe
from quant.validate.gate import is_consumed
from quant.validate.ledger import Ledger
```

在 `_FACTORS` 定义之后追加共享助手：
```python
def _make_factor(name: str, close, lookback: int, skip: int, window: int):
    """按名构因子矩阵，返回 (因子, 参数字典)。"""
    if name == "momentum":
        return Momentum(lookback=lookback, skip=skip).compute(close), {
            "lookback": lookback, "skip": skip,
        }
    if name == "ma_bias":
        return MABias(window=window).compute(close), {"window": window}
    raise KeyError(name)


def _backtest_report(
    name, params, factor, close, quantiles, side, freq, cost_bps,
    ledger_path, state_path, holdout_consumed,
) -> BacktestReport:
    """跑回测 + 记台账 + 算 DSR，组装报告卡。"""
    res = run_backtest(factor, close, n=quantiles, side=side, freq=freq, cost_bps=cost_bps)
    metrics = compute_metrics(res.nav, res.returns)
    avg_turnover = float(res.turnover[res.turnover > 0].mean())
    if avg_turnover != avg_turnover:  # NaN（从未建仓）
        avg_turnover = 0.0

    rets = res.returns.dropna()
    per_period_sr = sharpe(res.returns, periods_per_year=1)
    sk = float(skew(rets)) if len(rets) > 2 else 0.0
    ku = float(kurtosis(rets, fisher=False)) if len(rets) > 2 else 3.0

    ledger = Ledger(ledger_path)
    ledger.record({"factor": name, "params": params, "sharpe": per_period_sr})
    n_trials = ledger.count()
    sharpes = ledger.sharpes()
    var_sr = float(pd.Series(sharpes).var(ddof=1)) if len(sharpes) >= 2 else 0.0
    sr0 = expected_max_sharpe(var_sr, n_trials)
    dsr = deflated_sharpe(per_period_sr, sr0, n_obs=len(rets), skew=sk, kurt=ku)

    return BacktestReport(
        factor_name=name,
        params=params,
        annual_return=metrics.annual_return,
        sharpe=metrics.sharpe,
        max_drawdown=metrics.max_drawdown,
        calmar=metrics.calmar,
        monthly_win_rate=metrics.monthly_win_rate,
        avg_turnover=avg_turnover,
        deflated_sharpe=dsr,
        n_trials=n_trials,
        holdout_consumed=holdout_consumed,
    )
```

在文件末尾追加命令：
```python
@app.command("backtest")
def backtest_cmd(
    name: str = typer.Argument(..., help="因子名：momentum / ma_bias"),
    lookback: int = typer.Option(252, help="动量回看窗口"),
    skip: int = typer.Option(21, help="动量跳过窗口"),
    window: int = typer.Option(200, help="均线窗口（ma_bias）"),
    quantiles: int = typer.Option(5, help="分位档数"),
    side: str = typer.Option("long", help="long / long_short"),
    freq: str = typer.Option("M", help="调仓频率 M/W"),
    cost_bps: float = typer.Option(10.0, help="单边成本 bps"),
    mode: str = typer.Option("research", help="research / holdout / full"),
    holdout_years: int = typer.Option(2, help="holdout 锁定年数"),
    ledger_path: str = typer.Option("quant_out/ledger.jsonl", help="试验台账路径"),
    state_path: str = typer.Option("quant_out/holdout_state.json", help="holdout 状态文件"),
) -> None:
    if name not in _FACTORS:
        typer.echo(f"未知因子：{name}，可选 {list(_FACTORS)}", err=True)
        raise typer.Exit(code=1)

    close = load_price_matrix(field="close", market="us")
    close = apply_holdout(close, mode=mode, holdout_years=holdout_years)
    factor, params = _make_factor(name, close, lookback, skip, window)
    factor = Pipeline([winsorize, zscore])(factor)

    report = _backtest_report(
        name, params, factor, close, quantiles, side, freq, cost_bps,
        ledger_path, state_path, holdout_consumed=is_consumed(name, state_path),
    )
    typer.echo(report.to_markdown())
```

注：`pd` 已在 Plan 1 的 cli.py 顶部导入（Task 14 未显式 import pandas）。若未导入，在 import 区补 `import pandas as pd`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli_backtest.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/cli.py tests/test_cli_backtest.py
git commit -m "feat: quant backtest CLI (with DSR + ledger)"
```

---

## Task 12: CLI — `quant combine`

多因子合成 → 共线性预警 → 含成本回测合成总分。复用 `_make_factor` 与回测引擎。

**Files:**
- Modify: `quant/cli.py`
- Create: `tests/test_cli_combine.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cli_combine.py`:
```python
from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_combine_runs_end_to_end(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["combine", "momentum", "ma_bias", "--weighting", "equal",
         "--lookback", "20", "--skip", "1", "--window", "10",
         "--quantiles", "3", "--freq", "M", "--mode", "full"],
    )
    assert result.exit_code == 0, result.output
    assert "合成回测" in result.output
    assert "权重" in result.output


def test_combine_unknown_factor_errors(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app, ["combine", "momentum", "nope", "--mode", "full"]
    )
    assert result.exit_code != 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli_combine.py -v`
Expected: FAIL（No such command 'combine'）

- [ ] **Step 3: 在 cli.py 追加 combine import 与命令**

在 import 区追加：
```python
from quant.data.returns import forward_returns
from quant.eval.ic import ic_series, ic_summary
from quant.combine.synth import (
    combine_score,
    equal_weight,
    factor_correlation,
    high_correlation_warnings,
    ic_weight,
    zscore_factors,
)
```
（`forward_returns` / `ic_series` / `ic_summary` 若 Task 14 已导入则跳过重复。）

文件末尾追加命令：
```python
@app.command("combine")
def combine_cmd(
    names: list[str] = typer.Argument(..., help="因子名列表：momentum ma_bias"),
    weighting: str = typer.Option("equal", help="equal / ic"),
    lookback: int = typer.Option(252, help="动量回看窗口"),
    skip: int = typer.Option(21, help="动量跳过窗口"),
    window: int = typer.Option(200, help="均线窗口（ma_bias）"),
    horizon: int = typer.Option(21, help="IC 加权用的前瞻收益天数"),
    quantiles: int = typer.Option(5, help="分位档数"),
    side: str = typer.Option("long", help="long / long_short"),
    freq: str = typer.Option("M", help="调仓频率 M/W"),
    cost_bps: float = typer.Option(10.0, help="单边成本 bps"),
    mode: str = typer.Option("research", help="research / holdout / full"),
    holdout_years: int = typer.Option(2, help="holdout 锁定年数"),
) -> None:
    for nm in names:
        if nm not in _FACTORS:
            typer.echo(f"未知因子：{nm}，可选 {list(_FACTORS)}", err=True)
            raise typer.Exit(code=1)

    close = load_price_matrix(field="close", market="us")
    close = apply_holdout(close, mode=mode, holdout_years=holdout_years)

    raw = {nm: _make_factor(nm, close, lookback, skip, window)[0] for nm in names}
    zf = zscore_factors(raw)

    if weighting == "ic":
        fwd = forward_returns(close, horizon=horizon)
        ic_means = {
            nm: ic_summary(ic_series(f, fwd, method="spearman"))["ic_mean"]
            for nm, f in zf.items()
        }
        weights = ic_weight(ic_means)
    else:
        weights = equal_weight(names)

    score = combine_score(zf, weights)
    corr = factor_correlation(zf)
    warns = high_correlation_warnings(corr, threshold=0.7)

    res = run_backtest(score, close, n=quantiles, side=side, freq=freq, cost_bps=cost_bps)
    metrics = compute_metrics(res.nav, res.returns)

    lines = ["# 合成回测", "", "## 权重"]
    lines += [f"- {nm}：{w:.3f}" for nm, w in weights.items()]
    lines += ["", "## 共线性预警"]
    if warns:
        lines += [f"- {a} ~ {b}：相关 {v:.2f}" for a, b, v in warns]
    else:
        lines += ["- 无（阈值 0.7）"]
    lines += [
        "", "## 绩效",
        f"- 年化收益：{metrics.annual_return:.2%}",
        f"- Sharpe：{metrics.sharpe:.2f}",
        f"- 最大回撤：{metrics.max_drawdown:.2%}",
        f"- 月胜率：{metrics.monthly_win_rate:.2%}",
    ]
    typer.echo("\n".join(lines))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli_combine.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/cli.py tests/test_cli_combine.py
git commit -m "feat: quant combine CLI (multi-factor synthesis backtest)"
```

---

## Task 13: CLI — `quant holdout`（闸门）

定稿因子仅一次在 holdout 跑最终回测，跑前确认、跑后标记已消耗；已消耗再跑直接拒绝。

**Files:**
- Modify: `quant/cli.py`
- Create: `tests/test_cli_holdout.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cli_holdout.py`:
```python
from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def _args(tmp_path):
    return [
        "holdout", "momentum", "--yes",
        "--lookback", "20", "--skip", "1", "--quantiles", "3",
        "--freq", "M", "--holdout-years", "0",
        "--ledger-path", str(tmp_path / "ledger.jsonl"),
        "--state-path", str(tmp_path / "state.json"),
    ]


def test_holdout_runs_and_marks_consumed(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(app, _args(tmp_path))
    assert result.exit_code == 0, result.output
    assert "回测体检报告" in result.output
    assert "holdout 已消耗：是" in result.output


def test_holdout_second_run_rejected(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    first = runner.invoke(app, _args(tmp_path))
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, _args(tmp_path))
    assert second.exit_code != 0
    assert "已消耗" in second.output
```

注：`fake_lake` 数据全在 2020 年；`--holdout-years 0` 使 `apply_holdout(mode="holdout")` 取 `index > 末日往前 0 年` 仍保留末日点。为让 holdout 窗口非空可跑回测，holdout 命令固定用 `mode="holdout"`，并对空窗口给出友好报错（见实现 Step 3 守卫）。测试用 `holdout_years=0` 触发"保留尾部"路径，保证有数据。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli_holdout.py -v`
Expected: FAIL（No such command 'holdout'）

- [ ] **Step 3: 在 cli.py 追加 holdout import 与命令**

在 import 区追加：
```python
from quant.validate.gate import assert_not_consumed, mark_consumed
```
（与 Task 11 的 `is_consumed` 同源，合并为 `from quant.validate.gate import assert_not_consumed, is_consumed, mark_consumed`。）

文件末尾追加命令：
```python
@app.command("holdout")
def holdout_cmd(
    name: str = typer.Argument(..., help="定稿因子名"),
    lookback: int = typer.Option(252, help="动量回看窗口"),
    skip: int = typer.Option(21, help="动量跳过窗口"),
    window: int = typer.Option(200, help="均线窗口（ma_bias）"),
    quantiles: int = typer.Option(5, help="分位档数"),
    side: str = typer.Option("long", help="long / long_short"),
    freq: str = typer.Option("M", help="调仓频率 M/W"),
    cost_bps: float = typer.Option(10.0, help="单边成本 bps"),
    holdout_years: int = typer.Option(2, help="holdout 锁定年数"),
    yes: bool = typer.Option(False, "--yes", help="跳过确认弹窗"),
    ledger_path: str = typer.Option("quant_out/ledger.jsonl", help="试验台账路径"),
    state_path: str = typer.Option("quant_out/holdout_state.json", help="holdout 状态文件"),
) -> None:
    if name not in _FACTORS:
        typer.echo(f"未知因子：{name}，可选 {list(_FACTORS)}", err=True)
        raise typer.Exit(code=1)

    # 已消耗 → 拒绝（作弊保护）
    try:
        assert_not_consumed(name, state_path)
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    if not yes:
        typer.confirm(
            f"将在 holdout 上对 {name} 跑最终验证，仅此一次，之后锁定。确认？",
            abort=True,
        )

    close = load_price_matrix(field="close", market="us")
    close = apply_holdout(close, mode="holdout", holdout_years=holdout_years)
    if close.empty:
        typer.echo("holdout 窗口为空（数据时长不足锁定年数）", err=True)
        raise typer.Exit(code=1)

    factor, params = _make_factor(name, close, lookback, skip, window)
    factor = Pipeline([winsorize, zscore])(factor)

    report = _backtest_report(
        name, params, factor, close, quantiles, side, freq, cost_bps,
        ledger_path, state_path, holdout_consumed=True,
    )
    mark_consumed(name, state_path)
    typer.echo(report.to_markdown())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli_holdout.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 全量测试 + lint**

Run:
```bash
uv run pytest -v
uv run ruff check .
```
Expected: 全部 PASS（含 Plan 1 既有测试），ruff 无错误。

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py tests/test_cli_holdout.py
git commit -m "feat: quant holdout gate CLI (run-once)"
```

---

## Task 14: 真实数据冒烟验证

确认对真实 data_lake 能跑通新命令（手动，非自动化测试，避免 CI 依赖真实数据）。

**Files:** 无（手动验证）

- [ ] **Step 1: 真实数据单因子回测**

Run:
```bash
uv run quant backtest momentum --mode research --side long --freq M
```
Expected: 输出"回测体检报告"，年化/Sharpe/最大回撤/Calmar/月胜率/DSR 有数值，无异常。`quant_out/ledger.jsonl` 多一行。

- [ ] **Step 2: 多空 + 周频**

Run:
```bash
uv run quant backtest ma_bias --window 200 --mode research --side long_short --freq W
```
Expected: 同上，多空净敞口回测跑通。

- [ ] **Step 3: 多因子合成**

Run:
```bash
uv run quant combine momentum ma_bias --weighting ic --mode research
```
Expected: 输出权重、共线性预警、合成绩效，无异常。

- [ ] **Step 4: holdout 闸门（仅一次）**

Run:
```bash
uv run quant holdout momentum --mode research  # 注：holdout 命令固定 holdout 模式，无 --mode
uv run quant holdout momentum
```
注意：holdout 命令无 `--mode` 选项（固定 holdout 模式）。正确调用：
```bash
uv run quant holdout momentum            # 第一次：确认后跑，标记消耗
uv run quant holdout momentum            # 第二次：应被拒绝"holdout 已消耗"
```
Expected: 第一次跑出报告且"holdout 已消耗：是"；第二次非零退出并提示已消耗。验证后可删 `quant_out/holdout_state.json` 复位。

- [ ] **Step 5: 记录基线耗时**

Run:
```bash
time uv run quant backtest momentum --mode research
```
Expected: 记录耗时。目标单因子回测 < 2 秒（首次含 IO 可放宽）。若远超，记为 Plan 2b 缓存层输入。

- [ ] **Step 6: 无需 commit（纯验证）**

---

## Self-Review 记录

**Spec 覆盖**（对照 spec §7 多因子合成 + §8 真实回测/抗过拟合/holdout + §9.1 报告卡）：

| Spec 要求 | 实现位置 |
|---|---|
| §7 逐因子 zscore | Task 5 `zscore_factors` |
| §7 等权 / IC 加权 | Task 5 `equal_weight` / `ic_weight` |
| §7 合成总分 → 选股 | Task 5 `combine_score` + Task 12 CLI 回测 |
| §7 共线性检查（相关矩阵 + 预警） | Task 5 `factor_correlation` / `high_correlation_warnings` |
| §7 正交化预留 v2 | 明确不做（Plan 2b/v2），文档已注 |
| §8.1 默认 Q5 多头 / Q5−Q1 多空 | Task 3 `target_weights(side=...)` |
| §8.1 等权、月度（可周频）调仓 | Task 3 `rebalance_dates(freq=M/W)` |
| §8.1 单边 bps 成本按换手扣 | Task 4 `backtest(cost_bps=...)` |
| §8.1 净值/年化/Sharpe/回撤/Calmar/月胜率 | Task 2 `metrics.py` |
| §8.2 Walk-forward | Task 8 `walk_forward` |
| §8.2 Deflated Sharpe | Task 7 `deflated_sharpe` |
| §8.2 试验次数台账 ledger.jsonl | Task 6 `Ledger` + Task 11 记账 |
| §8.2 OOS/IS 比值 <0.5 报警 | Task 8 `oos_is_ratio` |
| §8.3 holdout 物理隔离 | Plan 1 `apply_holdout`（直接复用） |
| §8.3 定稿仅一次 + 标记已消耗 | Task 9 `gate.py` + Task 13 CLI |
| §9.1 报告卡红绿灯（含 DSR/holdout） | Task 10 `BacktestReport` |

**本计划明确不含（Plan 2b / v2）**：Streamlit web 四页、其余因子族（波动/流动性…）、行业中性化、缓存层、因子正交化、PBO、风险模型。

**类型一致性核对：**
- `BacktestResult(nav, returns, turnover)` 在 Task 4 定义；Task 8 walk-forward 与 Task 11/12/13 CLI 均用 `.returns` / `.nav` / `.turnover`，字段名一致。
- `BacktestMetrics(annual_return, sharpe, max_drawdown, calmar, monthly_win_rate)` 在 Task 2 定义；Task 11/12 CLI 逐字段引用一致。
- `compute_metrics(nav, returns, periods_per_year=252)`、`sharpe(returns, periods_per_year=252)` 在 Task 2 定义；Task 11 用 `sharpe(res.returns, periods_per_year=1)` 取每期 SR 喂 DSR，签名一致。
- `expected_max_sharpe(sr_variance, n_trials)` / `deflated_sharpe(sr, sr0, n_obs, skew=, kurt=)` 在 Task 7 定义；Task 11 `_backtest_report` 调用参数名一致。
- `Ledger.record/count/sharpes` 在 Task 6 定义；Task 11 调用一致。
- `is_consumed/mark_consumed/assert_not_consumed(factor_name, state_path)` 在 Task 9 定义；Task 11/13 CLI 调用一致。
- `_make_factor(name, close, lookback, skip, window)` 在 Task 11 定义，Task 12/13 复用，签名一致。
- `BacktestReport(...)` 字段在 Task 10 定义；Task 11 `_backtest_report` 构造时逐字段对应。

**已知前置假设**：Plan 1 的 Task 1–15 须先完成（`load_price_matrix`/`apply_holdout`/`Momentum`/`MABias`/`Pipeline`/`winsorize`/`zscore`/`ic_series`/`ic_summary`/`forward_returns` 及 `quant/cli.py` 的 `app`、`_FACTORS`、相关 import 已就位）。当前仓库 Plan 1 仅执行至 Task 3，执行本计划前需补完 Plan 1 Task 4–15。
