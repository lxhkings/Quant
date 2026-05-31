# Quant 因子族扩充 + 行业中性化 + 缓存 + Web Implementation Plan (Plan 2b/2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Plan 1（数据/单因子检验）+ Plan 2a（合成/回测/抗过拟合/holdout 闸门）之上，扩一批价量因子族、加行业中性化、加 parquet 缓存层、出 Streamlit web 四页，把研究闭环装进可点的界面。

**Architecture:** 因子全部走 `Factor` 基类 + 新建 `factor/registry.py` 统一注册（消除 CLI 里的 if/else 分发）。行业中性化作为可选处理步，在横截面内按 GICS 去均值。缓存层封 `load_or_compute(key, fn, dir)`，对前瞻收益/因子矩阵按 key 落 parquet。Web 分两层：`web/viewmodel.py` 纯函数（复用 Plan 1/2a 既有计算，可单测），`web/app.py` 薄 Streamlit 壳。依赖方向 `web → {report.runner, viewmodel deps} → {combine, backtest, validate, eval, process, factor.registry, data, cache}`，无环。

**Tech Stack:** Python 3.11、pandas 2.2、numpy、scipy、typer、pytest；新增可选 `streamlit`（放 `[project.optional-dependencies].web`，核心测试不依赖）。

**前置依赖（必须先执行）：** Plan 1（全 15 task，已完成）+ Plan 2a（全 14 task）。本计划直接调用二者产出的接口（见下表）。**写作时 Plan 2a 尚未执行**，若 Plan 2a 实现与其计划接口漂移，执行本计划前需对齐。

| 接口 | 来源 | 签名 |
|---|---|---|
| 价格/成交量面板 | Plan 1 `data/panel.py` | `load_price_matrix(field="close"/"volume", market="us", root=None) -> DataFrame` |
| holdout 截断 | Plan 1 `data/holdout.py` | `apply_holdout(matrix, mode, holdout_years) -> DataFrame` |
| 前瞻收益 | Plan 1 `data/returns.py` | `forward_returns(close, horizon) -> DataFrame` |
| 因子基类 | Plan 1 `factor/base.py` | `Factor.compute(self, close, volume=None) -> DataFrame` |
| 既有因子 | Plan 1 | `Momentum(lookback,skip)`, `MABias(window)` |
| 处理链 | Plan 1 `process/pipeline.py` | `winsorize/zscore/Pipeline` |
| IC/分位 | Plan 1 `eval/` | `ic_series/ic_summary/quantile_returns/long_short_spread/turnover` |
| 单因子报告卡 | Plan 1 `report/scorecard.py` | `FactorReport` |
| 回测引擎 | Plan 2a `backtest/engine.py` | `backtest(factor, close, n, side, freq, cost_bps) -> BacktestResult` |
| 绩效 | Plan 2a `backtest/metrics.py` | `compute_metrics(nav, returns) -> BacktestMetrics`, `sharpe(returns, periods_per_year)` |
| 合成 | Plan 2a `combine/synth.py` | `zscore_factors/equal_weight/ic_weight/combine_score/factor_correlation/high_correlation_warnings` |
| 台账/DSR | Plan 2a `validate/` | `Ledger`, `expected_max_sharpe`, `deflated_sharpe` |
| holdout 闸门 | Plan 2a `validate/gate.py` | `is_consumed/mark_consumed/assert_not_consumed(factor_name, state_path)` |
| 回测报告卡 | Plan 2a `report/backtest_card.py` | `BacktestReport` |
| CLI | Plan 2a `cli.py` | `app`, `_FACTORS`, `_make_factor`, `_backtest_report`, `backtest_cmd`, `combine_cmd`, `holdout_cmd` |
| 合成 fixture | Plan 1 `tests/conftest.py` | `fake_lake` → `(root, instruments[3]=AAA/BBB/CCC, days[60])`；sectors：AAA/BBB=Tech，CCC=Energy |

---

## File Structure

| 文件 | 职责 |
|---|---|
| `quant/factor/library/short_reversal.py` | `ShortReversal` 短期反转因子 |
| `quant/factor/library/volatility.py` | `Volatility` 已实现波动率因子 |
| `quant/factor/library/rel_volume.py` | `RelativeVolume` 相对量能因子（用 volume） |
| `quant/factor/library/amihud.py` | `Amihud` 非流动性因子（用 close+volume） |
| `quant/factor/registry.py` | 因子注册表：`factor_names/make/needs_volume/compute_factor` |
| `quant/data/sectors.py` | `load_sectors()`：instrument_id → GICS 行业 |
| `quant/process/neutralize.py` | `sector_neutralize()`：横截面行业内去均值 |
| `quant/cache/__init__.py` / `quant/cache/store.py` | `cache_key()` / `load_or_compute()` parquet 缓存 |
| `quant/report/runner.py` | `run_backtest_report()`：从 Plan 2a 私有 `_backtest_report` 提取的公共服务（CLI + web 共用） |
| `quant/cli.py`（修改） | `_make_factor` 改注册表分发 + 新因子 + `--neutralize`；`factor_test` 复用 `_make_factor`；改用 `report.runner` |
| `quant/web/__init__.py` / `quant/web/viewmodel.py` | 四页纯逻辑：`workshop/combine/holdout_run/history` |
| `quant/web/app.py` | Streamlit 四页壳 |
| `pyproject.toml`（修改） | 加 `[web]` 可选依赖 `streamlit` |

**本计划明确不做（v2）：** 因子正交化（去共线残差）、PBO、风险模型、其余更冷门因子族（趋势 R²/ADX、Beta/偏度、规模代理等——按本计划 registry 同一 `Factor` 模式自行追加即可）、并行扫参。

---

## Task 1: 短期反转因子

**Files:**
- Create: `quant/factor/library/short_reversal.py`
- Create: `tests/test_factor_extra.py`

- [ ] **Step 1: 写失败测试**

`tests/test_factor_extra.py`:
```python
import numpy as np
import pandas as pd

from quant.factor.library.short_reversal import ShortReversal


def test_short_reversal_name():
    assert ShortReversal().name == "short_reversal"


def test_short_reversal_negative_when_rising():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0 * (1.01 ** i) for i in range(30)]}, index=idx)
    f = ShortReversal(window=21).compute(close)
    # 过去一段持续上涨 → 反转因子 = -过去收益 < 0
    assert f["AAA"].iloc[-1] < 0
    # 窗口不足处为 NaN
    assert np.isnan(f["AAA"].iloc[0])


def test_short_reversal_zero_when_flat():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 30}, index=idx)
    f = ShortReversal(window=21).compute(close)
    assert np.isclose(f["AAA"].iloc[-1], 0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_factor_extra.py -v`
Expected: FAIL（ModuleNotFoundError: short_reversal）

- [ ] **Step 3: 实现 short_reversal**

`quant/factor/library/short_reversal.py`:
```python
"""短期反转因子：过去 window 日涨幅取负 = -(close/close.shift(window) - 1)。

超跌反弹——近期跌得多的未来反弹概率高，故取负使"跌多=高分"。
"""

import pandas as pd

from quant.factor.base import Factor


class ShortReversal(Factor):
    name = "short_reversal"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        past_return = close / close.shift(self.window) - 1.0
        return -past_return
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_factor_extra.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/factor/library/short_reversal.py tests/test_factor_extra.py
git commit -m "feat: short reversal factor"
```

---

## Task 2: 已实现波动率因子

**Files:**
- Create: `quant/factor/library/volatility.py`
- Modify: `tests/test_factor_extra.py`（追加）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_factor_extra.py` 末尾追加：
```python
from quant.factor.library.volatility import Volatility


def test_volatility_name():
    assert Volatility().name == "volatility"


def test_volatility_zero_when_flat():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    f = Volatility(window=21).compute(close)
    assert np.isclose(f["AAA"].iloc[-1], 0.0)


def test_volatility_positive_when_choppy():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    prices = [100.0 + (5.0 if i % 2 else -5.0) for i in range(40)]
    close = pd.DataFrame({"AAA": prices}, index=idx)
    f = Volatility(window=21).compute(close)
    assert f["AAA"].iloc[-1] > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_factor_extra.py -k volatility -v`
Expected: FAIL（ModuleNotFoundError: volatility）

- [ ] **Step 3: 实现 volatility**

`quant/factor/library/volatility.py`:
```python
"""已实现波动率因子：过去 window 日日收益的标准差。

低波异象——低波动标的长期风险调整后收益更优（信号方向由检验环节判定）。
"""

import pandas as pd

from quant.factor.base import Factor


class Volatility(Factor):
    name = "volatility"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        daily_ret = close.pct_change()
        return daily_ret.rolling(self.window).std()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_factor_extra.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/factor/library/volatility.py tests/test_factor_extra.py
git commit -m "feat: realized volatility factor"
```

---

## Task 3: 相对量能因子（用 volume）

**Files:**
- Create: `quant/factor/library/rel_volume.py`
- Modify: `tests/test_factor_extra.py`（追加）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_factor_extra.py` 末尾追加：
```python
import pytest

from quant.factor.library.rel_volume import RelativeVolume


def test_rel_volume_name():
    assert RelativeVolume().name == "rel_volume"


def test_rel_volume_zero_when_constant():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    vol = pd.DataFrame({"AAA": [1_000_000.0] * 40}, index=idx)
    f = RelativeVolume(window=21).compute(close, vol)
    # 量恒定 → 相对均量 = 0
    assert np.isclose(f["AAA"].iloc[-1], 0.0)


def test_rel_volume_positive_on_spike():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    vols = [1_000_000.0] * 39 + [5_000_000.0]  # 末日放量
    vol = pd.DataFrame({"AAA": vols}, index=idx)
    f = RelativeVolume(window=21).compute(close, vol)
    assert f["AAA"].iloc[-1] > 0


def test_rel_volume_requires_volume():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 5}, index=idx)
    with pytest.raises(ValueError, match="volume"):
        RelativeVolume().compute(close, None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_factor_extra.py -k rel_volume -v`
Expected: FAIL（ModuleNotFoundError: rel_volume）

- [ ] **Step 3: 实现 rel_volume**

`quant/factor/library/rel_volume.py`:
```python
"""相对量能因子：成交量相对过去 window 日均量的偏离 = volume / MA(volume) - 1。

放量异动捕捉资金动向。需 volume 输入。
"""

import pandas as pd

from quant.factor.base import Factor


class RelativeVolume(Factor):
    name = "rel_volume"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        if volume is None:
            raise ValueError("rel_volume 因子需要 volume 输入")
        ma = volume.rolling(self.window).mean()
        return volume / ma - 1.0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_factor_extra.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/factor/library/rel_volume.py tests/test_factor_extra.py
git commit -m "feat: relative volume factor"
```

---

## Task 4: Amihud 非流动性因子（用 close+volume）

**Files:**
- Create: `quant/factor/library/amihud.py`
- Modify: `tests/test_factor_extra.py`（追加）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_factor_extra.py` 末尾追加：
```python
from quant.factor.library.amihud import Amihud


def test_amihud_name():
    assert Amihud().name == "amihud"


def test_amihud_lower_for_higher_dollar_volume():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    # 两标的价格走势相同（→ |日收益| 相同），但 BBB 成交量是 AAA 的 10 倍
    prices = [100.0 * (1 + (0.01 if i % 2 else -0.01)) ** i for i in range(40)]
    close = pd.DataFrame({"AAA": prices, "BBB": prices}, index=idx)
    vol = pd.DataFrame({"AAA": [1e6] * 40, "BBB": [1e7] * 40}, index=idx)
    f = Amihud(window=21).compute(close, vol)
    # 成交额大 → 非流动性低
    assert f["BBB"].iloc[-1] < f["AAA"].iloc[-1]


def test_amihud_requires_volume():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 5}, index=idx)
    with pytest.raises(ValueError, match="volume"):
        Amihud().compute(close, None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_factor_extra.py -k amihud -v`
Expected: FAIL（ModuleNotFoundError: amihud）

- [ ] **Step 3: 实现 amihud**

`quant/factor/library/amihud.py`:
```python
"""Amihud 非流动性因子：过去 window 日 |日收益| / 成交额 的均值。

成交额 = close × volume。值越大越不流动 → 流动性溢价（信号方向由检验判定）。
"""

import pandas as pd

from quant.factor.base import Factor


class Amihud(Factor):
    name = "amihud"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        if volume is None:
            raise ValueError("amihud 因子需要 volume 输入")
        daily_ret = close.pct_change().abs()
        dollar_vol = close * volume
        illiq = daily_ret / dollar_vol
        return illiq.rolling(self.window).mean()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_factor_extra.py -v`
Expected: PASS（12 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/factor/library/amihud.py tests/test_factor_extra.py
git commit -m "feat: amihud illiquidity factor"
```

---

## Task 5: 因子注册表

统一注册全部因子，消除 CLI/web 里的 if/else 分发，并标注哪些需 volume。

**Files:**
- Create: `quant/factor/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: 写失败测试**

`tests/test_registry.py`:
```python
import numpy as np
import pandas as pd
import pytest

from quant.factor.registry import compute_factor, factor_names, make, needs_volume


def test_factor_names_includes_all():
    names = set(factor_names())
    assert {"momentum", "ma_bias", "short_reversal", "volatility",
            "rel_volume", "amihud"} <= names


def test_make_builds_instance():
    f = make("momentum", lookback=20, skip=1)
    assert f.name == "momentum"


def test_make_unknown_raises():
    with pytest.raises(KeyError):
        make("nope")


def test_needs_volume_flags():
    assert needs_volume("amihud") is True
    assert needs_volume("rel_volume") is True
    assert needs_volume("momentum") is False


def test_compute_factor_close_only():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0 * (1.01 ** i) for i in range(30)]}, index=idx)
    out = compute_factor("momentum", close, lookback=20, skip=1)
    assert out.shape == close.shape


def test_compute_factor_with_volume():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 30}, index=idx)
    vol = pd.DataFrame({"AAA": [1e6] * 30}, index=idx)
    out = compute_factor("rel_volume", close, volume=vol, window=21)
    assert np.isclose(out["AAA"].iloc[-1], 0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL（ModuleNotFoundError: quant.factor.registry）

- [ ] **Step 3: 实现 registry**

`quant/factor/registry.py`:
```python
"""因子注册表：名字 → 类，统一构建与分发，标注 volume 依赖。"""

import pandas as pd

from quant.factor.base import Factor
from quant.factor.library.amihud import Amihud
from quant.factor.library.ma_bias import MABias
from quant.factor.library.momentum import Momentum
from quant.factor.library.rel_volume import RelativeVolume
from quant.factor.library.short_reversal import ShortReversal
from quant.factor.library.volatility import Volatility

_REGISTRY: dict[str, type[Factor]] = {
    "momentum": Momentum,
    "ma_bias": MABias,
    "short_reversal": ShortReversal,
    "volatility": Volatility,
    "rel_volume": RelativeVolume,
    "amihud": Amihud,
}
_NEEDS_VOLUME = {"rel_volume", "amihud"}


def factor_names() -> list[str]:
    """全部已注册因子名。"""
    return list(_REGISTRY)


def make(name: str, **params) -> Factor:
    """按名构建因子实例。未知名抛 KeyError。"""
    if name not in _REGISTRY:
        raise KeyError(f"未知因子：{name}，可选 {factor_names()}")
    return _REGISTRY[name](**params)


def needs_volume(name: str) -> bool:
    """该因子是否需要 volume 输入。"""
    return name in _NEEDS_VOLUME


def compute_factor(
    name: str,
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    **params,
) -> pd.DataFrame:
    """构建并计算因子；需 volume 的因子自动传入。"""
    f = make(name, **params)
    if needs_volume(name):
        return f.compute(close, volume)
    return f.compute(close)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/factor/registry.py tests/test_registry.py
git commit -m "feat: factor registry"
```

---

## Task 6: 行业板块加载

从 sectors 数据集读 instrument_id → GICS 行业（取每标的最新一条）。

**Files:**
- Create: `quant/data/sectors.py`
- Create: `tests/test_sectors.py`

- [ ] **Step 1: 写失败测试**

`tests/test_sectors.py`:
```python
from quant.data.sectors import load_sectors


def test_load_sectors_maps_instrument_to_sector(fake_lake):
    root, _, _ = fake_lake
    sec = load_sectors(market="us", root=root)
    assert sec["AAA"] == "Tech"
    assert sec["BBB"] == "Tech"
    assert sec["CCC"] == "Energy"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_sectors.py -v`
Expected: FAIL（ModuleNotFoundError: quant.data.sectors）

- [ ] **Step 3: 实现 sectors**

`quant/data/sectors.py`:
```python
"""行业板块加载：sectors 数据集 → instrument_id 到 GICS 行业的映射。"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from quant.config import data_lake_root


def load_sectors(market: str = "us", root: Path | None = None) -> pd.Series:
    """返回 index=instrument_id、值=sector 的 Series（每标的取最新一条记录）。"""
    root = root or data_lake_root()
    path = Path(root) / market / "sectors"
    df = (
        ds.dataset(str(path), format="parquet")
        .to_table(columns=["instrument_id", "date", "sector"])
        .to_pandas()
    )
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("instrument_id").last()
    return latest["sector"]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_sectors.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/data/sectors.py tests/test_sectors.py
git commit -m "feat: sector mapping loader"
```

---

## Task 7: 行业中性化处理步

横截面内按行业去均值（GICS 中性化），作为可选前置处理。

**Files:**
- Create: `quant/process/neutralize.py`
- Create: `tests/test_neutralize.py`

- [ ] **Step 1: 写失败测试**

`tests/test_neutralize.py`:
```python
import numpy as np
import pandas as pd

from quant.process.neutralize import sector_neutralize


def test_sector_neutralize_demeans_within_sector():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C", "D"]
    factor = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]] * 2, index=idx, columns=cols)
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = sector_neutralize(factor, sectors)
    # Tech {A,B} 均值 2 → A=-1,B=1；Energy {C,D} 均值 15 → C=-5,D=5
    assert np.allclose(out.iloc[0].values, [-1.0, 1.0, -5.0, 5.0])
    # 每个行业内逐行均值为 0
    assert np.isclose(out[["A", "B"]].iloc[0].mean(), 0.0)
    assert np.isclose(out[["C", "D"]].iloc[0].mean(), 0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_neutralize.py -v`
Expected: FAIL（ModuleNotFoundError: quant.process.neutralize）

- [ ] **Step 3: 实现 neutralize**

`quant/process/neutralize.py`:
```python
"""行业中性化：每个截面在各 GICS 行业内对因子值去均值。

剥掉行业整体偏移，留下行业内相对强弱。无行业映射的标的保持原值。
"""

import pandas as pd


def sector_neutralize(factor: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """逐截面（逐行）在行业内减去该行业均值。"""
    grp = sectors.reindex(factor.columns)
    out = factor.copy()
    for _, members in grp.groupby(grp):
        cols = list(members.index)
        sub = factor[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_neutralize.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/process/neutralize.py tests/test_neutralize.py
git commit -m "feat: sector neutralization"
```

---

## Task 8: parquet 缓存层

算过的矩阵按 key 落 parquet，再次请求直接读，避免重算。

**Files:**
- Create: `quant/cache/__init__.py`
- Create: `quant/cache/store.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cache.py`:
```python
import pandas as pd

from quant.cache.store import cache_key, load_or_compute


def test_cache_key_stable_and_distinct():
    k1 = cache_key("factor", "momentum", {"lookback": 252})
    k2 = cache_key("factor", "momentum", {"lookback": 252})
    k3 = cache_key("factor", "momentum", {"lookback": 100})
    assert k1 == k2
    assert k1 != k3


def test_load_or_compute_caches(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return pd.DataFrame({"AAA": [1.0, 2.0]})

    key = cache_key("test")
    first = load_or_compute(key, compute, tmp_path / "cache")
    second = load_or_compute(key, compute, tmp_path / "cache")
    # 第二次命中缓存，compute 不再调用
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL（ModuleNotFoundError: quant.cache）

- [ ] **Step 3: 实现 cache**

`quant/cache/__init__.py`:
```python
```

`quant/cache/store.py`:
```python
"""矩阵缓存：按 key 落 parquet，命中即读，未命中算后写。"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def cache_key(*parts) -> str:
    """把任意可序列化片段拼成稳定短哈希。"""
    raw = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def load_or_compute(
    key: str, compute: Callable[[], pd.DataFrame], cache_dir: Path
) -> pd.DataFrame:
    """命中 `<cache_dir>/<key>.parquet` 则读回，否则算后写。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{key}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    df = compute()
    df.to_parquet(fp)
    return df
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cache.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/cache/ tests/test_cache.py
git commit -m "feat: parquet matrix cache"
```

---

## Task 9: 回测报告服务（提取公共）

把 Plan 2a `cli.py` 里的私有 `_backtest_report` 提取成公共 `report/runner.py::run_backtest_report`，供 CLI 与 web 共用（DRY）。

**Files:**
- Create: `quant/report/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: 写失败测试**

`tests/test_runner.py`:
```python
import pandas as pd

from quant.report.backtest_card import BacktestReport
from quant.report.runner import run_backtest_report


def _world():
    idx = pd.bdate_range("2020-01-06", periods=5)
    cols = list("ABCDEFGHIJ")
    factor = pd.DataFrame([list(range(1, 11))] * 5, index=idx, columns=cols, dtype=float)
    close = pd.DataFrame(
        {c: [100.0 * (1 + 0.001 * i) ** t for t in range(5)] for i, c in enumerate(cols)},
        index=idx,
    )
    return factor, close


def test_run_backtest_report_records_trial(tmp_path):
    factor, close = _world()
    rep = run_backtest_report(
        "momentum", {"lookback": 1}, factor, close,
        quantiles=5, side="long", freq="M", cost_bps=0.0,
        ledger_path=tmp_path / "ledger.jsonl", holdout_consumed=False,
    )
    assert isinstance(rep, BacktestReport)
    assert rep.factor_name == "momentum"
    assert rep.n_trials == 1  # 台账记了一次试验


def test_run_backtest_report_trials_accumulate(tmp_path):
    factor, close = _world()
    lp = tmp_path / "ledger.jsonl"
    run_backtest_report("momentum", {"lookback": 1}, factor, close,
                        ledger_path=lp, holdout_consumed=False)
    rep2 = run_backtest_report("momentum", {"lookback": 1}, factor, close,
                              ledger_path=lp, holdout_consumed=False)
    assert rep2.n_trials == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL（ModuleNotFoundError: quant.report.runner）

- [ ] **Step 3: 实现 runner**

`quant/report/runner.py`:
```python
"""回测报告服务：跑回测 + 记试验台账 + 算 DSR，组装 BacktestReport。

CLI 与 web 共用，避免重复编排。
"""

from pathlib import Path

import pandas as pd
from scipy.stats import kurtosis, skew

from quant.backtest.engine import backtest as run_backtest
from quant.backtest.metrics import compute_metrics, sharpe
from quant.report.backtest_card import BacktestReport
from quant.validate.dsr import deflated_sharpe, expected_max_sharpe
from quant.validate.ledger import Ledger


def run_backtest_report(
    name: str,
    params: dict,
    factor: pd.DataFrame,
    close: pd.DataFrame,
    *,
    quantiles: int = 5,
    side: str = "long",
    freq: str = "M",
    cost_bps: float = 10.0,
    ledger_path: Path,
    holdout_consumed: bool,
) -> BacktestReport:
    """单因子回测 → 记账 → DSR → 报告卡。"""
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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 把 cli.py 改用 runner（删除 Plan 2a 私有 `_backtest_report`）**

在 `quant/cli.py`：
1. 删除整个 `def _backtest_report(...)` 函数定义。
2. 删除其专属 import（若仅它用）：`from scipy.stats import kurtosis, skew`、`from quant.backtest.metrics import compute_metrics, sharpe`、`from quant.report.backtest_card import BacktestReport`、`from quant.validate.dsr import deflated_sharpe, expected_max_sharpe`、`from quant.validate.ledger import Ledger`。
3. 加 import：`from quant.report.runner import run_backtest_report`。
4. 把 `backtest_cmd` 与 `holdout_cmd` 中 `report = _backtest_report(name, params, factor, close, quantiles, side, freq, cost_bps, ledger_path, state_path, holdout_consumed=...)` 改为：
```python
    report = run_backtest_report(
        name, params, factor, close,
        quantiles=quantiles, side=side, freq=freq, cost_bps=cost_bps,
        ledger_path=ledger_path, holdout_consumed=...,
    )
```
（`...` 保留原 holdout_consumed 实参：`backtest_cmd` 用 `is_consumed(name, state_path)`，`holdout_cmd` 用 `True`。`run_backtest` import 仍由 `combine_cmd` 使用，勿删。）

- [ ] **Step 6: 跑回归确认通过**

Run: `uv run pytest tests/test_cli_backtest.py tests/test_cli_holdout.py -v`
Expected: PASS（4 passed，行为不变）

- [ ] **Step 7: Commit**

```bash
git add quant/report/runner.py tests/test_runner.py quant/cli.py
git commit -m "refactor: extract run_backtest_report service (CLI+web shared)"
```

---

## Task 10: CLI 接入新因子 + 行业中性化

`_make_factor` 改注册表分发支持全部因子（含 volume）；`factor_test` 复用 `_make_factor`；两命令加 `--neutralize`。

**Files:**
- Modify: `quant/cli.py`
- Create: `tests/test_cli_extra.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cli_extra.py`:
```python
from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_factor_test_new_factor(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    # rel_volume 需 volume；走注册表分发应跑通
    result = runner.invoke(
        app,
        ["factor", "test", "rel_volume", "--window", "10",
         "--quantiles", "3", "--mode", "full"],
    )
    assert result.exit_code == 0, result.output
    assert "因子体检报告" in result.output


def test_backtest_with_neutralize(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["backtest", "volatility", "--window", "10", "--quantiles", "3",
         "--freq", "M", "--mode", "full", "--neutralize",
         "--ledger-path", str(tmp_path / "ledger.jsonl"),
         "--state-path", str(tmp_path / "state.json")],
    )
    assert result.exit_code == 0, result.output
    assert "回测体检报告" in result.output
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli_extra.py -v`
Expected: FAIL（`factor test rel_volume` 未知因子 / `--neutralize` 无此选项）

- [ ] **Step 3: 改 cli.py — 注册表分发 + volume + 中性化**

在 import 区追加：
```python
from quant.data.sectors import load_sectors
from quant.factor.registry import compute_factor, factor_names, needs_volume
from quant.process.neutralize import sector_neutralize
```

把 `_FACTORS = {"momentum": Momentum, "ma_bias": MABias}` 改为：
```python
_FACTORS = set(factor_names())
```
（`_FACTORS` 仅用于"是否已知因子"判断；`x in _FACTORS` 语义不变。涉及 `list(_FACTORS)` 的报错信息仍可用。`Momentum`/`MABias` 的直接 import 若不再被其它处引用可删，否则保留。）

把 Plan 2a 的 `_factor_params` / `_make_factor`（或 Plan 2a 内联的因子构建）替换为下列统一版本（若 Plan 2a 未单独抽 `_factor_params`，新增之）：
```python
def _factor_params(name: str, lookback: int, skip: int, window: int) -> dict:
    """按因子名挑出其构造参数。"""
    if name == "momentum":
        return {"lookback": lookback, "skip": skip}
    return {"window": window}  # ma_bias 及其余单窗因子


def _make_factor(
    name: str,
    close,
    lookback: int,
    skip: int,
    window: int,
    *,
    volume=None,
    neutralize: bool = False,
    market: str = "us",
):
    """注册表分发构因子；需要 volume 自动传；可选行业中性化。返回 (因子, 参数)。"""
    params = _factor_params(name, lookback, skip, window)
    factor = compute_factor(name, close, volume=volume, **params)
    if neutralize:
        factor = sector_neutralize(factor, load_sectors(market=market))
    return factor, params
```

改 `factor_test`（Plan 1 命令）：加 `--neutralize` 选项与 volume 加载，用 `_make_factor` 取代内联 if/else。把签名补一行选项、命令体前段改为：
```python
@factor_app.command("test")
def factor_test(
    name: str = typer.Argument(..., help="因子名"),
    lookback: int = typer.Option(252, help="动量回看窗口"),
    skip: int = typer.Option(21, help="动量跳过窗口"),
    window: int = typer.Option(200, help="单窗因子窗口"),
    horizon: int = typer.Option(21, help="前瞻收益天数"),
    quantiles: int = typer.Option(5, help="分位档数"),
    mode: str = typer.Option("research", help="research / holdout / full"),
    holdout_years: int = typer.Option(2, help="holdout 锁定年数"),
    neutralize: bool = typer.Option(False, "--neutralize", help="行业中性化"),
) -> None:
    if name not in _FACTORS:
        typer.echo(f"未知因子：{name}，可选 {sorted(_FACTORS)}", err=True)
        raise typer.Exit(code=1)

    close = load_price_matrix(field="close", market="us")
    close = apply_holdout(close, mode=mode, holdout_years=holdout_years)
    volume = load_price_matrix(field="volume", market="us")
    volume = apply_holdout(volume, mode=mode, holdout_years=holdout_years)

    factor, params = _make_factor(
        name, close, lookback, skip, window, volume=volume, neutralize=neutralize
    )
    factor = Pipeline([winsorize, zscore])(factor)
    fwd = forward_returns(close, horizon=horizon)
    # —— 以下 IC/分位/报告卡组装保持 Plan 1 原样 ——
    ...
```
（`...` 代表保留 Plan 1 `factor_test` 自 `ic = ic_series(...)` 起到 `typer.echo(report.to_markdown())` 的原有代码不变。）

改 `backtest_cmd`（Plan 2a 命令）：在签名末尾加 `neutralize: bool = typer.Option(False, "--neutralize", help="行业中性化")`；在 `close = apply_holdout(...)` 之后加载 volume 并把因子构建改为带 volume + neutralize：
```python
    volume = load_price_matrix(field="volume", market="us")
    volume = apply_holdout(volume, mode=mode, holdout_years=holdout_years)
    factor, params = _make_factor(
        name, close, lookback, skip, window, volume=volume, neutralize=neutralize
    )
    factor = Pipeline([winsorize, zscore])(factor)
```
（替换 Plan 2a `backtest_cmd` 里原 `factor, params = _make_factor(name, close, lookback, skip, window)` + Pipeline 两行。`holdout_cmd` 可同样加 volume + neutralize；本计划测试只覆盖 `backtest_cmd`，`holdout_cmd` 改动同形，按需施。）

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli_extra.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 全量回归 + lint**

Run:
```bash
uv run pytest -q
uv run ruff check .
```
Expected: 全部 PASS（含 Plan 1/2a 既有测试），ruff 无错误。

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py tests/test_cli_extra.py
git commit -m "feat: registry-backed CLI factors + sector neutralize flag"
```

---

## Task 11: Web 视图模型（四页纯逻辑）

四页的计算逻辑抽成纯函数，复用 Plan 1/2a 接口，不 import streamlit → 可单测。

**Files:**
- Create: `quant/web/__init__.py`
- Create: `quant/web/viewmodel.py`
- Create: `tests/test_viewmodel.py`

- [ ] **Step 1: 写失败测试**

`tests/test_viewmodel.py`:
```python
import pandas as pd

from quant.web import viewmodel


def test_available_factors_nonempty():
    assert "momentum" in viewmodel.available_factors()


def test_workshop_returns_markdown(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    md = viewmodel.workshop(
        "momentum", lookback=20, skip=1, window=10,
        horizon=5, quantiles=3, mode="full", neutralize=False,
    )
    assert "因子体检报告" in md
    assert "momentum" in md


def test_combine_returns_payload(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    out = viewmodel.combine(
        ["momentum", "ma_bias"], weighting="equal",
        lookback=20, skip=1, window=10, horizon=5,
        quantiles=3, side="long", freq="M", cost_bps=10.0, mode="full",
    )
    assert set(out["weights"]) == {"momentum", "ma_bias"}
    assert "annual_return" in out["metrics"]


def test_history_lists_trials(tmp_path):
    from quant.validate.ledger import Ledger
    lp = tmp_path / "ledger.jsonl"
    Ledger(lp).record({"factor": "momentum", "sharpe": 1.2})
    rows = viewmodel.history(lp)
    assert len(rows) == 1
    assert rows[0]["factor"] == "momentum"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_viewmodel.py -v`
Expected: FAIL（ModuleNotFoundError: quant.web）

- [ ] **Step 3: 实现 viewmodel**

`quant/web/__init__.py`:
```python
```

`quant/web/viewmodel.py`:
```python
"""Web 四页的纯逻辑层：复用 Plan 1/2a 接口，返回数据/Markdown，不依赖 streamlit。"""

from pathlib import Path

from quant.backtest.engine import backtest as run_backtest
from quant.backtest.metrics import compute_metrics
from quant.combine.synth import (
    combine_score,
    equal_weight,
    factor_correlation,
    high_correlation_warnings,
    ic_weight,
    zscore_factors,
)
from quant.data.holdout import apply_holdout
from quant.data.panel import load_price_matrix
from quant.data.returns import forward_returns
from quant.data.sectors import load_sectors
from quant.eval.ic import ic_series, ic_summary
from quant.eval.quantiles import long_short_spread, quantile_returns, turnover
from quant.factor.registry import compute_factor, factor_names
from quant.process.neutralize import sector_neutralize
from quant.process.pipeline import Pipeline, winsorize, zscore
from quant.report.runner import run_backtest_report
from quant.report.scorecard import FactorReport
from quant.validate.gate import assert_not_consumed, mark_consumed
from quant.validate.ledger import Ledger

_TRADING_DAYS_YEAR = 252


def available_factors() -> list[str]:
    return factor_names()


def _factor_params(name: str, lookback: int, skip: int, window: int) -> dict:
    if name == "momentum":
        return {"lookback": lookback, "skip": skip}
    return {"window": window}


def _load_panels(mode: str, holdout_years: int, market: str = "us"):
    close = apply_holdout(load_price_matrix(field="close", market=market), mode, holdout_years)
    volume = apply_holdout(load_price_matrix(field="volume", market=market), mode, holdout_years)
    return close, volume


def _build(name, close, volume, lookback, skip, window, neutralize, market="us"):
    params = _factor_params(name, lookback, skip, window)
    factor = compute_factor(name, close, volume=volume, **params)
    if neutralize:
        factor = sector_neutralize(factor, load_sectors(market=market))
    return Pipeline([winsorize, zscore])(factor), params


def workshop(
    name: str, lookback: int = 252, skip: int = 21, window: int = 200,
    horizon: int = 21, quantiles: int = 5, mode: str = "research",
    neutralize: bool = False, holdout_years: int = 2,
) -> str:
    """因子工坊：单因子端到端体检，返回报告卡 Markdown。"""
    close, volume = _load_panels(mode, holdout_years)
    factor, params = _build(name, close, volume, lookback, skip, window, neutralize)
    fwd = forward_returns(close, horizon=horizon)
    summary = ic_summary(ic_series(factor, fwd, method="spearman"))
    qr = quantile_returns(factor, fwd, n=quantiles)
    q_means = qr.mean().tolist()
    spread = long_short_spread(factor, fwd, n=quantiles)
    avg_turnover = turnover(factor, n=quantiles).mean()
    report = FactorReport(
        factor_name=name, params=params,
        ic_mean=summary["ic_mean"], ic_ir=summary["ic_ir"],
        t_stat=summary["t_stat"], n=summary["n"],
        quantile_means=q_means,
        long_short_annual=spread.mean() * (_TRADING_DAYS_YEAR / horizon),
        monotonic=all(a <= b for a, b in zip(q_means, q_means[1:]) if a == a and b == b),
        avg_turnover=float(avg_turnover) if avg_turnover == avg_turnover else 0.0,
        holdout_consumed=(mode == "holdout"),
    )
    return report.to_markdown()


def combine(
    names: list[str], weighting: str = "equal",
    lookback: int = 252, skip: int = 21, window: int = 200, horizon: int = 21,
    quantiles: int = 5, side: str = "long", freq: str = "M", cost_bps: float = 10.0,
    mode: str = "research", neutralize: bool = False, holdout_years: int = 2,
) -> dict:
    """多因子合成：返回权重、共线性预警、合成回测绩效。"""
    close, volume = _load_panels(mode, holdout_years)
    raw = {
        nm: _build(nm, close, volume, lookback, skip, window, neutralize)[0]
        for nm in names
    }
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
    m = compute_metrics(res.nav, res.returns)
    return {
        "weights": weights,
        "warnings": warns,
        "metrics": {
            "annual_return": m.annual_return,
            "sharpe": m.sharpe,
            "max_drawdown": m.max_drawdown,
            "monthly_win_rate": m.monthly_win_rate,
        },
        "nav": res.nav,
    }


def holdout_run(
    name: str, lookback: int = 252, skip: int = 21, window: int = 200,
    quantiles: int = 5, side: str = "long", freq: str = "M", cost_bps: float = 10.0,
    neutralize: bool = False, holdout_years: int = 2,
    ledger_path: Path = Path("quant_out/ledger.jsonl"),
    state_path: Path = Path("quant_out/holdout_state.json"),
) -> str:
    """holdout 闸门：仅一次在 holdout 跑最终回测并标记消耗，返回报告卡 Markdown。"""
    assert_not_consumed(name, state_path)
    close, volume = _load_panels("holdout", holdout_years)
    factor, params = _build(name, close, volume, lookback, skip, window, neutralize)
    report = run_backtest_report(
        name, params, factor, close,
        quantiles=quantiles, side=side, freq=freq, cost_bps=cost_bps,
        ledger_path=ledger_path, holdout_consumed=True,
    )
    mark_consumed(name, state_path)
    return report.to_markdown()


def history(ledger_path: Path) -> list[dict]:
    """历史：列过往试验台账记录。"""
    return Ledger(ledger_path).entries()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_viewmodel.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/web/__init__.py quant/web/viewmodel.py tests/test_viewmodel.py
git commit -m "feat: web viewmodel (workshop/combine/holdout/history)"
```

---

## Task 12: Streamlit Web 四页

薄 Streamlit 壳，四页分别调 viewmodel。加 `[web]` 可选依赖。

**Files:**
- Modify: `pyproject.toml`
- Create: `quant/web/app.py`

- [ ] **Step 1: 加 streamlit 可选依赖**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 区追加 `web` 组（与既有 `dev` 并列）：
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
]
web = [
    "streamlit>=1.30.0",
]
```

- [ ] **Step 2: 装 web 依赖**

Run:
```bash
uv pip install -e ".[dev,web]"
```
Expected: 安装 streamlit 成功。

- [ ] **Step 3: 实现 Streamlit app**

`quant/web/app.py`:
```python
"""Quant 因子研究 Web —— 四页：因子工坊 / 多因子合成 / holdout 闸门 / 历史。

启动：uv run streamlit run quant/web/app.py
"""

from pathlib import Path

import streamlit as st

from quant.web import viewmodel

st.set_page_config(page_title="Quant 因子研究", layout="wide")
PAGES = ["因子工坊", "多因子合成", "holdout 闸门", "历史"]
page = st.sidebar.radio("页面", PAGES)
mode = st.sidebar.selectbox("数据模式", ["research", "full"], index=0)

if page == "因子工坊":
    st.header("因子工坊")
    name = st.selectbox("因子", viewmodel.available_factors())
    c1, c2, c3 = st.columns(3)
    lookback = c1.number_input("lookback", 1, 1000, 252)
    skip = c2.number_input("skip", 0, 250, 21)
    window = c3.number_input("window", 1, 1000, 200)
    horizon = c1.number_input("horizon", 1, 120, 21)
    quantiles = c2.number_input("分位档数", 2, 10, 5)
    neutralize = c3.checkbox("行业中性化")
    if st.button("跑检验"):
        md = viewmodel.workshop(
            name, lookback=lookback, skip=skip, window=window,
            horizon=horizon, quantiles=quantiles, mode=mode, neutralize=neutralize,
        )
        st.markdown(md)

elif page == "多因子合成":
    st.header("多因子合成")
    names = st.multiselect("因子（多选）", viewmodel.available_factors(),
                           default=["momentum", "ma_bias"])
    weighting = st.radio("加权法", ["equal", "ic"], horizontal=True)
    quantiles = st.number_input("分位档数", 2, 10, 5)
    if st.button("合成回测") and names:
        out = viewmodel.combine(names, weighting=weighting, quantiles=quantiles, mode=mode)
        st.subheader("权重")
        st.json(out["weights"])
        st.subheader("共线性预警（|相关|≥0.7）")
        st.write(out["warnings"] or "无")
        st.subheader("绩效")
        st.json(out["metrics"])
        st.line_chart(out["nav"])

elif page == "holdout 闸门":
    st.header("holdout 闸门")
    st.warning("定稿因子仅可在 holdout 跑一次，跑后锁定。")
    name = st.selectbox("定稿因子", viewmodel.available_factors())
    confirm = st.checkbox("我确认这是最终定稿，仅跑一次")
    if st.button("在 holdout 跑最终验证") and confirm:
        try:
            md = viewmodel.holdout_run(name)
            st.markdown(md)
            st.success("holdout 已消耗，因子已锁定。")
        except RuntimeError as e:
            st.error(str(e))

else:  # 历史
    st.header("历史试验台账")
    rows = viewmodel.history(Path("quant_out/ledger.jsonl"))
    if rows:
        st.dataframe(rows)
    else:
        st.info("暂无记录。先在因子工坊或回测里跑一次。")
```

- [ ] **Step 4: 手动冒烟（Streamlit 难自动化单测）**

Run:
```bash
uv run streamlit run quant/web/app.py --server.headless true &
sleep 5
curl -sf http://localhost:8501/_stcore/health
```
Expected: 返回 `ok`，无启动异常。验证后停掉进程（`kill %1`）。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml quant/web/app.py
git commit -m "feat: streamlit web (4 pages)"
```

---

## Task 13: 真实数据 + 缓存冒烟验证

确认新因子、中性化、合成、web 对真实 data_lake 跑通（手动，非自动化测试）。

**Files:** 无（手动验证）

- [ ] **Step 1: 新因子单因子体检**

Run:
```bash
uv run quant factor test volatility --mode research
uv run quant factor test amihud --mode research
uv run quant factor test rel_volume --mode research --neutralize
```
Expected: 各输出"因子体检报告"，IC/分位/换手有数值；`--neutralize` 跑通无异常。

- [ ] **Step 2: 新因子回测**

Run:
```bash
uv run quant backtest short_reversal --mode research --side long_short
```
Expected: 输出"回测体检报告"，绩效 + DSR 有数值。

- [ ] **Step 3: 多因子合成（含新因子）**

Run:
```bash
uv run quant combine momentum volatility amihud --weighting ic --mode research
```
Expected: 输出权重、共线性预警、合成绩效。

- [ ] **Step 4: Web 端到端**

Run:
```bash
uv run streamlit run quant/web/app.py
```
手动：① 因子工坊跑 volatility（勾中性化）→ 出报告卡；② 合成页选 3 因子 IC 加权 → 出权重/预警/净值图；③ 历史页看到台账记录。
Expected: 四页可点、无异常。

- [ ] **Step 5: 记录基线耗时**

Run:
```bash
time uv run quant backtest momentum --mode research
```
Expected: 记录耗时；对照 Plan 2a 基线。若 IO 重，可在 `panel/returns` 接 `cache.load_or_compute`（接口已就绪，按需接线，不在本计划强做）。

- [ ] **Step 6: 无需 commit（纯验证）**

---

## Self-Review 记录

**Spec 覆盖**（对照 spec §4.2 因子族 + §5.1 缓存 + §6.5 中性化 + §9.2 web 四页）：

| Spec 要求 | 实现位置 |
|---|---|
| §4.2 反转族（1 月反转） | Task 1 `ShortReversal` |
| §4.2 波动族（已实现波动率） | Task 2 `Volatility` |
| §4.2 量能/流动性（相对量能） | Task 3 `RelativeVolume` |
| §4.2 流动性（Amihud 非流动性） | Task 4 `Amihud` |
| §4.2 因子统一管理 | Task 5 `registry` |
| §4.2 其余更冷门族（趋势R²/ADX/Beta/偏度/规模代理…） | 明确不做（v2，按同一 Factor 模式自行追加） |
| §6.5 行业中性化（GICS 去均值，可选） | Task 6 `load_sectors` + Task 7 `sector_neutralize` + Task 10 `--neutralize` |
| §5.1 缓存（前瞻收益/因子矩阵 parquet） | Task 8 `cache/store.py`（接口就绪，Task 13 Step 5 说明接线点） |
| §9.2 web 因子工坊 | Task 11 `workshop` + Task 12 页1 |
| §9.2 web 多因子合成 | Task 11 `combine` + Task 12 页2 |
| §9.2 web holdout 闸门（确认弹窗 + 标记消耗） | Task 11 `holdout_run` + Task 12 页3 |
| §9.2 web 历史（列报告/台账） | Task 11 `history` + Task 12 页4 |

**类型一致性核对：**
- 新因子均继承 `Factor`，`compute(self, close, volume=None) -> DataFrame`，与 Plan 1 基类签名一致。
- `registry.compute_factor(name, close, volume=None, **params)` 在 Task 5 定义；Task 10 `_make_factor`、Task 11 `_build` 调用参数名一致。
- `load_sectors(market, root) -> Series` 在 Task 6 定义；Task 7/10/11 调用一致（Task 7 `sector_neutralize(factor, sectors)` 签名匹配）。
- `run_backtest_report(name, params, factor, close, *, quantiles, side, freq, cost_bps, ledger_path, holdout_consumed)` 在 Task 9 定义；Task 9 Step 5 改 CLI、Task 11 `holdout_run` 调用一致（关键字参数全对应）。
- `load_or_compute(key, compute, cache_dir)` / `cache_key(*parts)` 在 Task 8 定义。
- viewmodel `workshop/combine/holdout_run/history` 在 Task 11 定义；Task 12 app.py 调用参数名一致。
- `_factor_params` 在 Task 10（CLI）与 Task 11（viewmodel）各有一份同义实现——两层独立，可接受的薄重复；若后续合并，提取到 `factor/registry.py` 即可。

**Placeholder 扫描：** Task 10 Step 3 与 Task 11 中以 `...` 标注的"保留 Plan 1 原有代码"为对既有文件的增量编辑指引，非新写代码占位；其余步骤均含完整可运行代码。

**前置假设：** Plan 1（已完成）+ Plan 2a（须先执行）。Task 9 依赖 Plan 2a 的 `cli.py` 内 `_backtest_report`/`backtest_cmd`/`holdout_cmd`；Task 10 依赖 Plan 2a 的 `_make_factor`/`_FACTORS`/`backtest_cmd`。若 Plan 2a 实现与计划接口漂移，执行 Task 9/10 前先对齐。
