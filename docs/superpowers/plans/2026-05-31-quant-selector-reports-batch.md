# Quant 选股器 + 报告通俗化 + 批量体检 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有因子引擎上加三块：多因子量化选股器（最新截面打分选股，价量因子滑块）、报告通俗化（白话结论段）、批量因子体检（排行榜 + 独立 scan 台账不污染 DSR）。

**Architecture:** 选股器核心 `select/screen.py` 与批量核心 `report/leaderboard.py` 均为纯逻辑，全程走 `factor.registry`，不硬编码因子名（将来基本面因子注册即自动复用），不依赖 web。报告通俗化只改两张报告卡。web 页（选股器、排行榜）作为薄壳调用核心，依赖 Plan 2b 的 `quant/web/` 脚手架。

**Tech Stack:** Python 3.11、pandas、numpy、typer、pytest；web 用 streamlit + altair（altair 随 streamlit 安装）。

**设计文档：** `docs/superpowers/specs/2026-05-31-quant-selector-reports-batch-design.md`

---

## 前置依赖

- **Phase A（Task 1–5）：现在即可执行。** 依赖均为已完成模块：`factor.registry`、`combine.synth`、`data.{panel,holdout,sectors,returns}`、`eval.{ic,quantiles}`、`process.{pipeline,neutralize}`、`validate.ledger`、`report.{scorecard,backtest_card}`。
- **Phase B（Task 6–7）：依赖 Plan 2b web 层（Task 11–13）先完成**——需 `quant/web/__init__.py`、`quant/web/viewmodel.py`、`quant/web/app.py` 已存在。执行 Task 6/7 前确认这三个文件在位；若 Plan 2b 的 viewmodel/app 结构与本计划引用的结构（PAGES 列表 + `st.sidebar.radio` 分页 + `viewmodel.available_factors()`）有漂移，按实际结构对齐插入点。

已核实的接口签名（本计划据此编写）：

| 接口 | 签名 |
|---|---|
| `data.panel.load_price_matrix` | `(field="close", market="us", root=None) -> DataFrame` |
| `data.holdout.apply_holdout` | `(matrix, mode="research", holdout_years=2) -> DataFrame` |
| `data.returns.forward_returns` | `(close, horizon=1) -> DataFrame` |
| `data.sectors.load_sectors` | `(market="us", root=None) -> Series`（index=instrument_id，值=sector） |
| `factor.registry.compute_factor` | `(name, close, volume=None, **params) -> DataFrame` |
| `factor.registry.factor_names` | `() -> list[str]`（现 6 个价量因子） |
| `combine.synth.zscore_factors` | `(dict[str,DataFrame]) -> dict[str,DataFrame]` |
| `combine.synth.combine_score` | `(factors:dict, weights:dict) -> DataFrame` |
| `process.neutralize.sector_neutralize` | `(factor, sectors:Series) -> DataFrame` |
| `process.pipeline` | `winsorize`、`zscore`、`Pipeline([...])` |
| `eval.ic.ic_series/ic_summary` | `ic_series(factor,fwd,method) -> Series`；`ic_summary(Series) -> {ic_mean,ic_ir,t_stat,n,ic_std}` |
| `eval.quantiles.long_short_spread` | `(factor,fwd,n=5) -> Series` |
| `validate.ledger.Ledger` | `Ledger(path)`，`.record(dict)`、`.entries()`、`.count()` |

测试 fixture `fake_lake`（`tests/conftest.py`）：返回 `(root, [AAA,BBB,CCC], days[60])`；价格 AAA 每日 +0.5%（强动量）、BBB −0.3%、CCC 横盘；sectors：AAA/BBB=Tech、CCC=Energy。测试用 `monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))` 指向合成 lake（与既有测试一致）。

---

## File Structure

| 文件 | 职责 | 阶段 |
|---|---|---|
| `quant/report/scorecard.py`（改） | `FactorReport` 加 `_plain_verdict()` + 通俗结论段 | A |
| `quant/report/backtest_card.py`（改） | `BacktestReport` 加 `_plain_verdict()` + 通俗结论段 | A |
| `quant/select/__init__.py`（新） | 包标记 | A |
| `quant/select/screen.py`（新） | `SelectionResult` + `screen()`：最新截面多因子打分选股 | A |
| `quant/report/leaderboard.py`（新） | `scan_factors()`：批量因子体检排行榜 + 独立 scan 台账 | A |
| `quant/cli.py`（改） | 加 `factor test-all` 命令 | A |
| `quant/web/viewmodel.py`（改） | 加 `selector()`、`leaderboard()` 纯函数 | B |
| `quant/web/app.py`（改） | 加"选股器""批量排行榜"两页 | B |
| `tests/test_scorecard.py`（改） | 通俗结论断言 | A |
| `tests/test_backtest_card.py`（改） | 通俗结论断言 | A |
| `tests/test_screen.py`（新） | 选股核心测试 | A |
| `tests/test_leaderboard.py`（新） | 批量核心测试 | A |
| `tests/test_cli_testall.py`（新） | CLI test-all 测试 | A |
| `tests/test_viewmodel_selector.py`（新） | viewmodel selector/leaderboard 测试 | B |

依赖方向：`web → {select, report.leaderboard} → {factor.registry, combine, data, process, eval, validate}`，无环。

---

# Phase A：纯逻辑（现在即可执行）

## Task 1: FactorReport 通俗结论

**Files:**
- Modify: `quant/report/scorecard.py`
- Test: `tests/test_scorecard.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_scorecard.py` 末尾追加（自包含构造 `FactorReport`，不依赖既有 helper）：
```python
from quant.report.scorecard import FactorReport


def _make_fr(ic_ir, mono, ls, turnover=0.15):
    return FactorReport(
        factor_name="momentum", params={"lookback": 252},
        ic_mean=0.04, ic_ir=ic_ir, t_stat=2.5, n=200,
        quantile_means=[0.001, 0.003, 0.005, 0.008, 0.012],
        long_short_annual=ls, monotonic=mono,
        avg_turnover=turnover, holdout_consumed=False,
    )


def test_plain_verdict_strong_says_candidate():
    md = _make_fr(ic_ir=0.6, mono=True, ls=0.02).to_markdown()
    assert "通俗结论" in md
    assert "可进入候选池" in md


def test_plain_verdict_weak_says_not_recommended():
    md = _make_fr(ic_ir=0.1, mono=False, ls=-0.01).to_markdown()
    assert "不建议使用" in md


def test_plain_verdict_high_turnover_warns_cost():
    md = _make_fr(ic_ir=0.6, mono=True, ls=0.02, turnover=0.30).to_markdown()
    assert "交易成本" in md


def test_glossary_present():
    md = _make_fr(ic_ir=0.6, mono=True, ls=0.02).to_markdown()
    assert "IC-IR=因子预测力" in md
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_scorecard.py -k "plain_verdict or glossary" -v`
Expected: FAIL（断言找不到"通俗结论"等文案）

- [ ] **Step 3: 实现 `_plain_verdict` 并插入 `to_markdown`**

在 `quant/report/scorecard.py` 的 `FactorReport` 类中，`to_markdown` 方法**之前**加方法：
```python
    def _plain_verdict(self) -> str:
        greens = sum([self.ic_ir >= 0.5, self.monotonic, self.long_short_annual > 0])
        if greens == 3:
            head = "三关全过：预测力稳、分位单调、多空为正，**可进入候选池**。"
        elif greens == 0:
            head = "三关全不过：预测力弱、分位不单调或多空为负，**不建议使用**。"
        else:
            head = f"三关过 {greens}/3，**信号偏弱，需谨慎**。"
        cost = "换手偏高，留意交易成本。" if self.avg_turnover > 0.20 else ""
        return head + cost
```

把 `to_markdown` 的返回串里，标题行之后、`## 红绿灯结论` 之前插入通俗结论段。即把：
```python
        return f"""# 因子体检报告：{self.factor_name}

参数：{params}

## 红绿灯结论
```
改为：
```python
        return f"""# 因子体检报告：{self.factor_name}

参数：{params}

## 通俗结论

{self._plain_verdict()}

> 名词：IC-IR=因子预测力的稳定程度（越高越稳）；多空年化=买高分组、卖低分组的年化收益。

## 红绿灯结论
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_scorecard.py -v`
Expected: PASS（含原有 + 新增用例）

- [ ] **Step 5: Commit**

```bash
git add quant/report/scorecard.py tests/test_scorecard.py
git commit -m "feat: plain-language verdict on factor scorecard"
```

---

## Task 2: BacktestReport 通俗结论

**Files:**
- Modify: `quant/report/backtest_card.py`
- Test: `tests/test_backtest_card.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_backtest_card.py` 末尾追加：
```python
from quant.report.backtest_card import BacktestReport


def _make_br(annual, sharpe, dsr):
    return BacktestReport(
        factor_name="momentum", params={"lookback": 252},
        annual_return=annual, sharpe=sharpe, max_drawdown=-0.1,
        calmar=1.0, monthly_win_rate=0.55, avg_turnover=0.15,
        deflated_sharpe=dsr, n_trials=3, holdout_consumed=False,
    )


def test_bt_verdict_strong_says_live_candidate():
    md = _make_br(annual=0.15, sharpe=1.5, dsr=0.99).to_markdown()
    assert "通俗结论" in md
    assert "可考虑实盘候选" in md


def test_bt_verdict_weak_says_not_recommended():
    md = _make_br(annual=-0.05, sharpe=0.2, dsr=0.3).to_markdown()
    assert "不建议使用" in md


def test_bt_glossary_present():
    md = _make_br(annual=0.15, sharpe=1.5, dsr=0.99).to_markdown()
    assert "DSR=" in md
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_backtest_card.py -k "verdict or glossary" -v`
Expected: FAIL

- [ ] **Step 3: 实现 `_plain_verdict` 并插入 `to_markdown`**

在 `quant/report/backtest_card.py` 的 `BacktestReport` 类中，`to_markdown` 之前加方法：
```python
    def _plain_verdict(self) -> str:
        greens = sum([
            self.annual_return > 0,
            self.sharpe >= 1.0,
            self.deflated_sharpe >= 0.95,
        ])
        if greens == 3:
            return "含成本仍赚钱、风险调整后稳健、抗过拟合达标，**可考虑实盘候选**。"
        if greens == 0:
            return "含成本不赚钱、风险调整差、抗过拟合不达标，**不建议使用**。"
        return f"三关过 {greens}/3，**证据不足，需继续验证**。"
```

把 `to_markdown` 返回串里标题之后、`## 红绿灯结论` 之前插入：
```python
        return f"""# 回测体检报告：{self.factor_name}

参数：{params}

## 通俗结论

{self._plain_verdict()}

> 名词：Sharpe=每承担一单位风险换来的收益；DSR=剔除多重检验运气后的真实 Sharpe（≥0.95 才算可信）。

## 红绿灯结论
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_backtest_card.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quant/report/backtest_card.py tests/test_backtest_card.py
git commit -m "feat: plain-language verdict on backtest report card"
```

---

## Task 3: 选股核心 `select/screen.py`

最新截面多因子加权打分 → 0-100 → 切买入/备选池。纯逻辑，走 registry，不依赖 web。

**Files:**
- Create: `quant/select/__init__.py`
- Create: `quant/select/screen.py`
- Test: `tests/test_screen.py`

- [ ] **Step 1: 写失败测试**

`tests/test_screen.py`:
```python
from quant.select.screen import screen


def test_screen_ranks_and_zones(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    res = screen(["momentum"], {"momentum": 1.0}, top_n=1,
                 lookback=20, skip=1, mode="full")
    top = res.table.iloc[0]
    # AAA 强动量 → 排第一、综合分 100、买入池
    assert top["instrument_id"] == "AAA"
    assert top["zone"] == "buy"
    assert abs(top["score"] - 100.0) < 1e-9
    # 其余进备选池
    assert (res.table["zone"].iloc[1:] == "candidate").all()
    # 行业标签 join 成功
    assert top["sector"] == "Tech"


def test_screen_normalizes_weights(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    res = screen(["momentum", "ma_bias"], {"momentum": 1.0, "ma_bias": 3.0},
                 top_n=2, lookback=20, skip=1, window=10, mode="full")
    assert abs(sum(res.weights.values()) - 1.0) < 1e-9
    assert abs(res.weights["ma_bias"] - 0.75) < 1e-9


def test_screen_zero_weights_raises(fake_lake, monkeypatch):
    import pytest
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    with pytest.raises(ValueError, match="权重"):
        screen(["momentum"], {"momentum": 0.0}, top_n=1,
               lookback=20, skip=1, mode="full")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_screen.py -v`
Expected: FAIL（ModuleNotFoundError: quant.select.screen）

- [ ] **Step 3: 实现 screen**

`quant/select/__init__.py`:
```python
```

`quant/select/screen.py`:
```python
"""多因子量化选股：最新截面加权打分 → 0-100 综合分 → 切买入/备选池。

纯逻辑，走 factor.registry 动态取因子（将来基本面因子注册即自动复用）。
选股用最新截面（生产用途，非研究回测），不写 trial 台账、不算 DSR。
"""

from dataclasses import dataclass

import pandas as pd

from quant.combine.synth import combine_score, zscore_factors
from quant.data.holdout import apply_holdout
from quant.data.panel import load_price_matrix
from quant.data.sectors import load_sectors
from quant.factor.registry import compute_factor
from quant.process.neutralize import sector_neutralize


@dataclass
class SelectionResult:
    as_of: pd.Timestamp          # 选股所用最新交易日
    weights: dict[str, float]    # 归一后的因子权重（sum=1）
    table: pd.DataFrame          # 列：instrument_id, score(0-100), rank, zone, sector


def _factor_params(name: str, lookback: int, skip: int, window: int) -> dict:
    """按因子名挑构造参数（动量用 lookback/skip，其余单窗因子用 window）。"""
    if name == "momentum":
        return {"lookback": lookback, "skip": skip}
    return {"window": window}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """权重归一到 sum=1。全为 0 抛 ValueError。"""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("权重之和为 0，无法归一")
    return {k: v / total for k, v in weights.items()}


def screen(
    names: list[str],
    weights: dict[str, float],
    *,
    top_n: int = 10,
    lookback: int = 252,
    skip: int = 21,
    window: int = 200,
    neutralize: bool = False,
    mode: str = "full",
    holdout_years: int = 2,
    market: str = "us",
) -> SelectionResult:
    """对 names 因子按 weights 加权打分，输出最新截面选股表。"""
    close = apply_holdout(load_price_matrix(field="close", market=market), mode, holdout_years)
    volume = apply_holdout(load_price_matrix(field="volume", market=market), mode, holdout_years)
    sectors = load_sectors(market=market)

    raw = {}
    for nm in names:
        f = compute_factor(nm, close, volume=volume, **_factor_params(nm, lookback, skip, window))
        if neutralize:
            f = sector_neutralize(f, sectors)
        raw[nm] = f

    zf = zscore_factors(raw)
    nw = _normalize_weights(weights)
    score = combine_score(zf, nw)

    as_of = score.dropna(how="all").index[-1]
    row = score.loc[as_of].dropna()
    pct = row.rank(pct=True) * 100.0
    ranked = pct.sort_values(ascending=False)

    table = pd.DataFrame({"instrument_id": ranked.index, "score": ranked.values})
    table["rank"] = range(1, len(table) + 1)
    table["zone"] = ["buy" if i < top_n else "candidate" for i in range(len(table))]
    table["sector"] = table["instrument_id"].map(sectors).fillna("未知")
    return SelectionResult(as_of=as_of, weights=nw, table=table.reset_index(drop=True))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_screen.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/select/__init__.py quant/select/screen.py tests/test_screen.py
git commit -m "feat: multi-factor stock selector (latest cross-section scoring)"
```

---

## Task 4: 批量体检核心 `report/leaderboard.py`

循环全因子跑单因子检验，出排行榜表，记独立 scan 台账（不碰 DSR 主台账）。

**Files:**
- Create: `quant/report/leaderboard.py`
- Test: `tests/test_leaderboard.py`

- [ ] **Step 1: 写失败测试**

`tests/test_leaderboard.py`:
```python
from quant.report.leaderboard import scan_factors
from quant.validate.ledger import Ledger


def test_scan_factors_ranks_subset(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    lp = tmp_path / "scan_ledger.jsonl"
    table = scan_factors(
        ["momentum", "short_reversal"],
        lookback=20, skip=1, window=10, horizon=5, quantiles=3,
        mode="full", scan_ledger_path=lp,
    )
    assert set(table["factor"]) == {"momentum", "short_reversal"}
    assert list(table.columns) == [
        "factor", "ic_mean", "ic_ir", "t_stat", "long_short_annual"
    ]
    # 独立 scan 台账写了 2 条；DSR 主台账不受影响
    assert Ledger(lp).count() == 2


def test_scan_factors_defaults_to_all(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    table = scan_factors(
        lookback=20, skip=1, window=10, horizon=5, quantiles=3, mode="full",
        scan_ledger_path=tmp_path / "s.jsonl",
    )
    # 默认跑全部 6 个已注册因子
    assert len(table) == 6
    assert "amihud" in set(table["factor"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_leaderboard.py -v`
Expected: FAIL（ModuleNotFoundError: quant.report.leaderboard）

- [ ] **Step 3: 实现 leaderboard**

`quant/report/leaderboard.py`:
```python
"""批量因子体检：循环全因子跑 IC/多空，出排行榜表。

写独立 scan 台账（与 DSR 主 trial 台账分离）——批量扫描不应撑大 trial 数
而惩罚后续定稿因子的 Deflated Sharpe。
"""

from pathlib import Path

import pandas as pd

from quant.data.holdout import apply_holdout
from quant.data.panel import load_price_matrix
from quant.data.returns import forward_returns
from quant.eval.ic import ic_series, ic_summary
from quant.eval.quantiles import long_short_spread
from quant.factor.registry import compute_factor, factor_names
from quant.process.pipeline import Pipeline, winsorize, zscore
from quant.validate.ledger import Ledger

_TRADING_DAYS_YEAR = 252


def _factor_params(name: str, lookback: int, skip: int, window: int) -> dict:
    if name == "momentum":
        return {"lookback": lookback, "skip": skip}
    return {"window": window}


def scan_factors(
    names: list[str] | None = None,
    *,
    lookback: int = 252,
    skip: int = 21,
    window: int = 200,
    horizon: int = 21,
    quantiles: int = 5,
    mode: str = "research",
    holdout_years: int = 2,
    market: str = "us",
    scan_ledger_path: Path | None = None,
) -> pd.DataFrame:
    """批量跑因子，返回按 IC-IR 降序的排行榜表；可选写 scan 台账。"""
    names = names or factor_names()
    close = apply_holdout(load_price_matrix(field="close", market=market), mode, holdout_years)
    volume = apply_holdout(load_price_matrix(field="volume", market=market), mode, holdout_years)
    fwd = forward_returns(close, horizon=horizon)
    ledger = Ledger(scan_ledger_path) if scan_ledger_path else None

    rows = []
    for nm in names:
        f = compute_factor(nm, close, volume=volume, **_factor_params(nm, lookback, skip, window))
        f = Pipeline([winsorize, zscore])(f)
        summary = ic_summary(ic_series(f, fwd, method="spearman"))
        ls = long_short_spread(f, fwd, n=quantiles).mean() * (_TRADING_DAYS_YEAR / horizon)
        rows.append({
            "factor": nm,
            "ic_mean": summary["ic_mean"],
            "ic_ir": summary["ic_ir"],
            "t_stat": summary["t_stat"],
            "long_short_annual": ls,
        })
        if ledger is not None:
            ledger.record({"factor": nm, "ic_ir": summary["ic_ir"], "scan": True})

    return pd.DataFrame(rows).sort_values("ic_ir", ascending=False, ignore_index=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_leaderboard.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add quant/report/leaderboard.py tests/test_leaderboard.py
git commit -m "feat: batch factor leaderboard with separate scan ledger"
```

---

## Task 5: CLI `quant factor test-all`

**Files:**
- Modify: `quant/cli.py`
- Test: `tests/test_cli_testall.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cli_testall.py`:
```python
from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_factor_test_all_runs(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(app, [
        "factor", "test-all",
        "--lookback", "20", "--skip", "1", "--window", "10",
        "--horizon", "5", "--quantiles", "3", "--mode", "full",
        "--scan-ledger-path", str(tmp_path / "scan.jsonl"),
    ])
    assert result.exit_code == 0, result.output
    assert "排行榜" in result.output
    assert "momentum" in result.output
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli_testall.py -v`
Expected: FAIL（No such command 'test-all'）

- [ ] **Step 3: 实现 CLI 命令**

在 `quant/cli.py`：确认顶部已 `from pathlib import Path`（`backtest_cmd` 的 `--ledger-path` 选项已用 Path，应已存在；若无则加）。

在 `factor_app` 已定义之后（任意现有 `@factor_app.command(...)` 附近），追加：
```python
@factor_app.command("test-all")
def factor_test_all(
    lookback: int = typer.Option(252, help="动量回看窗口"),
    skip: int = typer.Option(21, help="动量跳过窗口"),
    window: int = typer.Option(200, help="单窗因子窗口"),
    horizon: int = typer.Option(21, help="前瞻收益天数"),
    quantiles: int = typer.Option(5, help="分位档数"),
    mode: str = typer.Option("research", help="research / holdout / full"),
    holdout_years: int = typer.Option(2, help="holdout 锁定年数"),
    scan_ledger_path: Path = typer.Option(
        Path("quant_out/scan_ledger.jsonl"), help="批量扫描台账路径（独立于 DSR 主台账）"
    ),
) -> None:
    """批量体检全部已注册因子，输出按 IC-IR 降序的排行榜。"""
    from quant.report.leaderboard import scan_factors

    table = scan_factors(
        lookback=lookback, skip=skip, window=window, horizon=horizon,
        quantiles=quantiles, mode=mode, holdout_years=holdout_years,
        scan_ledger_path=scan_ledger_path,
    )
    typer.echo("# 因子批量排行榜（按 IC-IR 降序）\n")
    typer.echo(table.to_string(index=False))
```

（用 `to_string` 而非 `to_markdown`，避免引入 `tabulate` 额外依赖。）

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli_testall.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 全量回归 + lint**

Run:
```bash
uv run pytest -q
uv run ruff check .
```
Expected: 全部 PASS，ruff 无错误。

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py tests/test_cli_testall.py
git commit -m "feat: quant factor test-all batch leaderboard CLI"
```

---

# Phase B：Web 页（依赖 Plan 2b web 层完成）

> 执行前确认 `quant/web/__init__.py`、`quant/web/viewmodel.py`、`quant/web/app.py` 已由 Plan 2b 建好，且 `viewmodel.available_factors()` 存在、`app.py` 用 `PAGES` 列表 + `st.sidebar.radio("页面", PAGES)` 分页。

## Task 6: 选股器页

**Files:**
- Modify: `quant/web/viewmodel.py`
- Modify: `quant/web/app.py`
- Test: `tests/test_viewmodel_selector.py`

- [ ] **Step 1: 写失败测试**

`tests/test_viewmodel_selector.py`:
```python
from quant.web import viewmodel


def test_selector_returns_table(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    out = viewmodel.selector(
        ["momentum"], {"momentum": 1.0}, top_n=1,
        lookback=20, skip=1, mode="full",
    )
    assert out["table"].iloc[0]["instrument_id"] == "AAA"
    assert out["table"].iloc[0]["zone"] == "buy"
    assert "as_of" in out
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_viewmodel_selector.py -v`
Expected: FAIL（AttributeError: module 'quant.web.viewmodel' has no attribute 'selector'）

- [ ] **Step 3: 在 viewmodel 加 selector**

在 `quant/web/viewmodel.py` 顶部 import 区追加：
```python
from quant.select.screen import screen
```

在文件末尾追加：
```python
def selector(
    names: list[str], weights: dict[str, float], top_n: int = 10,
    lookback: int = 252, skip: int = 21, window: int = 200,
    neutralize: bool = False, mode: str = "full", holdout_years: int = 2,
) -> dict:
    """选股器：最新截面多因子打分选股，返回截面日期、归一权重、选股表。"""
    res = screen(
        names, weights, top_n=top_n, lookback=lookback, skip=skip,
        window=window, neutralize=neutralize, mode=mode, holdout_years=holdout_years,
    )
    return {"as_of": str(res.as_of.date()), "weights": res.weights, "table": res.table}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_viewmodel_selector.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 在 app.py 加"选股器"页**

在 `quant/web/app.py`：

1. 顶部 import 区加：
```python
import altair as alt
```

2. `PAGES = [...]` 列表追加 `"选股器"`（放末尾）。

3. 在分页 if/elif 链末尾（最后一个 `else`/分支之后或之前，作为新的 `elif`）加：
```python
elif page == "选股器":
    st.header("多因子量化选股器")
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
        st.caption(f"截面日期：{out['as_of']}　归一权重：{out['weights']}")
        t = out["table"]
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

（`mode` 变量沿用 Plan 2b app.py 侧栏的数据模式选择；若侧栏 `mode` 不含 `"full"`，选股建议用最新全量数据，可在该页单独加 `mode = "full"`。）

- [ ] **Step 6: 手动冒烟（streamlit 难自动化）**

Run:
```bash
uv run streamlit run quant/web/app.py --server.headless true &
sleep 5
curl -sf http://localhost:8501/_stcore/health
```
Expected: 返回 `ok`。手动点"选股器"页：选 momentum、设权重、top-N=3 → 出水平条形图（买入池绿/备选池蓝）+ 选股表。验毕 `kill %1`。

- [ ] **Step 7: Commit**

```bash
git add quant/web/viewmodel.py quant/web/app.py tests/test_viewmodel_selector.py
git commit -m "feat: web stock selector page (weight sliders + score bar chart)"
```

---

## Task 7: 批量排行榜页

**Files:**
- Modify: `quant/web/viewmodel.py`
- Modify: `quant/web/app.py`
- Test: `tests/test_viewmodel_selector.py`（追加）

- [ ] **Step 1: 追加失败测试**

在 `tests/test_viewmodel_selector.py` 末尾追加：
```python
def test_leaderboard_returns_ranked(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    df = viewmodel.leaderboard(
        ["momentum", "short_reversal"],
        lookback=20, skip=1, window=10, horizon=5, quantiles=3,
        mode="full", scan_ledger_path=tmp_path / "s.jsonl",
    )
    assert "ic_ir" in df.columns
    assert set(df["factor"]) == {"momentum", "short_reversal"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_viewmodel_selector.py::test_leaderboard_returns_ranked -v`
Expected: FAIL（no attribute 'leaderboard'）

- [ ] **Step 3: 在 viewmodel 加 leaderboard**

在 `quant/web/viewmodel.py` import 区追加：
```python
from quant.report.leaderboard import scan_factors
```

文件末尾追加：
```python
def leaderboard(
    names: list[str] | None = None,
    lookback: int = 252, skip: int = 21, window: int = 200, horizon: int = 21,
    quantiles: int = 5, mode: str = "research", holdout_years: int = 2,
    scan_ledger_path: Path = Path("quant_out/scan_ledger.jsonl"),
):
    """批量排行榜：透传 scan_factors，返回排行榜 DataFrame。"""
    return scan_factors(
        names, lookback=lookback, skip=skip, window=window, horizon=horizon,
        quantiles=quantiles, mode=mode, holdout_years=holdout_years,
        scan_ledger_path=scan_ledger_path,
    )
```

（`Path` 应已在 viewmodel.py 中 import；若无，在 import 区加 `from pathlib import Path`。）

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_viewmodel_selector.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 在 app.py 加"批量排行榜"页**

在 `quant/web/app.py`：

1. `PAGES` 列表追加 `"批量排行榜"`。

2. 分页链末尾加：
```python
elif page == "批量排行榜":
    st.header("因子批量排行榜")
    st.caption("批量扫描写独立 scan 台账，不影响 DSR 主台账。")
    if st.button("批量体检全部因子"):
        df = viewmodel.leaderboard(mode=mode)
        st.dataframe(df)
```

- [ ] **Step 6: 手动冒烟**

Run:
```bash
uv run streamlit run quant/web/app.py --server.headless true &
sleep 5
curl -sf http://localhost:8501/_stcore/health
```
手动点"批量排行榜"页 → 点按钮 → 出全因子 IC-IR 排序表。验毕 `kill %1`。

- [ ] **Step 7: 全量回归 + lint + Commit**

```bash
uv run pytest -q
uv run ruff check .
git add quant/web/viewmodel.py quant/web/app.py tests/test_viewmodel_selector.py
git commit -m "feat: web batch leaderboard page"
```

---

## Self-Review 记录

**Spec 覆盖：**

| Spec 要求 | 实现位置 |
|---|---|
| P1 选股核心（registry 动态、0-100、买入/备选池、归一权重、行业标签） | Task 3 `select/screen.py` |
| P1 选股器页（滑块、条形图、行业标签、★ 通过颜色区分池） | Task 6 `viewmodel.selector` + app.py 选股器页 |
| P1 模块化复用（基本面因子注册即自动出滑块） | Task 3 走 `factor_names()`/`compute_factor`，零硬编码 |
| P2 scorecard 通俗结论 | Task 1 |
| P2 backtest_card 通俗结论 | Task 2 |
| P2 保留原指标表 | Task 1/2：仅在标题后插入新段，原表不动 |
| P3 批量体检核心 + 独立 scan 台账 | Task 4 `scan_factors` |
| P3 CLI test-all | Task 5 |
| P3 web 排行榜页 | Task 7 |
| 选股不写 trial 台账/不算 DSR | Task 3 screen 不引 ledger/dsr |
| 范围外（基本面/估值质量/正交化/下单/看板） | 不含，符合 spec |

**Placeholder 扫描：** 无 TBD/TODO；所有改 code 的步骤含完整代码块。Phase B 中"插入点"指引（PAGES/elif）为对 P0 既有文件的增量编辑，代码完整给出，非占位。

**类型一致性核对：**
- `SelectionResult(as_of, weights, table)` 在 Task 3 定义；Task 6 `selector` 取 `res.as_of/res.weights/res.table`，字段名一致。
- `screen(names, weights, *, top_n, lookback, skip, window, neutralize, mode, holdout_years, market)` 在 Task 3 定义；Task 6 `selector` 调用关键字一致（market 用默认）。
- `scan_factors(names=None, *, lookback, skip, window, horizon, quantiles, mode, holdout_years, market, scan_ledger_path)` 在 Task 4 定义；Task 5 CLI 与 Task 7 `leaderboard` 调用关键字一致。
- `_factor_params(name, lookback, skip, window)` 在 Task 3 与 Task 4 各一份（两模块独立，薄重复可接受，与 Plan 2b viewmodel/cli 既有约定一致）。
- 报告卡 `_plain_verdict(self)` 在 Task 1/2 定义，仅 `to_markdown` 内部调用，无跨任务引用风险。
- Ledger 仅用 `record`（Task 4）与 `count`（测试），均匹配已核实接口。

**依赖与执行顺序：** Task 1–5 无 P0 依赖，可立即按序执行（Task 5 末尾跑全量回归）。Task 6–7 须待 Plan 2b web 层就位后执行。
</content>
