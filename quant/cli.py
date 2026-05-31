"""Quant CLI。`quant factor test <name>` 跑单因子端到端体检。"""

import pandas as pd
import typer
from scipy.stats import kurtosis, skew

from quant.backtest.engine import backtest as run_backtest
from quant.backtest.metrics import compute_metrics, sharpe
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
from quant.eval.ic import ic_series, ic_summary
from quant.eval.quantiles import long_short_spread, quantile_returns, turnover
from quant.factor.library.ma_bias import MABias
from quant.factor.library.momentum import Momentum
from quant.process.pipeline import Pipeline, winsorize, zscore
from quant.report.backtest_card import BacktestReport
from quant.report.scorecard import FactorReport
from quant.validate.dsr import deflated_sharpe, expected_max_sharpe
from quant.validate.gate import assert_not_consumed, is_consumed, mark_consumed
from quant.validate.ledger import Ledger

app = typer.Typer(help="Quant 因子研究")
factor_app = typer.Typer(help="因子检验")
app.add_typer(factor_app, name="factor")

_FACTORS = {"momentum": Momentum, "ma_bias": MABias}
_TRADING_DAYS_YEAR = 252


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


@factor_app.command("test")
def factor_test(
    name: str = typer.Argument(..., help="因子名：momentum / ma_bias"),
    lookback: int = typer.Option(252, help="动量回看窗口"),
    skip: int = typer.Option(21, help="动量跳过窗口"),
    window: int = typer.Option(200, help="均线窗口（ma_bias）"),
    horizon: int = typer.Option(21, help="前瞻收益天数"),
    quantiles: int = typer.Option(5, help="分位档数"),
    mode: str = typer.Option("research", help="research / holdout / full"),
    holdout_years: int = typer.Option(2, help="holdout 锁定年数"),
) -> None:
    if name not in _FACTORS:
        typer.echo(f"未知因子：{name}，可选 {list(_FACTORS)}", err=True)
        raise typer.Exit(code=1)

    close = load_price_matrix(field="close", market="us")
    close = apply_holdout(close, mode=mode, holdout_years=holdout_years)

    if name == "momentum":
        factor = Momentum(lookback=lookback, skip=skip).compute(close)
        params = {"lookback": lookback, "skip": skip}
    else:
        factor = MABias(window=window).compute(close)
        params = {"window": window}

    factor = Pipeline([winsorize, zscore])(factor)
    fwd = forward_returns(close, horizon=horizon)

    ic = ic_series(factor, fwd, method="spearman")
    summary = ic_summary(ic)
    qr = quantile_returns(factor, fwd, n=quantiles)
    q_means = qr.mean().tolist()
    spread = long_short_spread(factor, fwd, n=quantiles)
    avg_turnover = turnover(factor, n=quantiles).mean()

    report = FactorReport(
        factor_name=name,
        params=params,
        ic_mean=summary["ic_mean"],
        ic_ir=summary["ic_ir"],
        t_stat=summary["t_stat"],
        n=summary["n"],
        quantile_means=q_means,
        long_short_annual=spread.mean() * (_TRADING_DAYS_YEAR / horizon),
        monotonic=_is_monotonic(q_means),
        avg_turnover=float(avg_turnover) if pd.notna(avg_turnover) else 0.0,
        holdout_consumed=(mode == "holdout"),
    )
    typer.echo(report.to_markdown())


def _is_monotonic(values: list[float]) -> bool:
    """分位平均收益是否单调递增（忽略 NaN）。"""
    clean = [v for v in values if v == v]
    return all(a <= b for a, b in zip(clean, clean[1:]))


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
