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
from quant.report.leaderboard import scan_factors
from quant.report.runner import run_backtest_report
from quant.report.scorecard import FactorReport
from quant.select.screen import screen
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
