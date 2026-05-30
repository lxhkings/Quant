"""Quant CLI。`quant factor test <name>` 跑单因子端到端体检。"""

import pandas as pd
import typer

from quant.data.holdout import apply_holdout
from quant.data.panel import load_price_matrix
from quant.data.returns import forward_returns
from quant.eval.ic import ic_series, ic_summary
from quant.eval.quantiles import long_short_spread, quantile_returns, turnover
from quant.factor.library.ma_bias import MABias
from quant.factor.library.momentum import Momentum
from quant.process.pipeline import Pipeline, winsorize, zscore
from quant.report.scorecard import FactorReport

app = typer.Typer(help="Quant 因子研究")
factor_app = typer.Typer(help="因子检验")
app.add_typer(factor_app, name="factor")

_FACTORS = {"momentum": Momentum, "ma_bias": MABias}
_TRADING_DAYS_YEAR = 252


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
