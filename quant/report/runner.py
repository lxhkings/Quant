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
    if pd.isna(avg_turnover):  # NaN
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
