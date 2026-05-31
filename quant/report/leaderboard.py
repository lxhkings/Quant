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
