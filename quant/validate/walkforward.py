"""Walk-forward：滚动窗口样本内选参、样本外验证。

参数只在 IS 调，OOS 仅验证，量化过拟合（OOS/IS Sharpe 比）。
"""

from __future__ import annotations

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
