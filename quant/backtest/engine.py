"""回测引擎：分位选股 → 构权重 → 滚动净值 → 扣成本。

约定（防前视）：
- 第 t 个调仓日用截至 t 的因子值构目标权重；
- 该权重经 shift(1) 应用到 t+1 起的日度收益；
- 换手成本在调仓日 t 当天扣减。
"""

from __future__ import annotations

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
