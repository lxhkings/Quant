"""回测引擎：分位选股 → 构权重 → 滚动净值 → 扣成本。

约定（防前视）：
- 第 t 个调仓日用截至 t 的因子值构目标权重；
- 该权重经 shift(1) 应用到 t+1 起的日度收益；
- 换手成本在调仓日 t 当天扣减。
"""

from __future__ import annotations

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
