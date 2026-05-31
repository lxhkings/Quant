"""前瞻收益矩阵。"""

import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    第 t 日的前瞻收益 = close[t+horizon] / close[t] - 1。

    末尾 horizon 行为 NaN（无未来数据）。
    """
    return close.shift(-horizon) / close - 1.0


def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """日度简单收益 = close[t] / close[t-1] - 1。首行为 NaN（无前值）。"""
    return close.pct_change()
