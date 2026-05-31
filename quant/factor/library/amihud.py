"""Amihud 非流动性因子：过去 window 日 |日收益| / 成交额 的均值。

成交额 = close × volume。值越大越不流动 → 流动性溢价（信号方向由检验判定）。
"""

import pandas as pd

from quant.factor.base import Factor


class Amihud(Factor):
    name = "amihud"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        if volume is None:
            raise ValueError("amihud 因子需要 volume 输入")
        daily_ret = close.pct_change().abs()
        dollar_vol = close * volume
        illiq = daily_ret / dollar_vol
        return illiq.rolling(self.window).mean()
