"""已实现波动率因子：过去 window 日日收益的标准差。

低波异象——低波动标的长期风险调整后收益更优（信号方向由检验环节判定）。
"""

import pandas as pd

from quant.factor.base import Factor


class Volatility(Factor):
    name = "volatility"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        daily_ret = close.pct_change()
        return daily_ret.rolling(self.window).std()
