"""均线乖离因子：收盘价相对 N 日均线的偏离 = close / MA(N) - 1。"""

import pandas as pd

from quant.factor.base import Factor


class MABias(Factor):
    name = "ma_bias"

    def __init__(self, window: int = 200):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        ma = close.rolling(self.window).mean()
        return close / ma - 1.0
