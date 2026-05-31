"""短期反转因子：过去 window 日涨幅取负 = -(close/close.shift(window) - 1)。

超跌反弹——近期跌得多的未来反弹概率高，故取负使"跌多=高分"。
"""

import pandas as pd

from quant.factor.base import Factor


class ShortReversal(Factor):
    name = "short_reversal"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        past_return = close / close.shift(self.window) - 1.0
        return -past_return
