"""动量因子：过去 lookback 日涨幅，跳过最近 skip 日（避开短期反转）。"""

import pandas as pd

from quant.factor.base import Factor


class Momentum(Factor):
    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback = lookback
        self.skip = skip

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        # 第 t 日动量 = close[t-skip] / close[t-skip-lookback] - 1
        recent = close.shift(self.skip)
        past = close.shift(self.skip + self.lookback)
        return recent / past - 1.0
