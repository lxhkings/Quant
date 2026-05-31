"""相对量能因子：成交量相对过去 window 日均量的偏离 = volume / MA(volume) - 1。

放量异动捕捉资金动向。需 volume 输入。
"""

import pandas as pd

from quant.factor.base import Factor


class RelativeVolume(Factor):
    name = "rel_volume"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        if volume is None:
            raise ValueError("rel_volume 因子需要 volume 输入")
        ma = volume.rolling(self.window).mean()
        return volume / ma - 1.0
