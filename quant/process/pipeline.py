"""因子处理：横截面（逐行）去极值、标准化，及可组合处理链。"""

from collections.abc import Callable

import pandas as pd

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """逐行（每个截面）把超出 [lower, upper] 分位的值截断到分位边界。"""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """逐行标准化：(x - mean) / std，忽略 NaN。"""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    return df.sub(mean, axis=0).div(std, axis=0)


class Pipeline:
    """按顺序应用一串处理函数。"""

    def __init__(self, steps: list[Transform]):
        self.steps = steps

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step(df)
        return df
