"""Holdout 时间轴物理隔离。

mode:
- "research"：只给研究期（去掉末尾 holdout_years 年），扫参/检验用
- "holdout"：只给锁定窗口（末尾 holdout_years 年），定稿后仅一次
- "full"：全量，不截断
"""

import pandas as pd
from pandas.tseries.offsets import DateOffset


def research_cutoff(index: pd.DatetimeIndex, holdout_years: int = 2) -> pd.Timestamp:
    """研究期最后一天 = 数据末日往前推 holdout_years 年。"""
    return index.max() - DateOffset(years=holdout_years)


def apply_holdout(
    matrix: pd.DataFrame,
    mode: str = "research",
    holdout_years: int = 2,
) -> pd.DataFrame:
    """按 mode 截断宽矩阵的时间轴。"""
    if mode == "full":
        return matrix
    if holdout_years <= 0:
        return matrix
    cutoff = research_cutoff(matrix.index, holdout_years)
    if mode == "research":
        return matrix.loc[matrix.index <= cutoff]
    if mode == "holdout":
        return matrix.loc[matrix.index > cutoff]
    raise ValueError(f"未知 mode: {mode!r}")
