"""行业中性化：每个截面在各 GICS 行业内对因子值去均值。

剥掉行业整体偏移，留下行业内相对强弱。无行业映射的标的保持原值。
"""

import pandas as pd


def sector_neutralize(factor: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """逐截面（逐行）在行业内减去该行业均值。"""
    grp = sectors.reindex(factor.columns)
    out = factor.copy()
    for _, members in grp.groupby(grp):
        cols = list(members.index)
        sub = factor[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out
