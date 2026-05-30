"""分位收益、多空价差、换手率。

每个截面按因子值分 n 档（1=最低，n=最高）：
- quantile_returns：各档前瞻收益时序（按截面等权平均）
- long_short_spread：顶档 − 底档
- turnover：顶档成员逐期变化比例（估算成本）
"""

import numpy as np
import pandas as pd


def _ranks(factor: pd.DataFrame, n: int) -> pd.DataFrame:
    """逐行把因子值分成 1..n 档；不足处为 NaN。"""

    def bucket(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < n:
            return pd.Series(np.nan, index=row.index)
        labels = pd.qcut(valid.rank(method="first"), n, labels=False, duplicates="drop") + 1
        return labels.reindex(row.index)

    return factor.apply(bucket, axis=1)


def quantile_returns(factor: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """返回 [date x quantile(1..n)] 的平均前瞻收益。"""
    f, r = factor.align(fwd_ret, join="inner")
    buckets = _ranks(f, n)
    rows = {}
    for date in buckets.index:
        b = buckets.loc[date]
        rets = r.loc[date]
        rows[date] = {q: rets[b == q].mean() for q in range(1, n + 1)}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=range(1, n + 1))


def long_short_spread(factor: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = 5) -> pd.Series:
    """顶档 − 底档 的多空价差时序。"""
    qr = quantile_returns(factor, fwd_ret, n)
    return qr[n] - qr[1]


def turnover(factor: pd.DataFrame, n: int = 5) -> pd.Series:
    """顶档（Q n）成员逐期换手 = |新成员 △ 旧成员| / 旧成员数。"""
    buckets = _ranks(factor, n)
    top_sets = [set(buckets.columns[(buckets.loc[d] == n).values]) for d in buckets.index]
    out = [np.nan]
    for prev, cur in zip(top_sets, top_sets[1:]):
        if not prev:
            out.append(np.nan)
            continue
        changed = len(prev.symmetric_difference(cur)) / 2
        out.append(changed / len(prev))
    return pd.Series(out, index=buckets.index)
