"""IC / RankIC 横截面检验。

IC = 每个截面（每日）因子值与前瞻收益的相关系数时序。
- method="pearson"：线性相关
- method="spearman"：秩相关（RankIC，抗异常值，主用）
全程向量化：逐行相关用矩阵运算，不按日循环。
"""

import numpy as np
import pandas as pd


def ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """返回每日 IC 时序。"""
    f, r = factor.align(fwd_ret, join="inner")
    # 任一为 NaN 的格子，两边都置 NaN，保证逐行配对一致
    valid = f.notna() & r.notna()
    f = f.where(valid)
    r = r.where(valid)

    if method == "spearman":
        f = f.rank(axis=1)
        r = r.rank(axis=1)
    elif method != "pearson":
        raise ValueError(f"未知 method: {method!r}")

    fc = f.sub(f.mean(axis=1), axis=0)
    rc = r.sub(r.mean(axis=1), axis=0)
    num = (fc * rc).sum(axis=1, min_count=1)
    den = np.sqrt((fc**2).sum(axis=1, min_count=1) * (rc**2).sum(axis=1, min_count=1))
    return num / den


def ic_summary(ic: pd.Series) -> dict:
    """IC 时序的统计量：均值、标准差、IR、t 值、有效样本数。"""
    ic = ic.dropna()
    n = len(ic)
    mean = ic.mean()
    std = ic.std(ddof=1)
    ir = mean / std if std else np.nan
    t_stat = ir * np.sqrt(n) if n else np.nan
    return {"ic_mean": mean, "ic_std": std, "ic_ir": ir, "t_stat": t_stat, "n": n}
