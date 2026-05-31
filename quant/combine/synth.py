"""多因子合成：逐因子 zscore → 加权 → 合成总分；含共线性相关矩阵与预警。

正交化（去共线）属 Plan 2b/ v2，本模块不实现。
"""

import pandas as pd

from quant.process.pipeline import zscore


def zscore_factors(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """逐因子做横截面标准化。"""
    return {name: zscore(f) for name, f in factors.items()}


def equal_weight(names: list[str]) -> dict[str, float]:
    """等权：每因子 1/N。"""
    w = 1.0 / len(names)
    return {n: w for n in names}


def ic_weight(ic_means: dict[str, float]) -> dict[str, float]:
    """IC 加权：权重 ∝ max(IC, 0)，归一。全为非正时退回等权。"""
    pos = {n: max(v, 0.0) for n, v in ic_means.items()}
    total = sum(pos.values())
    if total == 0:
        return equal_weight(list(ic_means))
    return {n: v / total for n, v in pos.items()}


def combine_score(
    factors: dict[str, pd.DataFrame], weights: dict[str, float]
) -> pd.DataFrame:
    """加权求和合成总分（因子应已标准化）。"""
    score = None
    for name, f in factors.items():
        term = f * weights[name]
        score = term if score is None else score.add(term, fill_value=0.0)
    return score


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """因子值两两相关矩阵（按对齐后的全体格点配对，逐对剔除 NaN）。"""
    names = list(factors)
    base = factors[names[0]]
    flat = {
        name: f.reindex(index=base.index, columns=base.columns).to_numpy().ravel()
        for name, f in factors.items()
    }
    return pd.DataFrame(flat).corr()


def high_correlation_warnings(
    corr: pd.DataFrame, threshold: float = 0.7
) -> list[tuple[str, str, float]]:
    """返回 |相关| ≥ 阈值的因子对列表。"""
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) >= threshold:
                out.append((cols[i], cols[j], float(v)))
    return out
