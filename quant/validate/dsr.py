"""Deflated Sharpe Ratio：多重检验折减后的真实显著性。"""

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """N 次独立试验下零假设的期望最大 Sharpe（多重检验基准 SR0）。

    sr_variance：各试验 Sharpe 的方差 V。N≤1 或 V≤0 时无惩罚，返回 0。
    """
    if n_trials <= 1 or sr_variance <= 0 or np.isnan(sr_variance):
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(
    sr: float, sr0: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """观测 Sharpe 扣减基准 SR0 后的真实显著性概率。

    sr/sr0：每期（非年化）Sharpe。n_obs：收益样本数 T。
    skew/kurt：收益的偏度与非超额峰度（正态=3）。返回 [0,1] 概率。
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr**2)
    stat = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(stat))
