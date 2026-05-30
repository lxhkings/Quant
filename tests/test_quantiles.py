import numpy as np
import pandas as pd

from quant.eval.quantiles import long_short_spread, quantile_returns, turnover


def _data():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = list("ABCDEFGHIJ")  # 10 标的
    # 因子值 1..10；前瞻收益与因子单调正相关
    factor = pd.DataFrame([list(range(1, 11))] * 2, index=idx, columns=cols, dtype=float)
    fwd = pd.DataFrame([[c * 0.01 for c in range(1, 11)]] * 2, index=idx, columns=cols, dtype=float)
    return factor, fwd


def test_quantile_returns_monotonic():
    factor, fwd = _data()
    qr = quantile_returns(factor, fwd, n=5)
    assert qr.shape[1] == 5
    # Q5（高因子）平均收益 > Q1（低因子）
    means = qr.mean()
    assert means.iloc[-1] > means.iloc[0]
    # 单调递增
    assert means.is_monotonic_increasing


def test_long_short_spread_positive():
    factor, fwd = _data()
    spread = long_short_spread(factor, fwd, n=5)
    assert (spread > 0).all()


def test_turnover_zero_when_stable():
    # 因子两期完全相同 → 分组不变 → 换手 0
    factor, fwd = _data()
    to = turnover(factor, n=5)
    assert np.isclose(to.dropna().iloc[0], 0.0)


def test_turnover_full_when_reversed():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = list("ABCDEFGHIJ")
    factor = pd.DataFrame(
        [list(range(1, 11)), list(range(10, 0, -1))], index=idx, columns=cols, dtype=float
    )
    to = turnover(factor, n=5)
    # 顶档（Q5）成员从 {高因子} 翻成 {低因子}，换手为 1（完全换血）
    assert to.dropna().iloc[0] > 0.5
