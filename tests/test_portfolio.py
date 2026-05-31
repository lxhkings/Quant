import numpy as np
import pandas as pd

from quant.backtest.engine import rebalance_dates, target_weights


def test_rebalance_dates_monthly_first_trading_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    reb = rebalance_dates(idx, freq="M")
    # 3 个月 → 3 个调仓日，各为当月首个交易日
    assert len(reb) == 3
    assert reb[0] == idx[idx.month == 1][0]
    assert reb[1] == idx[idx.month == 2][0]


def test_target_weights_long_top_quantile_equal():
    row = pd.Series({c: v for c, v in zip("ABCDEFGHIJ", range(1, 11))}, dtype=float)
    w = target_weights(row, n=5, side="long")
    # 10 标的 5 档 → 每档 2 个；顶档 = 两个最高因子 {I,J}，各 0.5
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["J"], 0.5)
    assert np.isclose(w["I"], 0.5)
    assert np.isclose(w["A"], 0.0)


def test_target_weights_long_short_nets_zero():
    row = pd.Series({c: v for c, v in zip("ABCDEFGHIJ", range(1, 11))}, dtype=float)
    w = target_weights(row, n=5, side="long_short")
    # 顶档 +0.5/+0.5，底档 -0.5/-0.5 → 净敞口 0，毛敞口 2
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(w.abs().sum(), 2.0)
    assert w["J"] > 0 and w["A"] < 0


def test_target_weights_all_zero_when_insufficient():
    row = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan}, dtype=float)
    w = target_weights(row, n=5, side="long")
    # 有效标的 < 档数 → 不建仓
    assert np.isclose(w.abs().sum(), 0.0)
