import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult, backtest


def _setup():
    """10 标的、5 个交易日，单月（单次调仓）。

    因子恒定排序 A<B<...<J；n=5 → 顶档 = {I,J} 各 0.5。
    收益：每标的每日固定，J=+2%、I=+1%、其余 0 → 顶档篮子日收益 1.5%。
    """
    idx = pd.bdate_range("2020-01-06", periods=5)  # 同一周/月内
    cols = list("ABCDEFGHIJ")
    factor = pd.DataFrame([list(range(1, 11))] * 5, index=idx, columns=cols, dtype=float)
    drift = {c: 0.0 for c in cols}
    drift["I"], drift["J"] = 0.01, 0.02
    close = pd.DataFrame(
        {c: [100.0 * (1 + drift[c]) ** i for i in range(5)] for c in cols}, index=idx
    )
    return factor, close


def test_zero_cost_nav_matches_basket():
    factor, close = _setup()
    res = backtest(factor, close, n=5, side="long", freq="M", cost_bps=0.0)
    assert isinstance(res, BacktestResult)
    # 顶档篮子（I,J 等权）日收益 = (1% + 2%)/2 = 1.5%；
    # 调仓日 idx[0] 建仓，权重 shift(1) 从 idx[1] 起生效 → 净值含 4 个 1.5% 复利
    expected = (1 + 0.015) ** 4
    assert np.isclose(res.nav.iloc[-1], expected, rtol=1e-6)


def test_cost_drags_nav_below_zero_cost():
    factor, close = _setup()
    free = backtest(factor, close, n=5, side="long", freq="M", cost_bps=0.0)
    paid = backtest(factor, close, n=5, side="long", freq="M", cost_bps=50.0)
    # 有成本净值 < 无成本净值
    assert paid.nav.iloc[-1] < free.nav.iloc[-1]


def test_turnover_full_on_first_rebalance():
    factor, close = _setup()
    res = backtest(factor, close, n=5, side="long", freq="M", cost_bps=0.0)
    # 首次建仓从空仓到 {I:0.5, J:0.5} → 换手 = 0.5（单边定义 /2）
    reb_turnover = res.turnover[res.turnover > 0]
    assert np.isclose(reb_turnover.iloc[0], 0.5)
