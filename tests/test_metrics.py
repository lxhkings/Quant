import numpy as np
import pandas as pd

from quant.backtest.metrics import (
    BacktestMetrics,
    annualized_return,
    calmar,
    compute_metrics,
    max_drawdown,
    monthly_win_rate,
    sharpe,
)


def test_annualized_return_doubling_in_one_year():
    nav = pd.Series([1.0, 2.0])
    # 2 个点、每年 2 期 → 跨 1 年 → 翻倍 = 100%
    assert np.isclose(annualized_return(nav, periods_per_year=2), 1.0)


def test_max_drawdown_known_path():
    nav = pd.Series([1.0, 1.2, 0.9, 1.0])
    # 峰值 1.2 → 谷 0.9 → 回撤 -0.25
    assert np.isclose(max_drawdown(nav), -0.25)


def test_sharpe_nan_when_flat():
    r = pd.Series([0.01, 0.01, 0.01])  # 标准差 0
    assert np.isnan(sharpe(r))


def test_sharpe_positive_when_mostly_up():
    r = pd.Series([0.02, -0.01, 0.03, 0.01])
    assert sharpe(r, periods_per_year=1) > 0


def test_calmar():
    assert np.isclose(calmar(0.2, -0.1), 2.0)
    assert np.isnan(calmar(0.2, 0.0))


def test_monthly_win_rate_half():
    # 1 月全涨、2 月全跌 → 月胜率 0.5
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    vals = [0.01 if d.month == 1 else -0.01 for d in idx]
    r = pd.Series(vals, index=idx)
    assert np.isclose(monthly_win_rate(r), 0.5)


def test_compute_metrics_bundle():
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.01] * 10, index=idx)
    nav = (1 + r).cumprod()
    m = compute_metrics(nav, r)
    assert isinstance(m, BacktestMetrics)
    assert m.annual_return > 0
    assert m.max_drawdown <= 0
