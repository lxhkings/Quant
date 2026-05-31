"""回测绩效指标：年化、Sharpe、最大回撤、Calmar、月胜率。

net 收益序列驱动一切。Sharpe 默认按 periods_per_year 年化（×sqrt(N)）；
传 periods_per_year=1 得每期（非年化）Sharpe，供 Deflated Sharpe 使用。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(nav: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """复合年化收益 = (末值/首值)^(年频/期数) - 1。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return float("nan")
    total = nav.iloc[-1] / nav.iloc[0]
    years = len(nav) / periods_per_year
    return total ** (1 / years) - 1


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sharpe = 均值/标准差 × sqrt(年频)。标准差为 0 或样本不足返回 NaN。"""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return r.mean() / sd * np.sqrt(periods_per_year)


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤 = min(nav/历史峰值 - 1)，为非正数。"""
    nav = nav.dropna()
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def calmar(annual: float, mdd: float) -> float:
    """Calmar = 年化收益 / |最大回撤|。回撤为 0 或 NaN 时返回 NaN。"""
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annual / abs(mdd)


def monthly_win_rate(returns: pd.Series) -> float:
    """月度收益为正的比例。"""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    monthly = (1 + r).resample("ME").prod() - 1
    if monthly.empty:
        return float("nan")
    return float((monthly > 0).mean())


@dataclass
class BacktestMetrics:
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    monthly_win_rate: float


def compute_metrics(
    nav: pd.Series, returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> BacktestMetrics:
    """从净值 + net 收益序列汇出全套指标。"""
    ann = annualized_return(nav, periods_per_year)
    mdd = max_drawdown(nav)
    return BacktestMetrics(
        annual_return=ann,
        sharpe=sharpe(returns, periods_per_year),
        max_drawdown=mdd,
        calmar=calmar(ann, mdd),
        monthly_win_rate=monthly_win_rate(returns),
    )
