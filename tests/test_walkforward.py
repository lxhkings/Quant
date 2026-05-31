import numpy as np
import pandas as pd

from quant.validate.walkforward import oos_is_ratio, walk_forward


def _world():
    """10 标的、120 交易日。标的按编号有固定日漂移：J 最强、A 最弱。

    构造两个候选因子：'good'=升序排名（追强者）、'bad'=降序（追弱者）。
    good 在 IS/OOS 都应正 Sharpe；bad 负。
    """
    idx = pd.bdate_range("2020-01-01", periods=120)
    cols = list("ABCDEFGHIJ")
    drift = {c: 0.0005 * i for i, c in enumerate(cols)}  # A=0 .. J=0.0045
    close = pd.DataFrame(
        {c: [100.0 * (1 + drift[c]) ** t for t in range(120)] for c in cols}, index=idx
    )
    asc = pd.DataFrame([list(range(1, 11))] * 120, index=idx, columns=cols, dtype=float)
    desc = pd.DataFrame([list(range(10, 0, -1))] * 120, index=idx, columns=cols, dtype=float)
    factors = {"good": asc, "bad": desc}
    return close, factors


def test_walk_forward_picks_good_param():
    close, factors = _world()
    wf = walk_forward(
        build=lambda p: factors[p],
        close=close,
        params=["good", "bad"],
        is_days=60,
        oos_days=30,
        n=5,
        side="long",
    )
    assert not wf.empty
    # 每个窗口 IS 都应选中 'good'
    assert (wf["best_param"] == "good").all()
    # OOS Sharpe 为正（信号持续）
    assert (wf["oos_sharpe"] > 0).all()


def test_oos_is_ratio_finite():
    close, factors = _world()
    wf = walk_forward(
        build=lambda p: factors[p],
        close=close,
        params=["good", "bad"],
        is_days=60,
        oos_days=30,
        n=5,
        side="long",
    )
    ratio = oos_is_ratio(wf)
    assert np.isfinite(ratio)
