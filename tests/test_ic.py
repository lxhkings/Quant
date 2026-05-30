import numpy as np
import pandas as pd

from quant.eval.ic import ic_series, ic_summary


def _frames():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = ["A", "B", "C", "D"]
    factor = pd.DataFrame([[1, 2, 3, 4]] * 3, index=idx, columns=cols, dtype=float)
    return factor, idx, cols


def test_perfect_rank_ic_is_one():
    factor, idx, cols = _frames()
    fwd = factor.copy()  # 收益与因子完全同序
    ic = ic_series(factor, fwd, method="spearman")
    assert np.allclose(ic.values, 1.0)


def test_inverse_rank_ic_is_minus_one():
    factor, idx, cols = _frames()
    fwd = pd.DataFrame([[4, 3, 2, 1]] * 3, index=idx, columns=cols, dtype=float)
    ic = ic_series(factor, fwd, method="spearman")
    assert np.allclose(ic.values, -1.0)


def test_ic_ignores_nan_cells():
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    factor = pd.DataFrame([[1.0, 2.0, np.nan]], index=idx, columns=["A", "B", "C"])
    fwd = pd.DataFrame([[1.0, 2.0, 999.0]], index=idx, columns=["A", "B", "C"])
    ic = ic_series(factor, fwd, method="spearman")
    assert np.isclose(ic.iloc[0], 1.0)  # 只用 A、B 两点，完全同序


def test_ic_summary_fields():
    ic = pd.Series([0.1, 0.2, 0.0, 0.3, -0.1])
    s = ic_summary(ic)
    assert set(s) == {"ic_mean", "ic_std", "ic_ir", "t_stat", "n"}
    assert np.isclose(s["ic_mean"], 0.1)
    assert s["n"] == 5
    assert np.isclose(s["ic_ir"], s["ic_mean"] / s["ic_std"])
