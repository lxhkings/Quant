import numpy as np
import pandas as pd

from quant.combine.synth import (
    combine_score,
    equal_weight,
    factor_correlation,
    high_correlation_warnings,
    ic_weight,
    zscore_factors,
)


def _factors():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = list("ABCD")
    a = pd.DataFrame([[1, 2, 3, 4]] * 3, index=idx, columns=cols, dtype=float)
    b = pd.DataFrame([[4, 3, 2, 1]] * 3, index=idx, columns=cols, dtype=float)
    return {"a": a, "b": b}


def test_zscore_factors_row_mean_zero():
    z = zscore_factors(_factors())
    assert np.isclose(z["a"].iloc[0].mean(), 0.0)


def test_equal_weight():
    assert equal_weight(["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_ic_weight_proportional_to_ic():
    w = ic_weight({"a": 0.04, "b": 0.02})
    assert np.isclose(w["a"], 2 / 3)
    assert np.isclose(w["b"], 1 / 3)


def test_ic_weight_negative_falls_back_to_equal():
    w = ic_weight({"a": -0.01, "b": -0.02})
    assert np.isclose(w["a"], 0.5)


def test_combine_score_weighted_sum():
    f = _factors()
    z = zscore_factors(f)
    score = combine_score(z, {"a": 0.5, "b": 0.5})
    # a 与 b 互为反向 zscore → 等权合成在每个截面应抵消为约 0
    assert np.allclose(score.iloc[0].values, 0.0, atol=1e-9)


def test_correlation_and_warnings():
    z = zscore_factors(_factors())
    corr = factor_correlation(z)
    # a、b 完全反向 → 相关 -1
    assert np.isclose(corr.loc["a", "b"], -1.0)
    warns = high_correlation_warnings(corr, threshold=0.7)
    assert warns and warns[0][2] < 0  # 触发预警，相关为负
