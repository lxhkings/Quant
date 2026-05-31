import numpy as np
import pandas as pd

from quant.factor.library.short_reversal import ShortReversal


def test_short_reversal_name():
    assert ShortReversal().name == "short_reversal"


def test_short_reversal_negative_when_rising():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0 * (1.01 ** i) for i in range(30)]}, index=idx)
    f = ShortReversal(window=21).compute(close)
    assert f["AAA"].iloc[-1] < 0
    assert np.isnan(f["AAA"].iloc[0])


def test_short_reversal_zero_when_flat():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 30}, index=idx)
    f = ShortReversal(window=21).compute(close)
    assert np.isclose(f["AAA"].iloc[-1], 0.0)
