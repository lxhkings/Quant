import numpy as np
import pandas as pd
import pytest

from quant.factor.library.amihud import Amihud
from quant.factor.library.rel_volume import RelativeVolume
from quant.factor.library.short_reversal import ShortReversal
from quant.factor.library.volatility import Volatility


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


def test_volatility_name():
    assert Volatility().name == "volatility"


def test_volatility_zero_when_flat():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    f = Volatility(window=21).compute(close)
    assert np.isclose(f["AAA"].iloc[-1], 0.0)


def test_volatility_positive_when_choppy():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    prices = [100.0 + (5.0 if i % 2 else -5.0) for i in range(40)]
    close = pd.DataFrame({"AAA": prices}, index=idx)
    f = Volatility(window=21).compute(close)
    assert f["AAA"].iloc[-1] > 0


def test_rel_volume_name():
    assert RelativeVolume().name == "rel_volume"


def test_rel_volume_zero_when_constant():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    vol = pd.DataFrame({"AAA": [1_000_000.0] * 40}, index=idx)
    f = RelativeVolume(window=21).compute(close, vol)
    assert np.isclose(f["AAA"].iloc[-1], 0.0)


def test_rel_volume_positive_on_spike():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    vols = [1_000_000.0] * 39 + [5_000_000.0]
    vol = pd.DataFrame({"AAA": vols}, index=idx)
    f = RelativeVolume(window=21).compute(close, vol)
    assert f["AAA"].iloc[-1] > 0


def test_rel_volume_requires_volume():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 5}, index=idx)
    with pytest.raises(ValueError, match="volume"):
        RelativeVolume().compute(close, None)


def test_amihud_name():
    assert Amihud().name == "amihud"


def test_amihud_lower_for_higher_dollar_volume():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    prices = [100.0 * (1 + (0.01 if i % 2 else -0.01)) ** i for i in range(40)]
    close = pd.DataFrame({"AAA": prices, "BBB": prices}, index=idx)
    vol = pd.DataFrame({"AAA": [1e6] * 40, "BBB": [1e7] * 40}, index=idx)
    f = Amihud(window=21).compute(close, vol)
    assert f["BBB"].iloc[-1] < f["AAA"].iloc[-1]


def test_amihud_requires_volume():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 5}, index=idx)
    with pytest.raises(ValueError, match="volume"):
        Amihud().compute(close, None)
