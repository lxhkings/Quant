import numpy as np
import pandas as pd
import pytest

from quant.factor.registry import compute_factor, factor_names, make, needs_volume


def test_factor_names_includes_all():
    names = set(factor_names())
    assert {"momentum", "ma_bias", "short_reversal", "volatility",
            "rel_volume", "amihud"} <= names


def test_make_builds_instance():
    f = make("momentum", lookback=20, skip=1)
    assert f.name == "momentum"


def test_make_unknown_raises():
    with pytest.raises(KeyError):
        make("nope")


def test_needs_volume_flags():
    assert needs_volume("amihud") is True
    assert needs_volume("rel_volume") is True
    assert needs_volume("momentum") is False


def test_compute_factor_close_only():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0 * (1.01 ** i) for i in range(30)]}, index=idx)
    out = compute_factor("momentum", close, lookback=20, skip=1)
    assert out.shape == close.shape


def test_compute_factor_with_volume():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 30}, index=idx)
    vol = pd.DataFrame({"AAA": [1e6] * 30}, index=idx)
    out = compute_factor("rel_volume", close, volume=vol, window=21)
    assert np.isclose(out["AAA"].iloc[-1], 0.0)
