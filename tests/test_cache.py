import pandas as pd

from quant.cache.store import cache_key, load_or_compute


def test_cache_key_stable_and_distinct():
    k1 = cache_key("factor", "momentum", {"lookback": 252})
    k2 = cache_key("factor", "momentum", {"lookback": 252})
    k3 = cache_key("factor", "momentum", {"lookback": 100})
    assert k1 == k2
    assert k1 != k3


def test_load_or_compute_caches(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return pd.DataFrame({"AAA": [1.0, 2.0]})

    key = cache_key("test")
    first = load_or_compute(key, compute, tmp_path / "cache")
    second = load_or_compute(key, compute, tmp_path / "cache")
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)
