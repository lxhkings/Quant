import numpy as np
import pandas as pd

from quant.data.returns import forward_returns


def test_horizon_1():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    close = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 121.0]}, index=idx)
    fwd = forward_returns(close, horizon=1)
    # 第 t 日值 = close[t+1]/close[t]-1
    assert np.isclose(fwd["AAA"].iloc[0], 0.10)
    assert np.isclose(fwd["AAA"].iloc[1], 0.10)
    assert np.isclose(fwd["AAA"].iloc[2], 0.0)
    assert np.isnan(fwd["AAA"].iloc[3])  # 最后一日无前瞻


def test_horizon_2():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    close = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 133.1]}, index=idx)
    fwd = forward_returns(close, horizon=2)
    assert np.isclose(fwd["AAA"].iloc[0], 0.21)   # 121/100-1
    assert np.isnan(fwd["AAA"].iloc[2])
    assert np.isnan(fwd["AAA"].iloc[3])
