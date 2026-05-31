import numpy as np
import pandas as pd

from quant.data.returns import simple_returns


def test_simple_returns_value():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]}, index=idx)
    r = simple_returns(close)
    assert np.isnan(r["AAA"].iloc[0])          # 首日无前值
    assert np.isclose(r["AAA"].iloc[1], 0.10)  # 110/100-1
    assert np.isclose(r["AAA"].iloc[2], -0.10) # 99/110-1
