import numpy as np
import pandas as pd

from quant.factor.library.ma_bias import MABias
from quant.factor.library.momentum import Momentum


def test_momentum_name():
    assert Momentum().name == "momentum"


def test_momentum_computes_skip_lookback():
    # 等比增长序列，动量恒正
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    close = pd.DataFrame({"AAA": [100.0 * (1.01 ** i) for i in range(300)]}, index=idx)
    f = Momentum(lookback=252, skip=21).compute(close)
    assert f.shape == close.shape
    # 前 lookback 行不足，为 NaN
    assert np.isnan(f["AAA"].iloc[250])
    # 充足后为正（上涨）
    assert f["AAA"].iloc[-1] > 0


def test_momentum_value():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 30}, index=idx)
    close.iloc[-1, 0] = 110.0  # 仅最后一天跳涨（在 skip 窗口内，应被跳过）
    f = Momentum(lookback=20, skip=5).compute(close)
    # 动量 = close[t-skip]/close[t-skip-lookback]-1；最后的跳涨在 skip 内不计入
    assert np.isclose(f["AAA"].iloc[-1], 0.0)


def test_ma_bias_name():
    assert MABias().name == "ma_bias"


def test_ma_bias_zero_when_flat():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0] * 40}, index=idx)
    f = MABias(window=20).compute(close)
    # 横盘时价格=均线，乖离=0
    assert np.isclose(f["AAA"].iloc[-1], 0.0)


def test_ma_bias_positive_when_above_ma():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame({"AAA": [100.0 * (1.01 ** i) for i in range(40)]}, index=idx)
    f = MABias(window=20).compute(close)
    # 持续上涨，价格在均线上方，乖离为正
    assert f["AAA"].iloc[-1] > 0
