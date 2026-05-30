import numpy as np

from quant.data.panel import load_price_matrix


def test_shape_and_orientation(fake_lake):
    root, instruments, days = fake_lake
    close = load_price_matrix(field="close", root=root)
    assert close.shape == (60, 3)
    assert set(close.columns) == set(instruments)
    assert [d.date() for d in close.index] == days


def test_values_are_float(fake_lake):
    root, _, _ = fake_lake
    close = load_price_matrix(field="close", root=root)
    assert close.dtypes.unique().tolist() == [np.float64]
    # AAA 强动量：末值 > 首值
    assert close["AAA"].iloc[-1] > close["AAA"].iloc[0]
    assert close["BBB"].iloc[-1] < close["BBB"].iloc[0]
