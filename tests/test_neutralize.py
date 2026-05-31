import numpy as np
import pandas as pd

from quant.process.neutralize import sector_neutralize


def test_sector_neutralize_demeans_within_sector():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C", "D"]
    factor = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]] * 2, index=idx, columns=cols)
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = sector_neutralize(factor, sectors)
    # Tech {A,B} 均值 2 → A=-1,B=1；Energy {C,D} 均值 15 → C=-5,D=5
    assert np.allclose(out.iloc[0].values, [-1.0, 1.0, -5.0, 5.0])
    # 每个行业内逐行均值为 0
    assert np.isclose(out[["A", "B"]].iloc[0].mean(), 0.0)
    assert np.isclose(out[["C", "D"]].iloc[0].mean(), 0.0)
