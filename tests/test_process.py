import numpy as np
import pandas as pd

from quant.process.pipeline import Pipeline, winsorize, zscore


def test_winsorize_clips_per_row():
    # 一行 [1..100]，1%/99% 截断把极端值拉回分位
    row = pd.DataFrame([list(range(1, 101))], dtype=float)
    out = winsorize(row, lower=0.01, upper=0.99)
    assert out.min(axis=1).iloc[0] >= np.nanpercentile(range(1, 101), 1)
    assert out.max(axis=1).iloc[0] <= np.nanpercentile(range(1, 101), 99)


def test_zscore_row_mean_zero_std_one():
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = zscore(df)
    assert np.isclose(out.mean(axis=1).iloc[0], 0.0)
    assert np.isclose(out.std(axis=1, ddof=0).iloc[0], 1.0)


def test_zscore_ignores_nan():
    df = pd.DataFrame([[1.0, np.nan, 3.0]])
    out = zscore(df)
    assert np.isnan(out.iloc[0, 1])
    assert np.isclose(np.nanmean(out.iloc[0].values), 0.0)


def test_pipeline_chains():
    df = pd.DataFrame([[1.0, 2.0, 3.0, 100.0]])
    pipe = Pipeline([winsorize, zscore])
    out = pipe(df)
    # 链式后均值约 0
    assert np.isclose(np.nanmean(out.iloc[0].values), 0.0)
