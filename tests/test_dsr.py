import numpy as np

from quant.validate.dsr import deflated_sharpe, expected_max_sharpe


def test_expected_max_zero_for_single_trial():
    # N=1 无多重检验惩罚
    assert expected_max_sharpe(sr_variance=1.0, n_trials=1) == 0.0


def test_expected_max_grows_with_trials():
    e10 = expected_max_sharpe(sr_variance=1.0, n_trials=10)
    e100 = expected_max_sharpe(sr_variance=1.0, n_trials=100)
    assert 0 < e10 < e100


def test_dsr_high_when_sr_beats_benchmark():
    # 强 SR、零基准、长样本、正态 → DSR≈1
    assert deflated_sharpe(sr=0.2, sr0=0.0, n_obs=1000) > 0.99


def test_dsr_low_when_sr_below_benchmark():
    assert deflated_sharpe(sr=0.0, sr0=0.1, n_obs=1000) < 0.01


def test_dsr_half_at_benchmark():
    assert np.isclose(deflated_sharpe(sr=0.1, sr0=0.1, n_obs=500), 0.5)


def test_dsr_nan_when_too_few_obs():
    assert np.isnan(deflated_sharpe(sr=0.2, sr0=0.0, n_obs=1))
