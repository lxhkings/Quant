import pandas as pd

from quant.data.holdout import apply_holdout, research_cutoff


def _matrix():
    idx = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    return pd.DataFrame({"AAA": range(len(idx))}, index=idx)


def test_cutoff_is_two_years_before_end():
    m = _matrix()
    cut = research_cutoff(m.index, holdout_years=2)
    assert cut == pd.Timestamp("2022-12-31")


def test_research_mode_drops_holdout():
    m = _matrix()
    out = apply_holdout(m, mode="research", holdout_years=2)
    assert out.index.max() <= pd.Timestamp("2022-12-31")
    assert out.index.min() == m.index.min()


def test_holdout_mode_keeps_only_locked_window():
    m = _matrix()
    out = apply_holdout(m, mode="holdout", holdout_years=2)
    assert out.index.min() > pd.Timestamp("2022-12-31")
    assert out.index.max() == m.index.max()


def test_full_mode_unchanged():
    m = _matrix()
    out = apply_holdout(m, mode="full", holdout_years=2)
    assert out.equals(m)
