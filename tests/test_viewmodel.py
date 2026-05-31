import pandas as pd

from quant.web import viewmodel


def test_available_factors_nonempty():
    assert "momentum" in viewmodel.available_factors()


def test_workshop_returns_markdown(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    md = viewmodel.workshop(
        "momentum", lookback=20, skip=1, window=10,
        horizon=5, quantiles=3, mode="full", neutralize=False,
    )
    assert "因子体检报告" in md
    assert "momentum" in md


def test_combine_returns_payload(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    out = viewmodel.combine(
        ["momentum", "ma_bias"], weighting="equal",
        lookback=20, skip=1, window=10, horizon=5,
        quantiles=3, side="long", freq="M", cost_bps=10.0, mode="full",
    )
    assert set(out["weights"]) == {"momentum", "ma_bias"}
    assert "annual_return" in out["metrics"]


def test_history_lists_trials(tmp_path):
    from quant.validate.ledger import Ledger
    lp = tmp_path / "ledger.jsonl"
    Ledger(lp).record({"factor": "momentum", "sharpe": 1.2})
    rows = viewmodel.history(lp)
    assert len(rows) == 1
    assert rows[0]["factor"] == "momentum"
