from quant.web import viewmodel


def test_selector_returns_table(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    out = viewmodel.selector(
        ["momentum"], {"momentum": 1.0}, top_n=1,
        lookback=20, skip=1, mode="full",
    )
    assert out["table"].iloc[0]["instrument_id"] == "AAA"
    assert out["table"].iloc[0]["zone"] == "buy"
    assert "as_of" in out
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-9
