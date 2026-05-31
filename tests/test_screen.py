from quant.select.screen import screen


def test_screen_ranks_and_zones(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    res = screen(["momentum"], {"momentum": 1.0}, top_n=1,
                 lookback=20, skip=1, mode="full")
    top = res.table.iloc[0]
    # AAA 强动量 → 排第一、综合分 100、买入池
    assert top["instrument_id"] == "AAA"
    assert top["zone"] == "buy"
    assert abs(top["score"] - 100.0) < 1e-9
    # 其余进备选池
    assert (res.table["zone"].iloc[1:] == "candidate").all()
    # 行业标签 join 成功
    assert top["sector"] == "Tech"


def test_screen_normalizes_weights(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    res = screen(["momentum", "ma_bias"], {"momentum": 1.0, "ma_bias": 3.0},
                 top_n=2, lookback=20, skip=1, window=10, mode="full")
    assert abs(sum(res.weights.values()) - 1.0) < 1e-9
    assert abs(res.weights["ma_bias"] - 0.75) < 1e-9


def test_screen_zero_weights_raises(fake_lake, monkeypatch):
    import pytest
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    with pytest.raises(ValueError, match="权重"):
        screen(["momentum"], {"momentum": 0.0}, top_n=1,
               lookback=20, skip=1, mode="full")
