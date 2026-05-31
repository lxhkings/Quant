from quant.report.leaderboard import scan_factors
from quant.validate.ledger import Ledger


def test_scan_factors_ranks_subset(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    lp = tmp_path / "scan_ledger.jsonl"
    table = scan_factors(
        ["momentum", "short_reversal"],
        lookback=20, skip=1, window=10, horizon=5, quantiles=3,
        mode="full", scan_ledger_path=lp,
    )
    assert set(table["factor"]) == {"momentum", "short_reversal"}
    assert list(table.columns) == [
        "factor", "ic_mean", "ic_ir", "t_stat", "long_short_annual"
    ]
    # 独立 scan 台账写了 2 条；DSR 主台账不受影响
    assert Ledger(lp).count() == 2


def test_scan_factors_defaults_to_all(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    table = scan_factors(
        lookback=20, skip=1, window=10, horizon=5, quantiles=3, mode="full",
        scan_ledger_path=tmp_path / "s.jsonl",
    )
    # 默认跑全部 6 个已注册因子
    assert len(table) == 6
    assert "amihud" in set(table["factor"])
