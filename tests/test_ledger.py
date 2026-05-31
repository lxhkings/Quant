from quant.validate.ledger import Ledger


def test_record_and_count(tmp_path):
    led = Ledger(tmp_path / "sub" / "ledger.jsonl")  # 自动建父目录
    led.record({"factor": "momentum", "sharpe": 1.2})
    led.record({"factor": "momentum", "sharpe": 0.8})
    assert led.count() == 2


def test_sharpes_skips_missing_and_nan(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.record({"factor": "a", "sharpe": 1.0})
    led.record({"factor": "b"})                     # 无 sharpe
    led.record({"factor": "c", "sharpe": float("nan")})
    assert led.sharpes() == [1.0]


def test_entries_empty_when_no_file(tmp_path):
    led = Ledger(tmp_path / "missing.jsonl")
    assert led.entries() == []
    assert led.count() == 0
