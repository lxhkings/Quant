from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_factor_test_all_runs(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(app, [
        "factor", "test-all",
        "--lookback", "20", "--skip", "1", "--window", "10",
        "--horizon", "5", "--quantiles", "3", "--mode", "full",
        "--scan-ledger-path", str(tmp_path / "scan.jsonl"),
    ])
    assert result.exit_code == 0, result.output
    assert "排行榜" in result.output
    assert "momentum" in result.output
