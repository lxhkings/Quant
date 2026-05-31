from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_backtest_runs_end_to_end(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["backtest", "momentum", "--lookback", "20", "--skip", "1",
         "--quantiles", "3", "--freq", "M", "--mode", "full",
         "--ledger-path", str(tmp_path / "ledger.jsonl"),
         "--state-path", str(tmp_path / "state.json")],
    )
    assert result.exit_code == 0, result.output
    assert "回测体检报告" in result.output
    assert "DSR" in result.output


def test_backtest_unknown_factor_errors(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["backtest", "nope", "--mode", "full",
         "--ledger-path", str(tmp_path / "ledger.jsonl"),
         "--state-path", str(tmp_path / "state.json")],
    )
    assert result.exit_code != 0
