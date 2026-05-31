from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_factor_test_new_factor(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    # rel_volume needs volume; registry dispatch should handle it
    result = runner.invoke(
        app,
        ["factor", "test", "rel_volume", "--window", "10",
         "--quantiles", "3", "--mode", "full"],
    )
    assert result.exit_code == 0, result.output
    assert "因子体检报告" in result.output


def test_backtest_with_neutralize(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["backtest", "volatility", "--window", "10", "--quantiles", "3",
         "--freq", "M", "--mode", "full", "--neutralize",
         "--ledger-path", str(tmp_path / "ledger.jsonl"),
         "--state-path", str(tmp_path / "state.json")],
    )
    assert result.exit_code == 0, result.output
    assert "回测体检报告" in result.output
