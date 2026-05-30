from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_factor_test_runs_end_to_end(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["factor", "test", "momentum", "--lookback", "20", "--skip", "1",
         "--mode", "full", "--quantiles", "3"],
    )
    assert result.exit_code == 0, result.output
    assert "因子体检报告" in result.output
    assert "momentum" in result.output


def test_unknown_factor_errors(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(app, ["factor", "test", "nope", "--mode", "full"])
    assert result.exit_code != 0
