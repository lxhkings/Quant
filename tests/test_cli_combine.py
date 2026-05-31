from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def test_combine_runs_end_to_end(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app,
        ["combine", "momentum", "ma_bias", "--weighting", "equal",
         "--lookback", "20", "--skip", "1", "--window", "10",
         "--quantiles", "3", "--freq", "M", "--mode", "full"],
    )
    assert result.exit_code == 0, result.output
    assert "合成回测" in result.output
    assert "权重" in result.output


def test_combine_unknown_factor_errors(fake_lake, monkeypatch):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(
        app, ["combine", "momentum", "nope", "--mode", "full"]
    )
    assert result.exit_code != 0
