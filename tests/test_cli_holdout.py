from typer.testing import CliRunner

from quant.cli import app

runner = CliRunner()


def _args(tmp_path):
    return [
        "holdout", "momentum", "--yes",
        "--lookback", "20", "--skip", "1", "--quantiles", "3",
        "--freq", "M", "--holdout-years", "0",
        "--ledger-path", str(tmp_path / "ledger.jsonl"),
        "--state-path", str(tmp_path / "state.json"),
    ]


def test_holdout_runs_and_marks_consumed(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    result = runner.invoke(app, _args(tmp_path))
    assert result.exit_code == 0, result.output
    assert "回测体检报告" in result.output
    assert "holdout 已消耗：是" in result.output


def test_holdout_second_run_rejected(fake_lake, monkeypatch, tmp_path):
    root, _, _ = fake_lake
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(root))
    first = runner.invoke(app, _args(tmp_path))
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, _args(tmp_path))
    assert second.exit_code != 0
    assert "已消耗" in second.output
