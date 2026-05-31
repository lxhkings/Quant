import pandas as pd

from quant.report.backtest_card import BacktestReport
from quant.report.runner import run_backtest_report


def _world():
    idx = pd.bdate_range("2020-01-06", periods=5)
    cols = list("ABCDEFGHIJ")
    factor = pd.DataFrame([list(range(1, 11))] * 5, index=idx, columns=cols, dtype=float)
    close = pd.DataFrame(
        {c: [100.0 * (1 + 0.001 * i) ** t for t in range(5)] for i, c in enumerate(cols)},
        index=idx,
    )
    return factor, close


def test_run_backtest_report_records_trial(tmp_path):
    factor, close = _world()
    rep = run_backtest_report(
        "momentum", {"lookback": 1}, factor, close,
        quantiles=5, side="long", freq="M", cost_bps=0.0,
        ledger_path=tmp_path / "ledger.jsonl", holdout_consumed=False,
    )
    assert isinstance(rep, BacktestReport)
    assert rep.factor_name == "momentum"
    assert rep.n_trials == 1


def test_run_backtest_report_trials_accumulate(tmp_path):
    factor, close = _world()
    lp = tmp_path / "ledger.jsonl"
    run_backtest_report("momentum", {"lookback": 1}, factor, close,
                        ledger_path=lp, holdout_consumed=False)
    rep2 = run_backtest_report("momentum", {"lookback": 1}, factor, close,
                              ledger_path=lp, holdout_consumed=False)
    assert rep2.n_trials == 2
