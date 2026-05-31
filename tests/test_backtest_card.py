from quant.report.backtest_card import BacktestReport


def _report(sharpe=1.5, dsr=0.97, annual=0.18):
    return BacktestReport(
        factor_name="momentum",
        params={"lookback": 252, "skip": 21},
        annual_return=annual,
        sharpe=sharpe,
        max_drawdown=-0.12,
        calmar=1.5,
        monthly_win_rate=0.58,
        avg_turnover=0.20,
        deflated_sharpe=dsr,
        n_trials=8,
        holdout_consumed=False,
    )


def test_markdown_contains_key_fields():
    md = _report().to_markdown()
    assert "momentum" in md
    assert "Sharpe" in md
    assert "DSR" in md
    assert "lookback" in md


def test_green_light_when_strong():
    md = _report(sharpe=1.5, dsr=0.97, annual=0.18).to_markdown()
    assert "🟢" in md


def test_red_light_when_weak():
    md = _report(sharpe=0.3, dsr=0.40, annual=-0.05).to_markdown()
    assert "🔴" in md


def test_holdout_status_shown():
    md = _report().to_markdown()
    assert "holdout" in md.lower()


def test_bt_verdict_strong_says_live_candidate():
    md = _report(annual=0.15, sharpe=1.5, dsr=0.99).to_markdown()
    assert "通俗结论" in md
    assert "可考虑实盘候选" in md


def test_bt_verdict_weak_says_not_recommended():
    md = _report(annual=-0.05, sharpe=0.2, dsr=0.3).to_markdown()
    assert "不建议使用" in md


def test_bt_glossary_present():
    md = _report(annual=0.15, sharpe=1.5, dsr=0.99).to_markdown()
    assert "DSR=" in md
