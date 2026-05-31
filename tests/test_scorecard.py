from quant.report.scorecard import FactorReport


def _report(ic_ir=0.6, mono=True, ls=0.02, turnover=0.15):
    return FactorReport(
        factor_name="momentum",
        params={"lookback": 252, "skip": 21},
        ic_mean=0.04,
        ic_ir=ic_ir,
        t_stat=2.5,
        n=200,
        quantile_means=[0.001, 0.003, 0.005, 0.008, 0.012],
        long_short_annual=ls,
        monotonic=mono,
        avg_turnover=turnover,
        holdout_consumed=False,
    )


def test_markdown_contains_key_fields():
    md = _report().to_markdown()
    assert "momentum" in md
    assert "IC-IR" in md
    assert "0.6" in md
    assert "lookback" in md


def test_green_light_when_strong():
    md = _report(ic_ir=0.6, mono=True, ls=0.02).to_markdown()
    assert "🟢" in md


def test_red_light_when_weak():
    md = _report(ic_ir=0.1, mono=False, ls=-0.01).to_markdown()
    assert "🔴" in md


def test_holdout_status_shown():
    md = _report().to_markdown()
    assert "holdout" in md.lower()


def test_plain_verdict_strong_says_candidate():
    md = _report(ic_ir=0.6, mono=True, ls=0.02).to_markdown()
    assert "通俗结论" in md
    assert "可进入候选池" in md


def test_plain_verdict_weak_says_not_recommended():
    md = _report(ic_ir=0.1, mono=False, ls=-0.01).to_markdown()
    assert "不建议使用" in md


def test_plain_verdict_high_turnover_warns_cost():
    md = _report(ic_ir=0.6, mono=True, ls=0.02, turnover=0.30).to_markdown()
    assert "交易成本" in md


def test_glossary_present():
    md = _report(ic_ir=0.6, mono=True, ls=0.02).to_markdown()
    assert "IC-IR=因子预测力" in md
