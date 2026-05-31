from quant.web._shared import GLOSSARY


def test_glossary_covers_core_terms():
    required = {
        "lookback", "skip", "window", "horizon",
        "IC", "ICIR", "分位档数", "共线性", "DSR", "行业中性化",
    }
    assert required <= set(GLOSSARY)


def test_glossary_values_are_nonempty_strings():
    assert all(isinstance(v, str) and v.strip() for v in GLOSSARY.values())
