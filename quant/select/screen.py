"""多因子量化选股：最新截面加权打分 → 0-100 综合分 → 切买入/备选池。

纯逻辑，走 factor.registry 动态取因子（将来基本面因子注册即自动复用）。
选股用最新截面（生产用途，非研究回测），不写 trial 台账、不算 DSR。
"""

from dataclasses import dataclass

import pandas as pd

from quant.combine.synth import combine_score, zscore_factors
from quant.data.holdout import apply_holdout
from quant.data.panel import load_price_matrix
from quant.data.sectors import load_sectors
from quant.factor.registry import compute_factor
from quant.process.neutralize import sector_neutralize


@dataclass
class SelectionResult:
    as_of: pd.Timestamp          # 选股所用最新交易日
    weights: dict[str, float]    # 归一后的因子权重（sum=1）
    table: pd.DataFrame          # 列：instrument_id, score(0-100), rank, zone, sector


def _factor_params(name: str, lookback: int, skip: int, window: int) -> dict:
    """按因子名挑构造参数（动量用 lookback/skip，其余单窗因子用 window）。"""
    if name == "momentum":
        return {"lookback": lookback, "skip": skip}
    return {"window": window}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """权重归一到 sum=1。全为 0 抛 ValueError。"""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("权重之和为 0，无法归一")
    return {k: v / total for k, v in weights.items()}


def screen(
    names: list[str],
    weights: dict[str, float],
    *,
    top_n: int = 10,
    lookback: int = 252,
    skip: int = 21,
    window: int = 200,
    neutralize: bool = False,
    mode: str = "full",
    holdout_years: int = 2,
    market: str = "us",
) -> SelectionResult:
    """对 names 因子按 weights 加权打分，输出最新截面选股表。"""
    close = apply_holdout(load_price_matrix(field="close", market=market), mode, holdout_years)
    volume = apply_holdout(load_price_matrix(field="volume", market=market), mode, holdout_years)
    sectors = load_sectors(market=market)

    raw = {}
    for nm in names:
        f = compute_factor(nm, close, volume=volume, **_factor_params(nm, lookback, skip, window))
        if neutralize:
            f = sector_neutralize(f, sectors)
        raw[nm] = f

    zf = zscore_factors(raw)
    nw = _normalize_weights(weights)
    score = combine_score(zf, nw)

    as_of = score.dropna(how="all").index[-1]
    row = score.loc[as_of].dropna()
    pct = row.rank(pct=True) * 100.0
    ranked = pct.sort_values(ascending=False)

    table = pd.DataFrame({"instrument_id": ranked.index, "score": ranked.values})
    table["rank"] = range(1, len(table) + 1)
    table["zone"] = ["buy" if i < top_n else "candidate" for i in range(len(table))]
    table["sector"] = table["instrument_id"].map(sectors).fillna("未知")
    return SelectionResult(as_of=as_of, weights=nw, table=table.reset_index(drop=True))
