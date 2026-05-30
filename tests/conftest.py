"""合成 data_lake fixture：hive 分区 parquet，结构与 TrendSpec 一致。"""

from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _trading_days(start: date, n: int) -> list[date]:
    """生成 n 个工作日（跳周末）。"""
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


@pytest.fixture
def fake_lake(tmp_path):
    """
    建 3 个标的 × 60 个交易日的合成 us/daily + components + sectors。

    价格设计：
    - AAA：每日 +0.5%（强动量）
    - BBB：每日 -0.3%（弱）
    - CCC：横盘（0%）
    返回 (root, instruments, trading_days)。
    """
    root = tmp_path / "data_lake"
    days = _trading_days(date(2020, 1, 1), 60)
    specs = {"AAA": (100.0, 0.005), "BBB": (100.0, -0.003), "CCC": (100.0, 0.0)}

    for inst, (p0, drift) in specs.items():
        prices, p = [], p0
        for _ in days:
            prices.append(p)
            p *= 1 + drift
        part = root / "us" / "daily" / f"instrument_id={inst}"
        part.mkdir(parents=True)
        tbl = pa.table({
            "ticker": [inst] * len(days),
            "date": days,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000_000] * len(days),
            "instrument_id": [inst] * len(days),
            "adj_factor": [1.0] * len(days),
        })
        pq.write_table(tbl, part / "2020.parquet")

    comp = root / "us" / "components"
    comp.mkdir(parents=True)
    pq.write_table(pa.table({
        "instrument_id": list(specs),
        "date": [days[0]] * len(specs),
        "event": ["IPO"] * len(specs),
        "event_details": ["first price record"] * len(specs),
    }), comp / "all.parquet")

    sect = root / "us" / "sectors"
    sect.mkdir(parents=True)
    pq.write_table(pa.table({
        "instrument_id": list(specs),
        "date": [date(2000, 1, 1)] * len(specs),
        "sector": ["Tech", "Tech", "Energy"],
        "sector_name": ["Software", "Hardware", "Oil"],
    }), sect / "all.parquet")

    return root, list(specs), days
