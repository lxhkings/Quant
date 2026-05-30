import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant.data.universe import membership_mask


def test_all_active_when_only_ipo(fake_lake):
    root, instruments, days = fake_lake
    index = pd.to_datetime(days)
    mask = membership_mask(index, instruments, root=root)
    assert mask.shape == (60, 3)
    # fixture 里 IPO 都在首日，全程在场
    assert mask.all().all()


def test_delist_excludes_after(fake_lake):
    root, instruments, days = fake_lake
    # 给 CCC 补一条 DELIST 事件（第 30 天）
    comp = root / "us" / "components"
    pq.write_table(pa.table({
        "instrument_id": ["CCC"],
        "date": [days[30]],
        "event": ["DELIST"],
        "event_details": ["removed"],
    }), comp / "delist.parquet")

    index = pd.to_datetime(days)
    mask = membership_mask(index, instruments, root=root)
    assert mask["CCC"].iloc[29]      # DELIST 前在场
    assert not mask["CCC"].iloc[30]  # DELIST 当日起出场
    assert not mask["CCC"].iloc[59]
