"""PIT universe：从 components 的 IPO/DELIST 事件建成分布尔矩阵。"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from quant.config import data_lake_root


def membership_mask(
    index: pd.DatetimeIndex,
    instruments: list[str],
    market: str = "us",
    root: Path | None = None,
) -> pd.DataFrame:
    """
    返回 [date × instrument_id] 布尔矩阵。

    True = 该日该标的在场（IPO 日含 ~ DELIST 日不含）。
    无 DELIST 的标的，IPO 日起一直在场。
    """
    root = root or data_lake_root()
    path = Path(root) / market / "components"
    events = ds.dataset(str(path), format="parquet").to_table(
        columns=["instrument_id", "date", "event"]
    ).to_pandas()
    events["date"] = pd.to_datetime(events["date"])

    mask = pd.DataFrame(False, index=index, columns=instruments)
    for inst in instruments:
        e = events[events["instrument_id"] == inst]
        ipo = e.loc[e["event"] == "IPO", "date"]
        delist = e.loc[e["event"] == "DELIST", "date"]
        start = ipo.min() if not ipo.empty else index[0]
        active = index >= start
        if not delist.empty:
            active &= index < delist.min()
        mask[inst] = active
    return mask
