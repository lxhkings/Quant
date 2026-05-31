"""行业板块加载：sectors 数据集 → instrument_id 到 GICS 行业的映射。"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from quant.config import data_lake_root


def load_sectors(market: str = "us", root: Path | None = None) -> pd.Series:
    """返回 index=instrument_id、值=sector 的 Series（每标的取最新一条记录）。"""
    root = root or data_lake_root()
    path = Path(root) / market / "sectors"
    df = (
        ds.dataset(str(path), format="parquet")
        .to_table(columns=["instrument_id", "date", "sector"])
        .to_pandas()
    )
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("instrument_id").last()
    return latest["sector"]
