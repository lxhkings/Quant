"""价格面板加载：data_lake daily → 宽矩阵 [date × instrument_id]。"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from quant.config import data_lake_root


def load_price_matrix(
    field: str = "close",
    market: str = "us",
    root: Path | None = None,
) -> pd.DataFrame:
    """
    读 data_lake 的 daily 数据集，pivot 成宽矩阵。

    index=date（升序），columns=instrument_id，值为 float。
    parquet 中价格是 decimal，统一转 float64。
    """
    root = root or data_lake_root()
    path = Path(root) / market / "daily"
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=["instrument_id", "date", field])
    df = table.to_pandas()
    df[field] = df[field].astype("float64")
    wide = df.pivot(index="date", columns="instrument_id", values=field)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()
