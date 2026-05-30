"""价格面板加载：data_lake daily → 宽矩阵 [date × instrument_id]。"""

from pathlib import Path

import pandas as pd

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

    # 逐分区读取再合并，避免 pyarrow schema 合并冲突（string/large_string/dict）
    frames = []
    for part_dir in sorted(path.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("instrument_id="):
            continue
        inst_id = part_dir.name.split("=", 1)[1]
        for pq_file in part_dir.glob("*.parquet"):
            df = pd.read_parquet(pq_file, columns=["date", field])
            df["instrument_id"] = inst_id
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet files found under {path}")

    df = pd.concat(frames, ignore_index=True)
    df[field] = df[field].astype("float64")
    wide = df.pivot(index="date", columns="instrument_id", values=field)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()
