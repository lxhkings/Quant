"""矩阵缓存：按 key 落 parquet，命中即读，未命中算后写。"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def cache_key(*parts) -> str:
    """把任意可序列化片段拼成稳定短哈希。"""
    raw = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def load_or_compute(
    key: str, compute: Callable[[], pd.DataFrame], cache_dir: Path
) -> pd.DataFrame:
    """命中 `<cache_dir>/<key>.parquet` 则读回，否则算后写。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{key}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    df = compute()
    df.to_parquet(fp)
    return df
