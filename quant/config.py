"""配置：data_lake 根路径解析。"""

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent.parent / "TrendSpec" / "data_lake"


def data_lake_root() -> Path:
    """返回 data_lake 根目录。env QUANT_DATA_LAKE_ROOT 优先，否则默认同级 TrendSpec。"""
    env = os.getenv("QUANT_DATA_LAKE_ROOT")
    return Path(env) if env else _DEFAULT
