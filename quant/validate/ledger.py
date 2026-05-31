"""试验台账：每次回测/扫参追加一行 JSON，记录 factor/params/sharpe。

纪律——扫了多少组参数必须记账，Deflated Sharpe 用该次数 N 折减。
"""

import json
import math
from pathlib import Path


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict) -> None:
        """追加一条试验记录。"""
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self) -> list[dict]:
        """读全部记录（文件不存在返回空）。"""
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def count(self) -> int:
        """试验次数 N。"""
        return len(self.entries())

    def sharpes(self) -> list[float]:
        """所有有效（非 NaN）Sharpe 值，供 DSR 估方差。"""
        return [
            e["sharpe"]
            for e in self.entries()
            if "sharpe" in e and not math.isnan(e["sharpe"])
        ]
