"""Holdout 闸门：强制每个因子仅在 holdout 上验证一次。

状态文件 {factor_name: 消耗时间戳}。消耗后再跑 = 作弊 → 拒绝。
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _load(state_path: Path) -> dict:
    p = Path(state_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def is_consumed(factor_name: str, state_path: Path) -> bool:
    """该因子的 holdout 是否已消耗。"""
    return factor_name in _load(state_path)


def mark_consumed(factor_name: str, state_path: Path) -> None:
    """标记该因子 holdout 已消耗（写时间戳）。"""
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _load(p)
    data[factor_name] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def assert_not_consumed(factor_name: str, state_path: Path) -> None:
    """已消耗则抛 RuntimeError。"""
    if is_consumed(factor_name, state_path):
        raise RuntimeError(f"holdout 已消耗：{factor_name}（再跑=作弊）")
