"""因子注册表：名字 → 类，统一构建与分发，标注 volume 依赖。"""

import pandas as pd

from quant.factor.base import Factor
from quant.factor.library.amihud import Amihud
from quant.factor.library.ma_bias import MABias
from quant.factor.library.momentum import Momentum
from quant.factor.library.rel_volume import RelativeVolume
from quant.factor.library.short_reversal import ShortReversal
from quant.factor.library.volatility import Volatility

_REGISTRY: dict[str, type[Factor]] = {
    "momentum": Momentum,
    "ma_bias": MABias,
    "short_reversal": ShortReversal,
    "volatility": Volatility,
    "rel_volume": RelativeVolume,
    "amihud": Amihud,
}
_NEEDS_VOLUME = {"rel_volume", "amihud"}


def factor_names() -> list[str]:
    """全部已注册因子名。"""
    return list(_REGISTRY)


def make(name: str, **params) -> Factor:
    """按名构建因子实例。未知名抛 KeyError。"""
    if name not in _REGISTRY:
        raise KeyError(f"未知因子：{name}，可选 {factor_names()}")
    return _REGISTRY[name](**params)


def needs_volume(name: str) -> bool:
    """该因子是否需要 volume 输入。"""
    return name in _NEEDS_VOLUME


def compute_factor(
    name: str,
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    **params,
) -> pd.DataFrame:
    """构建并计算因子；需 volume 的因子自动传入。"""
    f = make(name, **params)
    if needs_volume(name):
        return f.compute(close, volume)
    return f.compute(close)
