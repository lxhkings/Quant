"""因子抽象基类。因子 = 把价量宽矩阵映射为因子值宽矩阵 [date × instrument_id]。"""

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """所有因子的基类。"""

    name: str

    @abstractmethod
    def compute(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
        """返回与输入同形的因子值矩阵。数据不足处为 NaN。"""
        ...
