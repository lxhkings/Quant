"""回测报告卡：BacktestReport dataclass + Markdown 渲染（含红绿灯）。"""

from dataclasses import dataclass


def _light(ok: bool) -> str:
    return "🟢" if ok else "🔴"


@dataclass
class BacktestReport:
    factor_name: str
    params: dict
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    monthly_win_rate: float
    avg_turnover: float
    deflated_sharpe: float
    n_trials: int
    holdout_consumed: bool

    def to_markdown(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        ret_ok = self.annual_return > 0
        sharpe_ok = self.sharpe >= 1.0
        dsr_ok = self.deflated_sharpe >= 0.95
        return f"""# 回测体检报告：{self.factor_name}

参数：{params}

## 红绿灯结论

| 关卡 | 指标 | 判定 |
|---|---|---|
| 含成本收益 | 年化 {self.annual_return:.2%} | {_light(ret_ok)} |
| 风险调整 | Sharpe = {self.sharpe:.2f} | {_light(sharpe_ok)} |
| 抗过拟合 | DSR = {self.deflated_sharpe:.3f}（{self.n_trials} 次试验） | {_light(dsr_ok)} |

## 绩效

- 年化收益：{self.annual_return:.2%}
- Sharpe：{self.sharpe:.2f}
- 最大回撤：{self.max_drawdown:.2%}
- Calmar：{self.calmar:.2f}
- 月胜率：{self.monthly_win_rate:.2%}
- 平均换手：{self.avg_turnover:.2%}

## Holdout 状态

- holdout 已消耗：{"是" if self.holdout_consumed else "否"}
"""
