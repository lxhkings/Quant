"""因子报告卡：FactorReport dataclass + Markdown 渲染（含红绿灯）。"""

from dataclasses import dataclass


def _light(ok: bool) -> str:
    return "🟢" if ok else "🔴"


@dataclass
class FactorReport:
    factor_name: str
    params: dict
    ic_mean: float
    ic_ir: float
    t_stat: float
    n: int
    quantile_means: list[float]
    long_short_annual: float
    monotonic: bool
    avg_turnover: float
    holdout_consumed: bool

    def _plain_verdict(self) -> str:
        greens = sum([self.ic_ir >= 0.5, self.monotonic, self.long_short_annual > 0])
        if greens == 3:
            head = "三关全过：预测力稳、分位单调、多空为正，**可进入候选池**。"
        elif greens == 0:
            head = "三关全不过：预测力弱、分位不单调或多空为负，**不建议使用**。"
        else:
            head = f"三关过 {greens}/3，**信号偏弱，需谨慎**。"
        cost = "换手偏高，留意交易成本。" if self.avg_turnover > 0.20 else ""
        return head + cost

    def to_markdown(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        ic_ok = self.ic_ir >= 0.5
        ls_ok = self.long_short_annual > 0
        q_rows = "\n".join(
            f"| Q{i + 1} | {m:.4f} |" for i, m in enumerate(self.quantile_means)
        )
        return f"""# 因子体检报告：{self.factor_name}

参数：{params}

## 通俗结论

{self._plain_verdict()}

> 名词：IC-IR=因子预测力的稳定程度（越高越稳）；多空年化=买高分组、卖低分组的年化收益。

## 红绿灯结论

| 关卡 | 指标 | 判定 |
|---|---|---|
| 预测力 | IC-IR = {self.ic_ir:.3f} | {_light(ic_ok)} |
| 单调性 | {"单调" if self.monotonic else "不单调"} | {_light(self.monotonic)} |
| 多空收益 | 年化 {self.long_short_annual:.2%} | {_light(ls_ok)} |

## IC 统计

- IC 均值：{self.ic_mean:.4f}
- IC-IR：{self.ic_ir:.3f}
- t 值：{self.t_stat:.2f}
- 有效样本数：{self.n}

## 分位平均收益

| 分位 | 平均前瞻收益 |
|---|---|
{q_rows}

## 成本

- 顶档平均换手：{self.avg_turnover:.2%}

## Holdout 状态

- holdout 已消耗：{"是" if self.holdout_consumed else "否"}
"""
