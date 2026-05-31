# Quant 因子研究横截面引擎

多因子量化选股、因子体检、回测验证的端到端研究平台。

## 快速开始

```bash
# 安装
uv sync

# 单因子体检
uv run quant factor test momentum

# 回测验证
uv run quant backtest momentum

# 批量体检全部因子
uv run quant factor test-all

# 多因子合成回测
uv run quant combine momentum ma_bias --weighting ic

# Web UI
uv run streamlit run quant/web/app.py
```

## 模块结构

| 模块 | 职责 |
|---|---|
| `quant/factor/` | 因子注册表 + 6 个内置价量因子（momentum, ma_bias, short_reversal, amihud, volatility, rel_volume） |
| `quant/data/` | 数据加载（价量矩阵、行业、收益）、holdout 分割 |
| `quant/process/` | 预处理流水线（winsorize → zscore）、行业中性化 |
| `quant/combine/` | 多因子合成（等权 / IC 加权）、共线性预警 |
| `quant/eval/` | IC 检验、分位收益、多空价差 |
| `quant/backtest/` | 回测引擎 + 绩效指标 |
| `quant/report/` | 因子报告卡（通俗结论 + 红绿灯）、回测报告卡、批量排行榜 |
| `quant/select/` | 多因子选股器（最新截面打分 → 0-100 → 买入/备选池） |
| `quant/validate/` | DSR 抗过拟合、holdout 闸门、试验台账 |
| `quant/web/` | Streamlit Web UI（6 页） |
| `quant/cli.py` | Typer CLI 入口 |

## CLI 命令

| 命令 | 说明 |
|---|---|
| `quant factor test <name>` | 单因子端到端体检（IC + 分位 + 红绿灯报告卡） |
| `quant factor test-all` | 批量体检全部已注册因子，输出 IC-IR 排行榜 |
| `quant backtest <name>` | 单因子回测验证（含成本、DSR） |
| `quant combine <names>` | 多因子合成回测 |
| `quant holdout <name>` | holdout 闸门（仅一次，跑后锁定） |

## Web UI

```bash
uv run streamlit run quant/web/app.py
```

| 页面 | 功能 |
|---|---|
| 因子工坊 | 单因子体检，参数可调 |
| 多因子合成 | 选因子 + 加权法 → 合成回测 |
| holdout 闸门 | 定稿因子最终验证（仅一次） |
| 历史 | 试验台账记录 |
| 选股器 | 多因子加权打分选股，权重滑块 + 条形图 |
| 批量排行榜 | 一键体检全部因子，IC-IR 排序表 |

## 因子扩展

在 `quant/factor/library/` 下新建模块，用 `@register` 装饰器注册即可自动出现在 CLI、Web UI 和选股器中：

```python
from quant.factor.registry import register

@register("my_factor", needs_volume=False)
def my_factor(close, *, window=20):
    ...
```

## 报告通俗化

因子报告卡和回测报告卡均含"通俗结论"段——用白话概括三关（预测力 / 单调性 / 多空、收益 / Sharpe / DSR）是否通过，并附名词解释。

## 测试

```bash
uv run pytest -q        # 全量（128 用例）
uv run ruff check .     # lint
```
