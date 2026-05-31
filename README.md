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

# Web UI（后台启动，输出静默）
uv run streamlit run quant/web/app.py --server.headless true > /dev/null 2>&1 &
# 停止
kill $!
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
| `quant/web/` | Streamlit Web UI（多页漏斗流程：首页总览 + 6 个步骤页 + 共用组件） |
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
# 启动（streamlit 自动发现 pages/ 目录生成多页导航）
uv run streamlit run quant/web/app.py --server.headless true

# 后台启动（终端可继续用）
uv run streamlit run quant/web/app.py --server.headless true > /tmp/st.log 2>&1 &
# 停止
kill $!
```

多页漏斗流程，像筛子一样：想法多 → 层层筛 → 活下来的少数 → 组合产出。

| 页面 | 步骤 | 功能 |
|---|---|---|
| 🏠 总览 | — | 首页，漏斗图 + 6 张入口卡 |
| ① 因子扫描 | 粗筛 | 一键体检全库因子，按 ICIR 排序 |
| ② 因子工坊 | 精测 | 单因子深度体检，参数随因子显隐，hover 出术语白话 |
| ③ 多因子合成 | 增强 | 加权合成 + 共线性预警 + 4 个 metric 卡绩效概览 |
| ④ holdout 闸门 | 终验锁定 | 定稿因子最终验证（仅一次，跑后锁定），红色风险条 |
| ⑤ 选股器 | 产出 | 多因子加权打分 → 今日买入池，条形图可视化 |
| 📒 台账 | — | 全程试验记录（工坊、回测、holdout 自动写入） |

每页顶部有「第 N 步 / 5 + 上下步链接 + 这步干啥」引导，参数 hover 显示术语白话解释。

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
uv run pytest -q        # 全量（130 用例）
uv run ruff check .     # lint
```
