# Quant 选股器 + 报告通俗化 + 批量体检 设计文档

> 状态：已与用户确认设计，待写实施计划（writing-plans）。
> 前置：Plan 2b（factors-web）Task 9–13 收尾由用户完成（P0），本设计的 P1–P3 建立其上。

## 背景

Plan 1（数据/单因子检验）、Plan 2a（回测/抗过拟合/holdout 闸门）、Plan 2b 的 Task 1–8（4 个价量因子族 + registry + sectors + 行业中性化 + parquet 缓存）已完成。Plan 2b 的 web 层（Task 9 提取 `run_backtest_report`、Task 10 CLI 接注册表 + `--neutralize`、Task 11–13 Streamlit 四页）尚未完成，由用户先行收尾（记为 **P0**）。

用户提出 4 类新需求：
1. 因子选股加到页面（mockup 已给：多因子量化选股器）
2. 输出报告不通俗易懂
3. 不能批量跑全部因子体检
4. 还有其它功能没有

经讨论确定本次范围为 **P1 选股器 + P2 报告通俗化 + P3 批量体检**；P4 基本面接入与 P5 锦上添花项暂不做。

## 关键数据约束

`data_lake/us/daily` 仅含 `ticker/date/open/high/low/close/volume/instrument_id/adj_factor`——**纯价量，无基本面**。现有因子库 6 个全为价量因子：`momentum / ma_bias / short_reversal / volatility / rel_volume / amihud`。

mockup 中的"估值因子""质量因子"需 PE/PB、ROE 等财报数据，当前 data_lake 不具备。**决策：选股器先用价量因子滑块（出路 A）**，基本面接入（估值/质量因子）列为后续独立子项目（P4），本次不做。选股器需**模块化**设计，使将来基本面因子注册进 registry 后零改动复用。

## 锁定决策

| 决策项 | 选定 |
|---|---|
| 选股器因子来源 | 现有价量因子；通过 `factor.registry` 动态取，不硬编码 |
| 估值/质量因子 | 本次不做，后续基本面子项目（P4） |
| 综合得分归一 | 截面百分位排名 × 100（0–100，抗极值） |
| 买入池切分 | top-N（默认 top 20%，可手填 N） |
| 因子权重 | 手动滑块，归一到 sum=1，传 `combine_score` 的 weights（新增 `weighting="manual"` 语义） |
| 选股器与 holdout/DSR | 选股用最新截面（生产用途），不写 trial 台账、不算 DSR；界面提示"前提是该组合已通过验证" |
| 批量体检与 DSR | 写独立 `scan_ledger.jsonl`，**不污染** DSR 主 trial 台账 |
| 报告通俗化形式 | 现有报告卡顶部加白话结论段 + 红绿灯一句话；保留原指标表 |

## 架构与模块边界

新增纯逻辑模块 `quant/select/screen.py`（不依赖 streamlit，可单测）。**不硬编码因子名**，全程走 `factor.registry`：

```
select/screen.py
  screen(names, weights, top_n, neutralize, mode, market) -> SelectionResult
    1. 取最新截面 panels（复用 data/panel + data/holdout）
    2. 逐因子 registry.compute_factor（默认参数）→ 因子矩阵（复用 factor/registry）
    3. zscore_factors + combine_score(zf, weights)（复用 combine/synth）
    4. 取最后一行（最新交易日）综合分 → 截面 percentile rank × 100
    5. 按分降序 rank → 前 top_n 进买入池(buy)，其余备选池(candidate)
    6. join load_sectors → 行业标签（复用 data/sectors）
    返回 SelectionResult：rows[instrument_id/ticker, score(0-100), rank, zone, sector]
```

复用点全部现成：`factor.registry`、`combine.synth`（`zscore_factors`/`combine_score`）、`data.sectors`、`data.panel`、`data.holdout`、`process.neutralize`（可选中性化）。选股器本质 = "combine 只取最新截面 + 归一化 0-100 + 切买入/备选池"，新增代码量小。

**未来复用兑现点**：基本面来了，估值/质量因子注册进 registry → `factor_names()` 自动多出两项 → 选股器滑块自动出现，选股器代码**一行不改**。

依赖方向：`web → select → {factor.registry, combine, data, process}`，无环。

## P1 选股器（mockup 落地）

| mockup 元素 | 实现 |
|---|---|
| 综合得分 0–100 | 截面百分位排名 × 100 |
| 买入池（绿）/ 备选池（蓝） | `top_n` 切分 |
| 因子权重滑块 + 总权重显示 100% | 滑块值归一 sum=1，传 `combine_score` weights |
| 行业标签 | `load_sectors` join |
| ★ 标记 | 买入池条目标星 |
| 水平条形图 | Streamlit `altair` 水平 bar，按 zone 着色（buy=绿 / candidate=蓝） |

- 滑块因子集 = `factor_names()` 全量（现 6 个价量因子）。
- 每因子取默认参数；滑块只调**权重**（与 mockup 一致，不暴露 lookback/window 等）。
- 选股使用最新截面（`mode="full"`，含 holdout/实盘期）；属生产用途，非研究回测——不写 trial 台账、不算 DSR。
- 界面提示：选股前提是该因子组合已通过研究验证（仅提示，不强制阻断）。

**新增文件/改动：**
- `quant/select/__init__.py`
- `quant/select/screen.py`：`SelectionResult` dataclass + `screen(...)`
- `quant/combine/synth.py`：**无需改**——`combine_score(factors, weights)` 已直接吃 weights dict（已核实）。滑块归一化（sum=1）放在 `select/screen.py` 内做
- `quant/web/viewmodel.py`：加 `selector(...)` 纯函数（返回 SelectionResult 或其 dict）
- `quant/web/app.py`：第五页"选股器"——权重滑块 + top_n 输入 + 中性化勾选 + altair 水平条形图 + 行业标签
- `tests/test_screen.py`：合成 fixture 验证排序/归一 0-100/切池/行业 join

## P2 报告通俗化

`quant/report/scorecard.py`（FactorReport）与 `quant/report/backtest_card.py`（BacktestReport）各加白话结论段，渲染在报告卡顶部：

- `_plain_verdict()`：依红绿灯综合判定输出一段人话——这因子能不能用、为什么（如"IC-IR 0.6 偏强，分位单调，多空年化为正 → 可进入候选；但换手 15% 偏高，注意成本"）。
- 关键术语加一行括注解释（IC-IR=预测力稳定度、DSR=剔除多重检验后的真 Sharpe 等）。
- **保留**原有指标表，不删。

**改动文件：** `quant/report/scorecard.py`、`quant/report/backtest_card.py`、对应测试 `tests/test_scorecard.py`、`tests/test_backtest_card.py`（断言白话段含关键词）。

## P3 批量体检

`quant factor test-all`：循环 `factor_names()` 全因子，对每个跑单因子体检（IC/Sharpe 等），汇成排行榜表（按 IC-IR 或 Sharpe 排序）。

- 写独立 `scan_ledger.jsonl`（与 DSR 主 `ledger.jsonl` 分离）；批量扫描**不触** DSR 主台账，避免撑大 trial 数惩罚后续定稿因子。
- web 第六页：排行榜表 + 一键批量跑。

**新增/改动文件：**
- `quant/cli.py`：加 `test_all_cmd`（`quant factor test-all`）
- `quant/web/viewmodel.py`：加 `leaderboard(...)`
- `quant/web/app.py`：第六页"批量排行榜"
- `tests/test_cli_testall.py`、`tests/test_viewmodel.py`（追加 leaderboard 用例）

## 明确不做（本次范围外）

- 基本面数据接入（futu 财报 → data_lake 基本面集）、估值/质量因子（P4，后续独立子项目）
- 因子正交化、PBO、风险模型（沿用 Plan 2b v2 边界）
- 选股器实盘下单 / 导出 broker 格式（futu 交易另议）
- 首页研究看板、IC 衰减曲线、相关热力图（P5，可选锦上添花）

## 阶段依赖

```
P0（用户收尾 Plan 2b web Task 9–13）
  └─ P1 选股器（第五页）
  └─ P3 批量体检（第六页）
P2 报告通俗化（独立，不依赖 P0/web）
```

P2 不依赖 web，可与 P0 并行。P1/P3 依赖 P0 的 web 壳与 registry-backed CLI。

## 验证标准

- P1：合成 fixture 下 `screen` 返回行按综合分降序、分在 0–100、top_n 切池正确、行业标签正确；web 选股器页冒烟可点出条形图。
- P2：报告卡含白话结论段关键词；强因子出"可用"措辞、弱因子出"不建议"措辞。
- P3：`test-all` 输出全因子排行榜；scan_ledger 落地且 DSR 主台账 trial 数不变。
</content>
</invoke>
