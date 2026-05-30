# Quant 因子研究流程 MVP 设计文档

- 日期：2026-05-30
- 项目：`/Users/xiaohong/Project/Quant`
- 状态：设计已确认，待写实现计划

## 1. 背景与定位

个人开发者要一套类似专业量化团队的因子研究流程。专业团队完整链路为：

> 经济假设 → 单因子检验 → 因子处理 → 因子组合 → 组合构建 → 真实回测 → 抗过拟合验证 → holdout/纸交易 → 监控衰减

本 MVP **聚焦阶段 2–7 的"因子研究"段**，并扩展到多因子合成选股。终点交付物是**因子体检报告 + 多因子选股回测**，不做实盘/纸交易/衰减监控（留 v2/v3）。

### 1.1 与专业团队流程的对照（诚实定位）

| 专业团队阶段 | 本 MVP | 说明 |
|---|---|---|
| 经济假设 | 人脑提，不工具化 | MVP 不需要 |
| 单因子检验（IC/分位/衰减） | ✅ 完整 | 核心 |
| 因子处理（去极值/标准化/中性化） | ✅ | 核心 |
| 多因子合成（加权/共线性） | ✅ 等权+IC加权 | 正交化留 v2 |
| 组合构建（优化器） | 等权近似 | MVP 够用 |
| 真实回测（成本/滑点） | ✅ 简版 | 冲击成本简化 |
| 抗过拟合（WF/DSR/PBO） | ✅ WF+DSR | PBO 留 v2 |
| holdout 纪律 | ✅ 物理隔离 | 核心 |
| 纸交易/实盘 | ❌ | v2 边界 |
| 衰减监控（实盘 vs 回测） | ❌ | v3，需长期运行 |
| 风险模型（Barra 类） | ❌ | 个人开发太重 |

保留了"科学性"三根支柱：**样本外验证、试验次数记账、holdout 物理隔离** —— 这是个人开发者从"调参调到 Sharpe 好看"升级到"有统计纪律"的关键一跳。

## 2. 关键决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 市场/品种 | 美股日频 | 数据已在群晖 + TrendSpec data_lake |
| 与 TrendSpec 关系 | 新 Quant 项目，复用其 data_lake 数据层 | 职责清晰，不动旧码 |
| 架构方案 | 方案 A：独立向量化横截面引擎 | 因子研究天然向量化；TrendSpec 事件驱动逐标的引擎形状不匹配 |
| 因子维度 | 单因子体检 + 多因子合成 | 用户要多因子选股 |
| 基本面数据 | **v1 只做价量因子** | yfinance 财报非 PIT，回测会产生未来函数；Value/Quality 待接入 PIT 源 |
| Web | Streamlit | 个人开发最快，图表内置；薄展示层调用核心库 |
| 性能 | 全程宽矩阵 numpy + 缓存 + 并行接口预留 | 见 §5 |

### 2.1 为何 v1 不做 Value/Quality

yfinance 财报数据存在三重致命问题，直接回测会让 holdout/抗过拟合纪律失效：

1. **无 PIT**：给的是今天的/重述后的财报，不是"历史某天市场已知"的值 → 严重未来函数。
2. **历史太短**：基本面仅约 4–5 年，配不上 ~16 年价格回测。
3. **幸存者偏差**：退市公司拿不到财报。

价量因子从 OHLCV 派生，全程 PIT 干净，可回测全历史。Value/Quality 等接入 PIT 源（Sharadar SF1 或自建 SEC EDGAR）后再加。

## 3. 数据基础

复用 TrendSpec `data_lake`（只读）。可用数据：

| 数据 | 来源 | 用途 |
|---|---|---|
| OHLCV 日线（已复权，~2010–2026） | `data_lake/us/daily` | 所有价量因子 |
| OHLCV 周线 | `data_lake/us/weekly` | 周频因子（可选） |
| GICS 行业 | `data_lake/us/sectors` | 行业中性化、行业内排名 |
| SP500 成分变动 | `data_lake/us/components` | PIT universe，无生存者偏差 |
| 指数价 | `index_prices` | Beta、特质波动基准 |

默认股票池：**SP500 PIT 成分**（含当时已退市股，避免生存者偏差）。

## 4. 模块架构与数据契约

横截面因子研究的核心数据形态：**宽矩阵 `[date × instrument_id]`**。所有横截面运算（排序、分位、IC）都是矩阵行操作，向量化、快。

```
quant/
  data/          # 数据访问层 — 读 TrendSpec data_lake，组装 PIT 面板
    panel.py     #   load_prices() → 宽矩阵; PIT universe 过滤; holdout 物理截断
    returns.py   #   forward_returns(h) → 前瞻收益矩阵（预算一次、缓存）
  factor/        # 因子定义 + 计算
    base.py      #   Factor 协议: compute(panel) → 因子值宽矩阵
    library/     #   单文件一因子（见 §4.2）
    cache.py     #   因子矩阵 parquet 缓存（key=因子+参数+池+区间）
  process/       # 因子处理（阶段3）
    pipeline.py  #   winsorize → standardize → neutralize 可组合链
  eval/          # 单因子检验（阶段2）
    ic.py        #   IC/RankIC 时序、IR、t值、衰减曲线
    quantiles.py #   分位收益、多空价差、换手率
  combine/       # 多因子合成（阶段4）
    composite.py #   标准化 → 等权/IC加权 → 合成总分；共线性检查
  backtest/      # 真实回测（阶段6）
    portfolio.py #   分位/多空/多因子组合，向量化，含成本/滑点
    metrics.py   #   Sharpe/回撤/Calmar/月胜率
  validate/      # 抗过拟合（阶段7）
    walkforward.py
    deflated.py  #   deflated Sharpe、试验次数台账
    holdout.py   #   时间轴锁定，研究期物理隔离
  report/        # 体检报告
    scorecard.py #   因子 → Markdown/HTML 报告卡
  web/
    app.py       #   Streamlit 三页（薄层，调用核心库）
  cli.py         # quant factor test <name> ... / quant combine ...
```

### 4.1 模块间契约（松耦合，任一模块可独立换实现/优化）

| 契约 | 形态 |
|---|---|
| 价格面板 | `DataFrame[date × instrument_id]`（close 等宽矩阵） |
| 因子值 | `DataFrame[date × instrument_id]`（原始/处理后） |
| 前瞻收益 | `DataFrame[date × instrument_id]`，按 horizon h |
| 单因子结果 | dataclass `FactorReport`（IC 序列/分位表/回测 metrics/试验台账） |
| 多因子结果 | dataclass `CompositeReport`（成分权重/共线性矩阵/合成回测） |

每个模块输入输出都是这几个结构，内部实现（pandas/numpy/polars）可独立替换。

### 4.2 v1 出厂因子库（全价量，可回测全历史）

部分逻辑从 TrendSpec 移植（momentum/volatility/turnover/ma_bias/sector_*），不重写。

| 族 | 因子 | 逻辑 |
|---|---|---|
| 动量 | 12-1 月动量、6-1 月、52 周高点距离 | 强者恒强 |
| 反转 | 1 月短期反转 | 超跌反弹 |
| 均线 | MA 乖离（价/MA−1）、距 200 日线、MA 多头排列度、MACD | 趋势位置 |
| 波动 | 已实现波动率、特质波动（剔指数） | 低波异象 |
| 流动性 | 换手率、Amihud 非流动性（\|涨幅\|/成交额） | 流动性溢价 |
| 量能 | 量趋势、放量异动、OBV 类 | 资金动向 |
| 趋势质量 | 趋势 R²、ADX 类 | 趋势纯度 |
| Beta/偏度 | 对指数 Beta、收益偏度（彩票效应） | 风险定价 |
| 规模代理 | 价格×均量 的市值粗代理 | 小盘溢价（粗，非真实市值） |
| 行业相对 | GICS 行业内排名、行业去均值 | 中性化（已有） |

## 5. 性能框架

SP500 规模约 500 名 × 4000 日 ≈ 200 万格（Float64 ~16MB），单次 IC/分位 ~100ms 级。瓶颈在**参数扫描 × walk-forward 多窗口**的放大。三层策略：

**① 数据形态 —— 全程宽矩阵 numpy，零 Python 日期循环**
- IC 时序 = 因子矩阵与前瞻收益矩阵逐行 rank 相关，一次算完整条时序。
- 分位组合 = 按行 rank 分桶 → mask 矩阵 × 收益矩阵。
- 换手 = 持仓 mask 逐行差分。
- 原则：能写成 numpy 2D 数组操作的，绝不写 `for date`。

**② 缓存 —— 算过的不重算**

| 缓存物 | key | 存储 |
|---|---|---|
| 前瞻收益矩阵 | 池+区间+horizon | parquet，一次算全程复用 |
| 因子原始值矩阵 | 因子+参数+池+区间 | parquet（借鉴 TrendSpec factor_cache） |
| 价格面板 | 池+区间 | 进程内 LRU + parquet |

**③ 并行 —— 模块边界预留，v1 不强做**
- 扫参/walk-forward 窗口接口设计为纯函数 `(params) → FactorReport`，v2 直接 joblib/进程池 map，内核不动。

**选型**：核心矩阵数学 numpy + pandas；parquet IO 成瓶颈再换 polars（契约是 DataFrame，可热插拔）。

**性能验收基线**：
- 单因子全检验（IC+分位+回测，4000 日 SP500）< 2 秒
- 100 组参数扫描 < 30 秒（串行）

### 5.1 性能演进路线（v1 不集成，触发条件达到再上）

| 框架 | 触发条件 | 价值 |
|---|---|---|
| VectorBT (Numba) | 需路径依赖逻辑（动态止损/加仓/撮合） | C 级加速；但属逐标的信号模拟，与本系统横截面定位不同，复用价值低 |
| Polars | Pandas 索引对齐/IO 成瓶颈 | Rust 多线程、零拷贝、Lazy 计算图 |
| Numba | 出现必须按时间 for 循环的微观逻辑 | 填补向量化盲区，避免回退原生循环 |
| Xarray | 数据演进为 `[日期×资产×因子×参数]` 4D | 标签化多维索引比 MultiIndex 清晰 |
| CuPy/JAX | 池扩到 Russell3000 + 扫参上万组 | GPU 百倍加速 |

`backtest/portfolio.py` 接口保持干净（持仓 mask 矩阵 + 收益矩阵 → 净值），将来塞 Numba kernel 时内核不动。

## 6. 单因子检验方法学（阶段 2）

### 6.1 预测力 —— IC 家族

| 指标 | 定义 | 看什么 |
|---|---|---|
| IC（每日） | 因子值 vs 前瞻收益 Pearson | 截面预测力 |
| RankIC（每日） | Spearman（抗异常值，主用） | 单调预测力 |
| IC 均值 / IC_IR | mean(IC) / std(IC) | 稳定性，IR>0.5 算好 |
| IC t 值 / p 值 | mean/std×√N | 统计显著性 |
| IC 衰减曲线 | horizon=1/5/20/60 的 IC | 因子半衰期、最优调仓频率 |

### 6.2 单调性 —— 分位收益

- 每日按因子值分 5 档，算各档前瞻收益。
- 输出：分位累计净值曲线、Q5−Q1 多空价差年化、单调性检验（秩相关）。

### 6.3 成本与可行性

- 换手率（分位组合逐期持仓变化）→ 估算成本拖累。
- 多空价差扣成本后是否仍存在。

### 6.4 因子健康

- 覆盖度（每日有效值占比）、自相关（因子稳定性）、极端值占比。

### 6.5 因子处理（阶段 3，作为可选前置链）

默认开：winsorize（1%/99%）+ zscore 标准化。行业中性化（GICS 去均值）可选。

## 7. 多因子合成（阶段 4）

- 每因子先标准化（zscore）。
- 加权：等权 / IC 加权（按历史 IC 大小）两种。
- 合成总分 → 选股。
- 共线性检查：因子两两相关矩阵，高相关预警。
- 正交化（去共线，Gram-Schmidt/回归取残差）：接口预留，v2 实现。

## 8. 真实回测 + 抗过拟合 + holdout（阶段 6–7）

### 8.1 真实回测

- 默认组合：Q5 多头（或 Q5−Q1 多空），等权，月度调仓（可配周频）。
- 成本：单边 bps 可配（默认 10），按换手扣。
- 输出：净值、年化、Sharpe、最大回撤、Calmar、月胜率。
- 与阶段 2 分位收益区别：含成本 + 真实调仓节奏 + 权重约束，回答"能不能交易"。

### 8.2 抗过拟合

| 武器 | 防什么 | 实现 |
|---|---|---|
| Walk-forward | 参数只在样本内调，样本外验证 | 滚动切窗，IS 调参 → OOS 记录，看 OOS 衰减 |
| Deflated Sharpe (DSR) | 多次试验后 Sharpe 虚高 | 按试验次数 N + 偏度峰度折减，出真实显著性 |
| 试验次数台账 | 暗中多重检验 | ledger 记录每次扫参组数 → 喂给 DSR |
| OOS/IS 比值 | 过拟合程度 | OOS Sharpe / IS Sharpe，<0.5 报警 |

纪律：**扫了多少组参数必须记账**，DSR 用该 N 折减。借鉴 TrendSpec 的 `ledger.jsonl`。

### 8.3 Holdout 物理隔离

- 数据时间轴切三段：**训练/验证 | holdout（锁）**。默认 holdout = 最后 2 年。
- 研究期（IC 检验、扫参、walk-forward）**代码层面拿不到 holdout 数据**：`data` 层按 mode 过滤，research mode 物理截断。
- 因子定稿后，**仅一次**在 holdout 跑最终报告。再回去调 = 作弊，报告卡标记"holdout 已消耗"。

## 9. 交付物与界面

### 9.1 报告卡（CLI + web 共用）

- 因子 → `FactorReport` → 渲染 Markdown/HTML，存 `quant_out/reports/<factor>-<ts>.html`。
- 内容：红绿灯结论（IC-IR / 单调性 / 成本后多空 / DSR / holdout 状态）+ 图表（IC 时序、衰减、分位净值、多空回撤）+ 参数与试验台账。

### 9.2 Streamlit web 四页

1. **因子工坊**：选因子 → 调参 → 选处理链 → 跑检验 → 内嵌报告卡。
2. **多因子合成**：选多个因子 → 选加权法 → 看共线性 → 合成选股回测。
3. **holdout 闸门**：定稿因子，一键（确认弹窗）在 holdout 跑最终验证，标记已消耗。
4. **历史**：列过往报告卡，对比 IC-IR。

## 10. 测试策略

复用 TrendSpec 习惯：SQLite/小 parquet fixture，不 mock 数据模型，用合成数据构造"已知答案"做金标准断言。

| 层 | 测什么 |
|---|---|
| data | PIT 过滤正确、holdout 物理截断生效（研究期取不到锁定数据） |
| factor | 因子矩阵形状/数值；移植因子与 TrendSpec 对账 |
| process | winsorize/zscore/中性化后分布正确 |
| eval | 纯随机因子 IC≈0；构造单调因子分位单调 |
| combine | 等权/IC 加权权重正确、共线性矩阵对 |
| backtest | 零成本对账、换手计算、净值复现 |
| validate | DSR 在已知输入下数值正确、ledger 计数对 |

## 11. 构建顺序（每步可验收）

```
1. data 层（读 data_lake + PIT + holdout 截断） → 验：面板正确、锁定生效
2. factor base + 价量因子库（动量/均线/波动/流动性...） → 验：因子矩阵+TrendSpec 对账
3. process 链（winsorize/zscore/中性化） → 验：处理后分布
4. eval（IC/分位/换手/衰减） → 验：合成因子金标准
5. combine（等权/IC加权+共线性） → 验：权重与相关矩阵
6. backtest（成本组合+metrics） → 验：零成本对账
7. validate（walkforward/DSR/ledger/holdout） → 验：数值正确、holdout 仅一次
8. report 卡（Markdown/HTML） → 验：渲染出图
9. Streamlit web 四页 → 验：端到端跑通单因子+多因子
10. 性能基线测试（<2s / 扫参<30s） → 验：达标
```

## 12. 明确不做（YAGNI 边界）

- 基本面因子（Value/Quality）—— 待 PIT 源
- 多因子正交化、PBO、风险模型（Barra）—— v2
- 纸交易/实盘、衰减监控 —— v2/v3
- 组合优化器（均值方差/风险预算）—— 等权近似够用
- 实时数据/盘中 —— 仅日频
