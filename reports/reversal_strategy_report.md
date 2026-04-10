# 反转策略实验报告（Reversal Strategy）

> 课程作业交付：Week1–Week3  
> 环境：Python 3.13（兼容）  
> 数据：CSMAR `TRD_Mnth`（`week2-data.xlsx`）  

## 1. 研究问题与策略思路

**研究问题**：A 股月度截面是否存在短期反转效应？该效应对参数（形成期、TopK、标准化）是否敏感？加入交易成本后是否仍具可交易性？

**策略思路**：买入过去一段时间表现最差的“输家”，期望其未来一段时间发生均值回归（reversal）。

## 2. 数据与样本说明

- **原始数据**：`week2-data.xlsx`
- **字段**：`Stkcd`（股票代码）、`Trdmnt`（交易月份）、`Mretwd`（月收益，含分红再投资）等
- **样本区间**：2005-01-31 至 2025-12-31（约 252 个月）
- **处理产物**：`data/processed/monthly_returns.csv`（行=月份，列=股票，值=月收益率）

## 3. 防泄露（避免未来函数）说明

本作业严格进行时间对齐，避免使用未来信息：

- **信号只使用历史数据**：在 \(t\) 期形成信号时，仅使用 \(t-1\) 及以前的收益信息（`shift(1)`）。
- **组合收益与权重对齐**：\(t\) 期持仓权重用于计算 \(t+1\) 期收益（`weights.shift(1) * returns`）。
- **样本限制**：策略构建与参数比较均过滤 2026 年及以后数据。

相关实现与验证：

- 代码：`src/signal.py`、`src/portfolio.py`
- 检查：`scripts/check_week1.py`

## 4. 基线策略（Week1）

### 4.1 信号定义

（1 个月反转）

\[
s_{i,t} = -r_{i,t-1}
\]

### 4.2 组合构建

- **选股**：TopK（默认 K=50，选取信号最高的 K 只股票）
- **权重**：等权
- **再平衡**：月度

### 4.3 基线回测结果（Gross，无成本）

基线结果来源于 `run_week1_baseline.py` 的输出与 `exp_log.md` 记录：

- 年化收益：14.77%
- 夏普比率：0.3714
- 最大回撤：86.66%
- 年化换手率：1143.24%

对应图表：`results/week1_nav.png`

## 5. 参数比较（Week2）

要求：不能只展示“最好结果”，需展示比较过程。本报告对应 `exp_log.md` 中 1–14 条实验记录，并给出图表支撑。

### 5.1 形成期（formation period）

对比集合：\(\{1,3,6,12\}\) 月，其他保持固定（标准化=Rank，TopK=50，再平衡=月度）。

图表：`results/week2_formation_period.png`

**结论**：形成期 1M 在该样本下夏普更高（见 `exp_log.md` 1–4）。

### 5.2 信号标准化方法

对比集合：Raw / Rank / Z-score / Winsorization（其他固定：形成期=6M，TopK=50，再平衡=月度）。

图表：`results/week2_standardization.png`

**结论**：TopK 策略主要由排序决定，标准化差异整体不大（见 `exp_log.md` 5–8）。

### 5.3 TopK/选股数量

对比集合：\(\{20,50,100,200\}\)（其他固定：形成期=6M，标准化=Rank，再平衡=月度）。

图表：`results/week2_topk.png`

**结论**：TopK 越大越分散，波动下降、夏普可能上升，但年化收益也可能被摊薄（见 `exp_log.md` 9–12）。

### 5.4 再平衡频率（补充）

对比：月度 M vs 季度 Q（其他固定：形成期=6M，标准化=Rank，TopK=50）。

图表：`results/week2_rebalance.png`

**结论**：月度 Sharpe 略高，但交易频繁成本压力更大（见 `exp_log.md` 13–14）。

## 6. 交易成本与可交易性（Week3）

### 6.1 成本模型

\[
r^{net}_t = r^{gross}_t - Turnover_t \times c,\quad c = \frac{cost\_bps}{10000}
\]

其中

\[
Turnover_t = 0.5 \sum_i |w_{i,t} - w_{i,t-1}|
\]

实现：`src/backtest.py`（`compute_turnover()`、`apply_cost()`）

### 6.2 成本敏感性结果

运行脚本：`experiments/week3_cost_sensitivity.py`  
图表：`results/week3_cost_sensitivity.png`

| 成本(bps) | 年化收益 | 夏普 | 最大回撤 |
|---:|---:|---:|---:|
| 0  | 14.77% | 0.3714 | 86.66% |
| 10 | 13.48% | 0.3408 | 88.00% |
| 20 | 12.20% | 0.3101 | 89.21% |
| 50 | 8.45%  | 0.2182 | 92.16% |

**结论**：策略年化换手率较高（约 1143%），对成本非常敏感；成本从 0 上升到 50 bps 会显著侵蚀收益与夏普，需在执行与约束（再平衡/过滤）上进行改进。

## 7. 主要结论与局限

### 7.1 主要结论（基于证据）

- **反转效应是否存在**：在样本期内，基线策略在无成本下获得正向夏普（证据：`exp_log.md` Week1 + `results/week1_nav.png`）。
- **参数敏感性**：形成期与 TopK 对风险收益结构影响更明显；标准化方法差异较小（证据：Week2 图表与 `exp_log.md` 1–14）。
- **加入成本是否仍成立**：在 10–50 bps 成本下，收益与夏普显著下降（证据：Week3 成本敏感性表 + `results/week3_cost_sensitivity.png`）。

### 7.2 局限与改进方向

- **高换手带来成本与冲击**：需要进一步降低换手或采用更现实的执行模型。
- **现实约束未完全纳入**：T+1、涨跌停、停牌、ST、低流动性等会进一步削弱可实现收益。
- **参数网格的“全局最优”**：当前主要为单因子逐维对照，若要严格求全局最优需进行 4 维网格组合回测。

## 8. 复现方法（命令）

```bash
python scripts/preprocess_week2_data.py
python run_week1_baseline.py
python scripts/check_week1.py
python experiments/week2_all.py
python experiments/week3_cost_sensitivity.py
```

## 9. 关键输出文件（图表/表格）

- `results/week1_nav.png`
- `results/week2_formation_period.png`
- `results/week2_standardization.png`
- `results/week2_topk.png`
- `results/week2_rebalance.png`
- `results/week3_cost_sensitivity.png`

