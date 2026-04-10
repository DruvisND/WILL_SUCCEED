# Week02 四维实验：单因子设计与产出规范

来源：`https://quant-suibe.netlify.app/week02`

---

## 1. 实验设计原则：单因子（One-At-a-Time）

每次只改 1 个参数，其余保持基线不变，否则无法解释“是谁导致了结果变化”。

推荐基线（课程示例）：

```python
baseline_config = {
  "formation_period": 1,
  "topk": 50,
  "standardization": "raw",
  "rebalance": "M",
  "weighting": "EW",
}
```

---

## 2. 维度 1：形成期 K（1/3/6/12）

### 信号
\[
s_{i,t} = -\\sum_{j=1}^{K} r_{i,t-j}
\]

### 实验固定
- TopK=50
- 标准化=Raw
- 再平衡=M
- 权重=等权

### 输出建议
- 表格：K vs 年化收益/夏普/最大回撤/年化换手
- 图：累计净值曲线对比 `results/week2_formation_period.png`

---

## 3. 维度 2：标准化（Raw/Rank/Z-score/Winsor）

### 方法
- Raw：不处理
- Rank：截面 `rank(pct=True)`
- Z-score：截面标准化
- Winsor：截面分位数截尾/压缩（课程示例 5%-95%）

### 实验固定
- 形成期 K=6（通常以维度1最优值）
- TopK=50
- 再平衡=M

### 课程关键点
对于“只做 TopK 选股”的策略，标准化通常不改变排序，因此对选股影响小（但对后续多因子/优化权重很重要）。

---

## 4. 维度 3：TopK（20/50/100/200）

### 实验固定
- 形成期 K=6
- 标准化=Raw
- 再平衡=M

### 预期权衡
- TopK 小：更集中，可能收益更高但风险更高
- TopK 大：更分散，可能稀释 alpha，但回撤与波动可能下降

### 输出建议
- 图：`results/week2_topk.png`
- 表：TopK vs 年化收益/夏普/最大回撤/年化换手

---

## 5. 维度 4：再平衡频率（M vs Q）

### 实验固定
- 形成期 K=6
- TopK=50
- 标准化=Raw

### 季度再平衡实现要点
- 只在 3/6/9/12 月更新权重，其余月份保持上一期权重不变

### 输出建议
- 图：`results/week2_rebalance.png`
- 表：M vs Q 的收益/夏普/回撤/换手对比

---

## 6. 产出与日志规范（Week02 4.8）

### 代码文件（对应脚本）
- `experiments/week2_formation_period.py`
- `experiments/week2_standardization.py`
- `experiments/week2_topk.py`
- `experiments/week2_rebalance.py`
- 或一键：`experiments/week2_all.py`

### 实验记录（exp_log.md）
每条记录至少包含：
- 日期
- 配置（四个维度的取值 + 样本期 + 是否含成本）
- 结果（年化收益、夏普、最大回撤、换手率等）
- 发现（简短解释/假设）

### 最优参数组合
以夏普或综合指标选择“最优组合”，并写出理由（信号强度 vs 换手 vs 成本敏感）。

