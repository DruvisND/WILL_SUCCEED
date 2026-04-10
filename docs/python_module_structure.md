# Python 代码结构与关键函数签名（Week01–03）

本文件给出课程三周反转策略实现的推荐模块划分、输入输出口径、以及每个模块的关键函数签名（重点：时间对齐与防未来函数）。

---

## 1) 数据层：预处理 + 加载

### `scripts/preprocess_week2_data.py`
**职责**：从 TRD_Mnth（`week2-data.xlsx`）生成宽表月度收益 `data/processed/monthly_returns.csv`

核心步骤：
- 读取 Excel（表头首行）
- 标准化字段名为 `Stkcd, Trdmnt, Mretwd`
- `Trdmnt` → 月末日期 `date`
- pivot 到宽表：index=date, columns=Stkcd, values=Mretwd
- 过滤时间 2005–2025

### `src/data_loader.py`
```python
def load_monthly_returns(csv_path: str | None = None) -> pd.DataFrame:
    """
    Returns
    -------
    returns_df: pd.DataFrame
        index=月末日期（DatetimeIndex）
        columns=股票代码（str）
        values=月度收益率（float）
    """
```

---

## 2) 信号层：反转信号

### `src/signal.py`
```python
def make_signal_reversal_1month(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    1 个月反转信号：s_{i,t} = -r_{i,t-1}
    关键：signal = -returns.shift(1)
    """
```

扩展（Week02 形成期 K）建议签名：
```python
def make_signal_reversal_kmonth(returns_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    K 个月反转信号：s_{i,t} = -sum_{j=1..k} r_{i,t-j}
    推荐：signal = -(returns.rolling(k).sum()).shift(1)
    """
```

标准化（Week02）建议签名：
```python
def standardize_signal(signal_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    method in {'raw','rank','zscore','winsor'}
    截面标准化：对每个日期（行）处理
    """
```

---

## 3) 组合层：TopK + 权重

### `src/portfolio.py`
```python
def build_topk_portfolio(
    signal_df: pd.DataFrame,
    topk: int = 50,
    rebalance: str = "M",
    weighting: str = "EW",
) -> pd.DataFrame:
    """
    Returns
    -------
    weights_df: pd.DataFrame
        index=月末日期
        columns=股票代码
        values=权重（每期权重和=1）
    逻辑：
    - 每期对 signal 排序，选 TopK
    - 等权：1/K；不足K时等权可用股票
    - 再平衡：
      - M：每月更新
      - Q：仅 3/6/9/12 更新，其余沿用上一期
    """
```

---

## 4) 回测层：收益对齐（防未来函数）

### `src/portfolio.py` / `src/backtest.py`
```python
def backtest_gross(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
    """
    Gross 回测（无成本）：
    portfolio_ret_t = sum_i w_{i,t-1} * r_{i,t}
    关键：weights.shift(1) * returns
    """
```

---

## 5) 评估层：指标 + 图

### `src/evaluation.py`
```python
def evaluate(portfolio_returns: pd.Series, weights_df: pd.DataFrame | None = None) -> dict:
    """
    输出：
    - 年化收益
    - 夏普比率
    - 最大回撤
    - 年化换手率（若给 weights_df）
    """

def plot_nav(portfolio_returns: pd.Series, save_path: str | None = None, title: str = ...) -> None:
    """
    输出净值曲线图（初始净值=1）
    """
```

换手率（Week03）口径：
```python
Turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|
AnnualTurnover = mean(Turnover_t) * 12
```

---

## 6) 成本层：Gross → Net（Week03）

建议新增函数（可放在 `src/evaluation.py` 或 `src/cost.py`）：
```python
def apply_cost(gross_returns: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """
    net = gross - turnover * (cost_bps/10000)
    """
```

成本敏感性：对 cost_bps in [0, 10, 20, 50] 批量评估净收益指标。

---

## 7) 运行入口（建议）

### Week01（基线）
- `run_week1_baseline.py`：预处理（若缺）→ 信号 → TopK → 回测 → 评估 → `results/week1_nav.png`

### Week02（参数实验）
- `experiments/week2_all.py`：运行 4 维实验脚本并输出图表与汇总指标

### Week03（成本与证据）
- 建议新增：`experiments/week3_cost_sensitivity.py` + `reports/strategy_card_v1.md`

