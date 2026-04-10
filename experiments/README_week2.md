# Week 2 反转策略实验

## 数据

- 将 **week2-data.xlsx** 放在项目根目录 `d:\Quant\` 或 `d:\Quant\data\`。
- 数据需包含列：**日期**（date/trade_date）、**股票代码**（code/ts_code/symbol）、**收益率**（return）或**收盘价**（close，将自动计算收益）。
- 策略**不使用 2026 年数据**（无未来函数）。

## 运行方式（Python 3.13）

```bash
# 进入项目根目录
cd d:\Quant

# 安装依赖
pip install -r requirements.txt

# 运行单个实验
python experiments/week2_formation_period.py   # 形成期对比 → results/week2_formation_period.png
python experiments/week2_standardization.py    # 标准化对比 → results/week2_standardization.png
python experiments/week2_topk.py               # TopK 对比   → results/week2_topk.png
python experiments/week2_rebalance.py          # 再平衡对比 → results/week2_rebalance.png

# 或一键运行全部
python experiments/week2_all.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `experiments/week2_formation_period.py` | 形成期 1/3/6/12 月 |
| `experiments/week2_standardization.py` | Raw / Rank / Z-score / Winsorization |
| `experiments/week2_topk.py` | TopK 20 / 50 / 100 / 200 |
| `experiments/week2_rebalance.py` | 再平衡 月度 M / 季度 Q |
| `experiments/week2_all.py` | 运行上述全部实验 |
| `results/week2_formation_period.png` | 形成期累计收益图 |
| `results/week2_topk.png` | TopK 累计收益图 |
| `results/week2_rebalance.png` | 再平衡频率累计收益图 |
| `exp_log.md` | 实验记录与 Week 2 发现总结 |

运行后请在 `exp_log.md` 中根据控制台输出的指标填写各条记录的“结果”，并更新“最优参数组合”为夏普比率最高的配置。
