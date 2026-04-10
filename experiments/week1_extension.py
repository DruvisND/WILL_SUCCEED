# -*- coding: utf-8 -*-
"""
Week1 拓展任务（可选）：
1) 月度收益直方图
2) 回撤曲线
3) 样本内/样本外对比（2005-2020 vs 2021-2025）
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_monthly_returns
from src.signal import make_signal_reversal_1month
from src.portfolio import build_topk_portfolio, backtest_gross
from src.evaluation import evaluate

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    returns = load_monthly_returns()
    signal = make_signal_reversal_1month(returns)
    weights = build_topk_portfolio(signal, topk=50, rebalance="M", weighting="EW")
    gross = backtest_gross(weights, returns).dropna()

    # 1) 收益直方图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gross.values, bins=40, alpha=0.8)
    ax.set_title("Week1 月度收益率分布")
    ax.set_xlabel("月度收益率")
    ax.set_ylabel("频数")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_hist = os.path.join(RESULTS_DIR, "week1_return_hist.png")
    fig.savefig(out_hist, dpi=150)
    plt.close()

    # 2) 回撤曲线
    nav = (1 + gross).cumprod()
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(drawdown.index, drawdown.values)
    ax.set_title("Week1 回撤曲线")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dd = os.path.join(RESULTS_DIR, "week1_drawdown.png")
    fig.savefig(out_dd, dpi=150)
    plt.close()

    # 3) 样本内/样本外对比
    ins = gross[gross.index.year <= 2020]
    oos = gross[gross.index.year >= 2021]
    m_ins = evaluate(ins, weights.loc[ins.index] if not ins.empty else None)
    m_oos = evaluate(oos, weights.loc[oos.index] if not oos.empty else None)
    compare = pd.DataFrame([{"sample": "in_sample_2005_2020", **m_ins}, {"sample": "out_of_sample_2021_2025", **m_oos}])
    out_csv = os.path.join(RESULTS_DIR, "week1_oos_compare.csv")
    compare.to_csv(out_csv, index=False)

    print(f"saved: {out_hist}")
    print(f"saved: {out_dd}")
    print(f"saved: {out_csv}")
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()

