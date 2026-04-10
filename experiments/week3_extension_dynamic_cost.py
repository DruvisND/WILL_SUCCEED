# -*- coding: utf-8 -*-
"""
Week3 拓展任务（可选）：
1) 动态成本模型：c_t = c0 + alpha * turnover_t
2) 分批交易近似：将成本按批次折减（示例 n=3）
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_monthly_returns
from src.signal import make_signal_reversal_1month
from src.portfolio import build_topk_portfolio, backtest_gross
from src.backtest import compute_turnover
from src.evaluation import evaluate

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def apply_dynamic_cost(gross: pd.Series, turnover: pd.Series, c0_bps: float = 10.0, alpha: float = 5.0) -> pd.Series:
    """
    动态成本：c_t(bps) = c0 + alpha * turnover_t
    net_t = gross_t - turnover_t * c_t/10000
    """
    c_t_bps = c0_bps + alpha * turnover.fillna(0.0)
    return gross - turnover.fillna(0.0) * (c_t_bps / 10000.0)


def apply_staged_trade(gross: pd.Series, turnover: pd.Series, cost_bps: float = 20.0, n_stage: int = 3) -> pd.Series:
    """
    分批交易近似：假设分 n 批降低冲击，成本按 1/sqrt(n) 折减。
    """
    eff_cost = cost_bps / np.sqrt(n_stage)
    return gross - turnover.fillna(0.0) * (eff_cost / 10000.0)


def main():
    returns = load_monthly_returns()
    signal = make_signal_reversal_1month(returns)
    weights = build_topk_portfolio(signal, topk=50, rebalance="M", weighting="EW")
    gross = backtest_gross(weights, returns).dropna()
    turnover = compute_turnover(weights).reindex(gross.index)

    net_fixed20 = gross - turnover.fillna(0.0) * (20 / 10000.0)
    net_dynamic = apply_dynamic_cost(gross, turnover, c0_bps=10, alpha=5)
    net_stage3 = apply_staged_trade(gross, turnover, cost_bps=20, n_stage=3)

    rows = []
    for name, series in [
        ("gross_0bps", gross),
        ("fixed_20bps", net_fixed20),
        ("dynamic_c0_10_alpha_5", net_dynamic),
        ("staged_3_batches_from_20bps", net_stage3),
    ]:
        m = evaluate(series, weights.loc[series.index])
        rows.append({"scenario": name, **m})
    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "week3_dynamic_cost_compare.csv")
    out.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in [
        ("gross_0bps", gross),
        ("fixed_20bps", net_fixed20),
        ("dynamic_cost", net_dynamic),
        ("staged_trade", net_stage3),
    ]:
        nav = (1 + series.fillna(0.0)).cumprod()
        ax.plot(nav.index, nav.values, label=name)
    ax.set_title("Week3 拓展：动态成本与分批交易近似")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "week3_dynamic_cost_compare.png")
    fig.savefig(out_png, dpi=150)
    plt.close()

    print(f"saved: {out_csv}")
    print(f"saved: {out_png}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

