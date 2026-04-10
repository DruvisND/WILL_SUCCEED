# -*- coding: utf-8 -*-
"""
Week2 再平衡频率实验
维度：月度 M、季度 Q（其余固定：形成期6月, Rank, TopK=50）
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reversal_core import run_reversal, metrics
from plot_style import apply_cn_font

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    apply_cn_font()
    all_series = {}
    all_metrics = []

    for freq, label in [("M", "月度"), ("Q", "季度")]:
        try:
            ret, _ = run_reversal(
                formation_months=6,
                standardization="rank",
                topk=50,
                rebalance_freq=freq,
            )
            ret = ret.dropna()
            all_series[label] = ret
            m = metrics(ret)
            m["rebalance"] = freq
            all_metrics.append(m)
        except FileNotFoundError as e:
            print(e)
            return

    if not all_series:
        return

    cum = pd.DataFrame({k: (1 + v).cumprod() for k, v in all_series.items()})
    cum = cum.dropna(how="all")
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col)
    ax.set_title("Week2 再平衡频率对比（反转策略）")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "week2_rebalance.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"已保存: {out}")

    mdf = pd.DataFrame(all_metrics)
    print("\n再平衡实验指标:")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
