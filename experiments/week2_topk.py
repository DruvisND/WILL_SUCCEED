# -*- coding: utf-8 -*-
"""
Week2 TopK 选择实验
维度：TopK 20, 50, 100, 200（其余固定：形成期6月, Rank, 月度再平衡）
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
    topk_list = [20, 50, 100, 200]
    all_series = {}
    all_metrics = []

    for k in topk_list:
        try:
            ret, _ = run_reversal(
                formation_months=6,
                standardization="rank",
                topk=k,
                rebalance_freq="M",
            )
            ret = ret.dropna()
            all_series[f"TopK={k}"] = ret
            m = metrics(ret)
            m["topk"] = k
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
    ax.set_title("Week2 TopK 对比（反转策略）")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "week2_topk.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"已保存: {out}")

    mdf = pd.DataFrame(all_metrics)
    print("\nTopK 实验指标:")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
