# -*- coding: utf-8 -*-
"""
Week2 标准化方法对比实验
维度：Raw, Rank, Z-score, Winsorization（其余固定：形成期6月, TopK=50, 月度再平衡）
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

STD_NAMES = {"raw": "Raw", "rank": "Rank", "zscore": "Z-score", "winsor": "Winsorization"}


def main():
    apply_cn_font()
    all_series = {}
    all_metrics = []

    for std in ["raw", "rank", "zscore", "winsor"]:
        try:
            ret, _ = run_reversal(
                formation_months=6,
                standardization=std,
                topk=50,
                rebalance_freq="M",
            )
            ret = ret.dropna()
            all_series[STD_NAMES[std]] = ret
            m = metrics(ret)
            m["standardization"] = std
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
    ax.set_title("Week2 标准化方法对比（反转策略）")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "week2_standardization.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"已保存: {out}")

    mdf = pd.DataFrame(all_metrics)
    print("\n标准化实验指标:")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
