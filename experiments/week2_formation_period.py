# -*- coding: utf-8 -*-
"""
Week2 形成期探索实验
维度：形成期 1, 3, 6, 12 月（其余参数固定：Rank, TopK=50, 月度再平衡）
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reversal_core import run_reversal, metrics, CUTOFF_YEAR
from plot_style import apply_cn_font

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    apply_cn_font()
    formation_months_list = [1, 3, 6, 12]
    all_series = {}
    all_metrics = []

    for fm in formation_months_list:
        try:
            ret, _ = run_reversal(
                formation_months=fm,
                standardization="rank",
                topk=50,
                rebalance_freq="M",
            )
            ret = ret.dropna()
            all_series[f"形成期{fm}月"] = ret
            m = metrics(ret)
            m["formation_months"] = fm
            all_metrics.append(m)
        except FileNotFoundError as e:
            print(e)
            return

    if not all_series:
        return

    # 累计收益图
    cum = pd.DataFrame({k: (1 + v).cumprod() for k, v in all_series.items()})
    cum = cum.dropna(how="all")
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col)
    ax.set_title("Week2 形成期对比（反转策略）")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "week2_formation_period.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"已保存: {out}")

    # 打印指标
    mdf = pd.DataFrame(all_metrics)
    print("\n形成期实验指标:")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
