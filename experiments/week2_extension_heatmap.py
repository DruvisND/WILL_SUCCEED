# -*- coding: utf-8 -*-
"""
Week2 拓展任务（可选）：
形成期 × TopK 的二维热力图（以夏普为指标）
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reversal_core import run_reversal, metrics
from plot_style import apply_cn_font

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    apply_cn_font()
    formation_list = [1, 3, 6, 12]
    topk_list = [20, 50, 100, 200]
    rows = []
    for fm in formation_list:
        for k in topk_list:
            ret, _ = run_reversal(
                formation_months=fm,
                standardization="rank",
                topk=k,
                rebalance_freq="M",
            )
            m = metrics(ret.dropna())
            rows.append({"formation": fm, "topk": k, "sharpe": m.get("sharpe")})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="formation", columns="topk", values="sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("TopK")
    ax.set_ylabel("形成期（月）")
    ax.set_title("Week2 参数敏感性热力图（夏普）")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "week2_formation_topk_heatmap.png")
    fig.savefig(out_png, dpi=150)
    plt.close()
    out_csv = os.path.join(RESULTS_DIR, "week2_formation_topk_heatmap.csv")
    pivot.to_csv(out_csv)
    print(f"saved: {out_png}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()

