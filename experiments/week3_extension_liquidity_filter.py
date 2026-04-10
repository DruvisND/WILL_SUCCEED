# -*- coding: utf-8 -*-
"""
Week3 拓展任务（可选）：
流动性过滤（基于原始 TRD_Mnth 中成交额字段）
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.portfolio import build_topk_portfolio, backtest_gross
from src.evaluation import evaluate

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_return_and_liquidity() -> tuple[pd.DataFrame, pd.DataFrame]:
    xlsx = os.path.join(PROJECT_ROOT, "week2-data.xlsx")
    if not os.path.isfile(xlsx):
        raise FileNotFoundError("未找到 week2-data.xlsx，无法进行流动性过滤拓展。")
    # 仅读取必要字段，避免全表加载导致过慢
    required = ["Stkcd", "Trdmnt", "Mretwd", "Mnvaltrd"]
    df = pd.read_excel(xlsx, header=0, usecols=required)
    cols = list(df.columns)
    code_col = _find_col(cols, ["Stkcd", "stkcd", "股票代码", "证券代码"])
    date_col = _find_col(cols, ["Trdmnt", "trdmnt", "交易月份", "日期"])
    ret_col = _find_col(cols, ["Mretwd", "mretwd", "月收益率", "月回报率"])
    liq_col = _find_col(cols, ["Mnvaltrd", "mnvaltrd", "月成交额", "成交额"])
    if not all([code_col, date_col, ret_col, liq_col]):
        raise ValueError("原始数据缺少必要字段（code/date/return/liquidity）。")

    data = df[[code_col, date_col, ret_col, liq_col]].copy()
    data.columns = ["code", "date", "ret", "liq"]
    data["code"] = data["code"].astype(str)
    ds = data["date"].astype(str).str.replace("-", "", regex=False).str[:6]
    data["date"] = pd.to_datetime(ds, format="%Y%m") + pd.offsets.MonthEnd(0)
    data = data[data["date"].dt.year <= 2025]
    data["ret"] = pd.to_numeric(data["ret"], errors="coerce")
    data["liq"] = pd.to_numeric(data["liq"], errors="coerce")

    returns = data.pivot_table(index="date", columns="code", values="ret", aggfunc="mean").sort_index()
    liquidity = data.pivot_table(index="date", columns="code", values="liq", aggfunc="mean").sort_index()
    if returns.abs().quantile(0.99).quantile() > 1.0:
        returns = returns / 100.0
    return returns, liquidity


def main():
    returns, liquidity = load_return_and_liquidity()
    signal = -returns.shift(1)
    # 保留每月流动性排名前 70% 股票（去掉后 30%）
    liq_rank = liquidity.rank(axis=1, pct=True)
    signal_filtered = signal.where(liq_rank >= 0.3)

    w_base = build_topk_portfolio(signal, topk=50, rebalance="M", weighting="EW")
    w_flt = build_topk_portfolio(signal_filtered, topk=50, rebalance="M", weighting="EW")

    r_base = backtest_gross(w_base, returns)
    r_flt = backtest_gross(w_flt, returns)

    m_base = evaluate(r_base, w_base)
    m_flt = evaluate(r_flt, w_flt)

    out = pd.DataFrame(
        [
            {"scenario": "baseline_no_liquidity_filter", **m_base},
            {"scenario": "liquidity_filter_top70pct", **m_flt},
        ]
    )
    out_csv = os.path.join(RESULTS_DIR, "week3_liquidity_filter_compare.csv")
    out.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    nav_base = (1 + r_base.fillna(0.0)).cumprod()
    nav_flt = (1 + r_flt.fillna(0.0)).cumprod()
    ax.plot(nav_base.index, nav_base.values, label="baseline")
    ax.plot(nav_flt.index, nav_flt.values, label="liq_filter_top70pct")
    ax.set_title("Week3 拓展：流动性过滤对净值影响")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "week3_liquidity_filter_compare.png")
    fig.savefig(out_png, dpi=150)
    plt.close()

    print(f"saved: {out_csv}")
    print(f"saved: {out_png}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

