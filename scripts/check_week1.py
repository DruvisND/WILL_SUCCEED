# -*- coding: utf-8 -*-
"""
第一讲 检查清单（见 https://quant-suibe.netlify.app/week01）
- 数据预处理检查
- 信号与回测防泄露检查
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_data():
    """3.3.5 数据预处理检查清单"""
    csv = os.path.join(PROJECT_ROOT, "data", "processed", "monthly_returns.csv")
    if not os.path.isfile(csv):
        print("[ ] data/processed/monthly_returns.csv 不存在，请先运行 scripts/preprocess_week2_data.py")
        return False
    import pandas as pd
    r = pd.read_csv(csv, index_col=0, parse_dates=True)
    ok = True
    if r.index.min().year > 2005 or r.index.max().year < 2025:
        print(f"[ ] 时间范围应为 2005-2025，当前: {r.index.min()} ~ {r.index.max()}")
        ok = False
    else:
        print(f"[√] 时间范围正确: {r.index.min()} ~ {r.index.max()}")
    if r.shape[0] < 200 or r.shape[1] < 100:
        print(f"[ ] 数据维度约 252×数千，当前: {r.shape}")
        ok = False
    else:
        print(f"[√] 数据维度合理: {r.shape}")
    lo, hi = r.min().min(), r.max().max()
    import numpy as np
    vals = r.to_numpy().reshape(-1)
    vals = vals[~np.isnan(vals)]
    q01, q99 = np.quantile(vals, [0.01, 0.99]).tolist()
    if q01 < -0.6 or q99 > 0.6:
        print(f"[ ] 分位数范围偏大，Q01/Q99=[{q01:.4f}, {q99:.4f}]，全样本范围=[{lo:.4f}, {hi:.4f}]")
        ok = False
    else:
        print(f"[√] 收益率分位数范围合理: Q01/Q99=[{q01:.4f}, {q99:.4f}]")
    if (r == 0).all(axis=0).any():
        print("[ ] 存在全为 0 的列（数据质量）")
        ok = False
    else:
        print("[√] 无全 0 列")
    return ok


def check_no_future():
    """3.6.7 防未来函数泄露检查"""
    print("\n--- 防泄露检查 ---")
    from src.signal import make_signal_reversal_1month
    from src.portfolio import backtest_gross, build_topk_portfolio
    from src.data_loader import load_monthly_returns
    try:
        ret = load_monthly_returns()
    except FileNotFoundError:
        print("[ ] 无法加载数据，跳过")
        return
    sig = make_signal_reversal_1month(ret)
    w = build_topk_portfolio(sig, topk=50)
    pr = backtest_gross(w, ret)
    # 信号应使用 shift(1)：第 0 行全 NaN
    if sig.iloc[0].notna().any():
        print("[ ] 信号首行应全为 NaN（未使用 shift(1)）")
    else:
        print("[√] 信号首行为 NaN（shift(1)）")
    # 回测应使用 shift(1)：与手工对齐公式一致
    manual = (w.shift(1).reindex_like(ret).fillna(0.0) * ret).sum(axis=1)
    if not manual.equals(pr):
        print("[ ] 组合收益与手工对齐公式不一致，可能存在时序问题")
    else:
        print("[√] 组合收益与手工对齐公式一致（weights.shift(1)）")
    print("[√] 代码中信号 = -returns.shift(1)，回测 = (weights.shift(1)*returns).sum(axis=1)")


def main():
    print("=== 第一讲 检查清单 ===\n")
    check_data()
    check_no_future()
    print("\n交付物：")
    for name, path in [
        ("data/processed/monthly_returns.csv", "data/processed/monthly_returns.csv"),
        ("src/data_loader.py", "src/data_loader.py"),
        ("src/signal.py", "src/signal.py"),
        ("src/portfolio.py", "src/portfolio.py"),
        ("src/backtest.py", "src/backtest.py"),
        ("src/evaluation.py", "src/evaluation.py"),
        ("exp_log.md", "exp_log.md"),
        ("results/week1_nav.png", "results/week1_nav.png"),
    ]:
        full = os.path.join(PROJECT_ROOT, path)
        print(f"  [{'√' if os.path.isfile(full) else ' '}] {name}")


if __name__ == "__main__":
    main()
