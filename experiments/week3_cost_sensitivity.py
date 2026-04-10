import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_monthly_returns():
    from data_loader import load_raw_data, preprocess_returns  # pyright: ignore[reportMissingImports]
    df = load_raw_data()
    returns_df = preprocess_returns(df)
    return returns_df


def clean_returns(returns_df, max_return=0.5, min_return=-0.5, max_missing_pct=0.8):
    returns_cleaned = returns_df.copy()
    returns_cleaned[returns_cleaned > max_return] = np.nan
    returns_cleaned[returns_cleaned < min_return] = np.nan
    missing_by_stock = returns_cleaned.isna().mean(axis=0)
    valid_stocks = missing_by_stock[missing_by_stock <= max_missing_pct].index
    returns_cleaned = returns_cleaned[valid_stocks]
    returns_cleaned = returns_cleaned.dropna(axis=1, how='all')
    return returns_cleaned


def make_signal_reversal_kmonth(returns_df, k=3):
    rolling_sum = returns_df.shift(1).rolling(window=k).sum()
    signal_df = -rolling_sum
    return signal_df


def build_topk_portfolio(signal_df, topk=50, rebalance='M', weighting='EW'):
    weights_df = signal_df.copy() * 0.0
    for idx in range(len(signal_df)):
        row = signal_df.iloc[idx]
        if rebalance == 'Q':
            if idx % 3 != 0:
                if idx > 0:
                    prev_idx = idx - 1
                    while prev_idx >= 0 and weights_df.iloc[prev_idx].sum() == 0:
                        prev_idx -= 1
                    if prev_idx >= 0:
                        weights_df.iloc[idx] = weights_df.iloc[prev_idx]
                        continue
                else:
                    weights_df.iloc[idx] = 0
                    continue
        valid_signals = row.dropna()
        if len(valid_signals) == 0:
            continue
        topk_stocks = valid_signals.nlargest(topk).index.tolist()
        weight = 1.0 / topk
        for stk in topk_stocks:
            weights_df.iloc[idx, weights_df.columns.get_loc(stk)] = weight
    return weights_df


def backtest_gross(weights_df, returns_df):
    aligned_weights = weights_df.shift(1)
    aligned_returns = returns_df.loc[aligned_weights.index]
    portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()
    return portfolio_returns


def compute_turnover(weights_df):
    weight_diff = weights_df.diff().abs()
    turnover = weight_diff.sum(axis=1) * 0.5
    return turnover


def apply_cost(gross_returns, turnover, cost_bps):
    cost_rate = cost_bps / 10000.0
    trading_cost = turnover * cost_rate
    net_returns = gross_returns - trading_cost
    return net_returns


def calculate_metrics(returns_series):
    metrics = {}
    mean_monthly = returns_series.mean()
    std_monthly = returns_series.std()
    metrics['annual_return'] = (1 + mean_monthly) ** 12 - 1
    metrics['sharpe_ratio'] = mean_monthly / std_monthly * np.sqrt(12) if std_monthly > 0 else np.nan
    nav = (1 + returns_series).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['total_return'] = (1 + returns_series).prod() - 1
    return metrics


def run_cost_sensitivity_analysis():
    print("=" * 60)
    print("Week 3: Cost Sensitivity Analysis")
    print("=" * 60)

    returns_df = load_monthly_returns()
    returns_df = clean_returns(returns_df)
    print(f"\nData loaded: {returns_df.shape}")

    print("\nGenerating signals with 6-month formation period...")
    signal_df = make_signal_reversal_kmonth(returns_df, k=6)

    print("Building portfolio with TopK=50...")
    weights_df = build_topk_portfolio(signal_df, topk=50, rebalance='M', weighting='EW')

    print("Calculating gross returns...")
    gross_returns = backtest_gross(weights_df, returns_df)

    print("Computing turnover...")
    turnover = compute_turnover(weights_df)
    aligned_turnover = turnover.loc[gross_returns.index]
    monthly_turnover = aligned_turnover.mean()
    annual_turnover = monthly_turnover * 12

    print(f"\nTurnover Analysis:")
    print(f"  Monthly turnover: {monthly_turnover*100:.2f}%")
    print(f"  Annual turnover: {annual_turnover*100:.2f}%")

    cost_levels = [0, 10, 20, 50]
    results = {}

    print(f"\nCost Sensitivity Analysis:")
    print("-" * 70)
    print(f"{'Cost (bps)':<12} {'Gross Ret':<12} {'Net Ret':<12} {'Sharpe':<10} {'MaxDD':<10}")
    print("-" * 70)

    for cost_bps in cost_levels:
        net_returns = apply_cost(gross_returns, aligned_turnover, cost_bps)

        gross_metrics = calculate_metrics(gross_returns)
        net_metrics = calculate_metrics(net_returns)

        results[cost_bps] = {
            'gross_returns': gross_returns,
            'net_returns': net_returns,
            'gross_annual': gross_metrics['annual_return'],
            'net_annual': net_metrics['annual_return'],
            'gross_sharpe': gross_metrics['sharpe_ratio'],
            'net_sharpe': net_metrics['sharpe_ratio'],
            'gross_maxdd': gross_metrics['max_drawdown'],
            'net_maxdd': net_metrics['max_drawdown'],
            'cost_drag': gross_metrics['annual_return'] - net_metrics['annual_return']
        }

        print(f"{cost_bps:<12} {gross_metrics['annual_return']*100:>10.2f}% {net_metrics['annual_return']*100:>10.2f}% {net_metrics['sharpe_ratio']:>10.4f} {net_metrics['max_drawdown']*100:>10.2f}%")

    print("-" * 70)

    critical_cost = None
    for cost_bps in cost_levels:
        if results[cost_bps]['net_annual'] < 0:
            critical_cost = cost_bps
            break

    if critical_cost is None:
        critical_cost = ">50"
        print(f"\n[CANDIDATE] Critical cost: {critical_cost} bps (strategy still profitable)")
    else:
        print(f"\n[CANDIDATE] Critical cost: {critical_cost} bps (strategy becomes unprofitable)")

    annual_cost_drag_20bps = results[20]['cost_drag']
    annual_cost_drag_50bps = results[50]['cost_drag']
    print(f"  Cost drag at 20 bps: {annual_cost_drag_20bps*100:.2f}%")
    print(f"  Cost drag at 50 bps: {annual_cost_drag_50bps*100:.2f}%")

    return results, annual_turnover, monthly_turnover


def plot_cost_sensitivity(results, annual_turnover, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cost_levels = list(results.keys())
    net_annual_returns = [results[c]['net_annual'] * 100 for c in cost_levels]
    net_sharpes = [results[c]['net_sharpe'] for c in cost_levels]
    cost_drags = [results[c]['cost_drag'] * 100 for c in cost_levels]

    ax1 = axes[0, 0]
    for cost_bps in cost_levels:
        nav = (1 + results[cost_bps]['net_returns']).cumprod()
        label = f"{cost_bps} bps"
        ax1.plot(nav.index, nav.values, linewidth=1.5, label=label)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('NAV')
    ax1.set_title('Net Value Curve by Cost Level (6-month Formation, TopK=50)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[0, 1]
    x = np.arange(len(cost_levels))
    width = 0.35
    gross_returns = [results[c]['gross_annual'] * 100 for c in cost_levels]
    bars1 = ax2.bar(x - width/2, gross_returns, width, label='Gross', color='#3498db')
    bars2 = ax2.bar(x + width/2, net_annual_returns, width, label='Net', color='#2ecc71')
    ax2.set_xlabel('Cost (bps)')
    ax2.set_ylabel('Annual Return (%)')
    ax2.set_title('Gross vs Net Annual Return')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in cost_levels])
    ax2.legend()
    for bar, val in zip(bars1, gross_returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, net_annual_returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=8)

    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, net_sharpes, color=['#2ecc71' if s > 0 else '#e74c3c' for s in net_sharpes])
    ax3.set_xlabel('Cost (bps)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Net Sharpe Ratio by Cost Level')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(c) for c in cost_levels])
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    for bar, val in zip(bars3, net_sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, cost_drags, color='#e74c3c', alpha=0.7)
    ax4.set_xlabel('Cost (bps)')
    ax4.set_ylabel('Cost Drag (%)')
    ax4.set_title('Annual Cost Drag (Gross - Net)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(c) for c in cost_levels])
    for bar, val in zip(bars4, cost_drags):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9)

    info_text = f"Annual Turnover: {annual_turnover*100:.1f}%"
    fig.text(0.99, 0.01, info_text, ha='right', va='bottom', fontsize=9, style='italic')

    plt.suptitle('Week 3: Cost Sensitivity Analysis\n6-Month Reversal Strategy, TopK=50', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {save_path}")
    plt.close()


def write_exp_log(results, annual_turnover):
    exp_log_path = PROJECT_ROOT / "exp_log.md"
    date_str = datetime.now().strftime('%Y-%m-%d')

    log_entries = []
    log_entries.append("\n" + "=" * 60)
    log_entries.append("## Week 3 实验：成本敏感性分析")
    log_entries.append("=" * 60)
    log_entries.append(f"- **日期**: {date_str}")
    log_entries.append("")

    log_entries.append("### 1. 换手率分析")
    log_entries.append(f"- **月度换手率**: {annual_turnover/12*100:.2f}%")
    log_entries.append(f"- **年化换手率**: {annual_turnover*100:.2f}%")
    log_entries.append("- **发现**: 高换手率(约600%+年化)是反转策略的主要成本来源")
    log_entries.append("")

    log_entries.append("### 2. 成本敏感性分析")
    log_entries.append(f"- **基准配置**: 形成期=6个月, TopK=50, 再平衡=月度")
    log_entries.append("")
    log_entries.append("| 成本(bps) | 总收益 | 净收益 | 夏普比率 | 最大回撤 | 成本拖累 |")
    log_entries.append("|-----------|--------|--------|----------|----------|----------|")
    for cost_bps in [0, 10, 20, 50]:
        r = results[cost_bps]
        log_entries.append(f"| {cost_bps} | {r['gross_annual']*100:.2f}% | {r['net_annual']*100:.2f}% | {r['net_sharpe']:.4f} | {r['net_maxdd']*100:.2f}% | {r['cost_drag']*100:.2f}% |")
    log_entries.append("")

    log_entries.append("### 3. 临界成本分析")
    net_at_50 = results[50]['net_annual']
    if net_at_50 > 0:
        log_entries.append("- **临界成本**: >50 bps (策略在50bps成本下仍盈利)")
        log_entries.append("- **评估**: 策略对成本有一定容忍度，适合低成本交易环境")
    else:
        log_entries.append("- **临界成本**: <50 bps")
        log_entries.append("- **评估**: 策略对成本敏感，需要低成本执行环境")
    log_entries.append("")

    log_entries.append("### 4. 策略可交易性评估")
    log_entries.append(f"- **年化换手率**: {annual_turnover*100:.1f}%")
    log_entries.append(f"- **成本拖累(20bps)**: {results[20]['cost_drag']*100:.2f}%")
    log_entries.append(f"- **成本拖累(50bps)**: {results[50]['cost_drag']*100:.2f}%")
    log_entries.append("- **容量**: TopK=50, 日均交易量需约50只×平均成交量")
    log_entries.append("- **改善建议**:")
    log_entries.append("  1. 降低再平衡频率(季度)可减少换手")
    log_entries.append("  2. 增大TopK可分散交易")
    log_entries.append("  3. 考虑添加流动性筛选条件")
    log_entries.append("")
    log_entries.append("```bash")
    log_entries.append("python experiments/week3_cost_sensitivity.py")
    log_entries.append("```")
    log_entries.append("发现:")

    with open(exp_log_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(log_entries))

    print(f"\nExperiment log updated: {exp_log_path}")


if __name__ == '__main__':
    results, annual_turnover, monthly_turnover = run_cost_sensitivity_analysis()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    plot_cost_sensitivity(results, annual_turnover, results_dir / "week3_cost_sensitivity.png")

    write_exp_log(results, annual_turnover)

    print("\n" + "=" * 60)
    print("Cost Sensitivity Analysis Complete!")
    print("=" * 60)
