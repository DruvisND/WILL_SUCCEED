import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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


def make_signal_reversal_1month(returns_df):
    signal_df = -returns_df.shift(1)
    return signal_df


def make_signal_reversal_kmonth(returns_df, k=3):
    rolling_sum = returns_df.shift(1).rolling(window=k).sum()
    signal_df = -rolling_sum
    return signal_df


def standardize_signal(signal_df, method='raw'):
    if method == 'raw':
        return signal_df
    standardized = signal_df.copy()
    if method == 'rank':
        standardized = signal_df.rank(axis=1, pct=True)
    elif method == 'zscore':
        row_mean = signal_df.mean(axis=1, skipna=True)
        row_std = signal_df.std(axis=1, skipna=True)
        row_std = row_std.replace(0, np.nan)
        standardized = signal_df.sub(row_mean, axis=0).div(row_std, axis=0)
    elif method == 'winsor':
        lower = signal_df.quantile(0.05, axis=1)
        upper = signal_df.quantile(0.95, axis=1)
        standardized = signal_df.clip(lower=lower, upper=upper, axis=0)
    return standardized


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


def calculate_turnover(weights_df):
    weight_diff = weights_df.diff().abs().sum(axis=1)
    weight_diff = weight_diff.dropna()
    avg_monthly_turnover = weight_diff.mean() / 2
    avg_annual_turnover = avg_monthly_turnover * 12
    return avg_annual_turnover


def run_single_experiment(returns_df, formation_period, topk, standardization, rebalance):
    if formation_period == 1:
        signal_df = make_signal_reversal_1month(returns_df)
    else:
        signal_df = make_signal_reversal_kmonth(returns_df, k=formation_period)

    signal_std = standardize_signal(signal_df, method=standardization)
    weights_df = build_topk_portfolio(signal_std, topk=topk, rebalance=rebalance, weighting='EW')
    portfolio_returns = backtest_gross(weights_df, returns_df)
    turnover = calculate_turnover(weights_df)

    metrics = {}
    mean_monthly = portfolio_returns.mean()
    std_monthly = portfolio_returns.std()
    metrics['annual_return'] = (1 + mean_monthly) ** 12 - 1
    metrics['sharpe_ratio'] = mean_monthly / std_monthly * np.sqrt(12) if std_monthly > 0 else np.nan
    nav = (1 + portfolio_returns).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['turnover'] = turnover
    metrics['portfolio_returns'] = portfolio_returns

    return metrics


def plot_formation_comparison(returns_df, save_path):
    print("\nGenerating formation period comparison chart...")
    results = {}
    for k in [1, 3, 6, 12]:
        metrics = run_single_experiment(returns_df, formation_period=k, topk=50,
                                       standardization='raw', rebalance='M')
        results[k] = metrics

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    formations = sorted(results.keys())
    annual_returns = [results[k]['annual_return'] * 100 for k in formations]
    sharpes = [results[k]['sharpe_ratio'] for k in formations]
    max_dds = [results[k]['max_drawdown'] * 100 for k in formations]
    turnovers = [results[k]['turnover'] * 100 for k in formations]

    colors = ['#2ecc71' if k == 6 else '#3498db' for k in formations]

    ax1 = axes[0, 0]
    bars1 = ax1.bar([str(k) for k in formations], annual_returns, color=colors)
    ax1.set_xlabel('Formation Period (months)')
    ax1.set_ylabel('Annual Return (%)')
    ax1.set_title('Annual Return by Formation Period')
    ax1.axhline(y=np.mean(annual_returns), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars1, annual_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax1.legend()

    ax2 = axes[0, 1]
    bars2 = ax2.bar([str(k) for k in formations], sharpes, color=colors)
    ax2.set_xlabel('Formation Period (months)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio by Formation Period')
    ax2.axhline(y=np.mean(sharpes), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars2, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax2.legend()

    ax3 = axes[1, 0]
    bars3 = ax3.bar([str(k) for k in formations], [abs(x) for x in max_dds], color=colors)
    ax3.set_xlabel('Formation Period (months)')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Max Drawdown by Formation Period')
    ax3.axhline(y=np.mean([abs(x) for x in max_dds]), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars3, max_dds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax3.legend()

    ax4 = axes[1, 1]
    bars4 = ax4.bar([str(k) for k in formations], turnovers, color=colors)
    ax4.set_xlabel('Formation Period (months)')
    ax4.set_ylabel('Annual Turnover (%)')
    ax4.set_title('Annual Turnover by Formation Period')
    ax4.axhline(y=np.mean(turnovers), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars4, turnovers):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9)
    ax4.legend()

    plt.suptitle('Week 2: Formation Period Comparison\n(TopK=50, Standardization=Raw, Rebalance=M)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.close()


def plot_topk_comparison(returns_df, best_formation, save_path):
    print("\nGenerating TopK comparison chart...")
    results = {}
    for topk in [20, 50, 100, 200]:
        metrics = run_single_experiment(returns_df, formation_period=best_formation, topk=topk,
                                       standardization='raw', rebalance='M')
        results[topk] = metrics

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    topks = sorted(results.keys())
    annual_returns = [results[k]['annual_return'] * 100 for k in topks]
    sharpes = [results[k]['sharpe_ratio'] for k in topks]
    max_dds = [results[k]['max_drawdown'] * 100 for k in topks]
    turnovers = [results[k]['turnover'] * 100 for k in topks]

    colors = ['#2ecc71' if k == 50 else '#3498db' for k in topks]

    ax1 = axes[0, 0]
    bars1 = ax1.bar([str(k) for k in topks], annual_returns, color=colors)
    ax1.set_xlabel('TopK')
    ax1.set_ylabel('Annual Return (%)')
    ax1.set_title('Annual Return by TopK')
    ax1.axhline(y=np.mean(annual_returns), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars1, annual_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax1.legend()

    ax2 = axes[0, 1]
    bars2 = ax2.bar([str(k) for k in topks], sharpes, color=colors)
    ax2.set_xlabel('TopK')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio by TopK')
    ax2.axhline(y=np.mean(sharpes), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars2, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax2.legend()

    ax3 = axes[1, 0]
    bars3 = ax3.bar([str(k) for k in topks], [abs(x) for x in max_dds], color=colors)
    ax3.set_xlabel('TopK')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Max Drawdown by TopK')
    ax3.axhline(y=np.mean([abs(x) for x in max_dds]), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars3, max_dds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax3.legend()

    ax4 = axes[1, 1]
    bars4 = ax4.bar([str(k) for k in topks], turnovers, color=colors)
    ax4.set_xlabel('TopK')
    ax4.set_ylabel('Annual Turnover (%)')
    ax4.set_title('Annual Turnover by TopK')
    ax4.axhline(y=np.mean(turnovers), color='red', linestyle='--', alpha=0.7, label='Average')
    for bar, val in zip(bars4, turnovers):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9)
    ax4.legend()

    plt.suptitle(f'Week 2: TopK Comparison\n(Formation={best_formation}m, Standardization=Raw, Rebalance=M)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.close()


def plot_rebalance_comparison(returns_df, best_formation, best_topk, save_path):
    print("\nGenerating rebalance frequency comparison chart...")
    results = {}
    for rebal in ['M', 'Q']:
        label = 'Monthly' if rebal == 'M' else 'Quarterly'
        metrics = run_single_experiment(returns_df, formation_period=best_formation, topk=best_topk,
                                       standardization='raw', rebalance=rebal)
        results[rebal] = metrics

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    rebals = ['M', 'Q']
    labels = ['Monthly', 'Quarterly']
    annual_returns = [results[k]['annual_return'] * 100 for k in rebals]
    sharpes = [results[k]['sharpe_ratio'] for k in rebals]
    max_dds = [results[k]['max_drawdown'] * 100 for k in rebals]
    turnovers = [results[k]['turnover'] * 100 for k in rebals]

    colors = ['#2ecc71', '#3498db']

    ax1 = axes[0, 0]
    bars1 = ax1.bar(labels, annual_returns, color=colors)
    ax1.set_xlabel('Rebalance Frequency')
    ax1.set_ylabel('Annual Return (%)')
    ax1.set_title('Annual Return by Rebalance Frequency')
    for bar, val in zip(bars1, annual_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)

    ax2 = axes[0, 1]
    bars2 = ax2.bar(labels, sharpes, color=colors)
    ax2.set_xlabel('Rebalance Frequency')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio by Rebalance Frequency')
    for bar, val in zip(bars2, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax3 = axes[1, 0]
    bars3 = ax3.bar(labels, [abs(x) for x in max_dds], color=colors)
    ax3.set_xlabel('Rebalance Frequency')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Max Drawdown by Rebalance Frequency')
    for bar, val in zip(bars3, max_dds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)

    ax4 = axes[1, 1]
    bars4 = ax4.bar(labels, turnovers, color=colors)
    ax4.set_xlabel('Rebalance Frequency')
    ax4.set_ylabel('Annual Turnover (%)')
    ax4.set_title('Annual Turnover by Rebalance Frequency')
    for bar, val in zip(bars4, turnovers):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=10)

    plt.suptitle(f'Week 2: Rebalance Frequency Comparison\n(Formation={best_formation}m, TopK={best_topk}, Standardization=Raw)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Week 2 Visualization Charts")
    print("=" * 60)

    returns_df = load_monthly_returns()
    returns_df = clean_returns(returns_df)
    print(f"\nData loaded: {returns_df.shape}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    plot_formation_comparison(returns_df, results_dir / "week2_formation_period.png")

    plot_topk_comparison(returns_df, best_formation=6, save_path=results_dir / "week2_topk.png")

    plot_rebalance_comparison(returns_df, best_formation=6, best_topk=50,
                             save_path=results_dir / "week2_rebalance.png")

    print("\n" + "=" * 60)
    print("All charts generated successfully!")
    print("=" * 60)
