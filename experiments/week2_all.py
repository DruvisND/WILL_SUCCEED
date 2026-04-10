import pandas as pd
import numpy as np
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
    metrics['total_return'] = (1 + portfolio_returns).prod() - 1

    return metrics


def find_best_config(results_dict, metric='sharpe_ratio'):
    best_val = -np.inf
    best_key = None
    for key, metrics in results_dict.items():
        if metrics[metric] > best_val:
            best_val = metrics[metric]
            best_key = key
    return best_key, best_val


def run_formation_period_experiments(returns_df):
    print("\n" + "=" * 60)
    print("维度1: 形成期探索")
    print("固定: TopK=50, 标准化=Raw, 再平衡=M")
    print("=" * 60)

    results = {}
    for k in [1, 3, 6, 12]:
        print(f"\n--- 形成期 K={k} ---")
        metrics = run_single_experiment(returns_df, formation_period=k, topk=50,
                                        standardization='raw', rebalance='M')
        results[k] = metrics
        print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  年化换手: {metrics['turnover']*100:.2f}%")

    return results


def run_standardization_experiments(returns_df, best_formation):
    print("\n" + "=" * 60)
    print("维度2: 标准化探索")
    print(f"固定: 形成期={best_formation}, TopK=50, 再平衡=M")
    print("=" * 60)

    results = {}
    for std in ['raw', 'rank', 'zscore', 'winsor']:
        print(f"\n--- 标准化: {std} ---")
        metrics = run_single_experiment(returns_df, formation_period=best_formation, topk=50,
                                        standardization=std, rebalance='M')
        results[std] = metrics
        print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  年化换手: {metrics['turnover']*100:.2f}%")

    return results


def run_topk_experiments(returns_df, best_formation):
    print("\n" + "=" * 60)
    print("维度3: TopK探索")
    print(f"固定: 形成期={best_formation}, 标准化=Raw, 再平衡=M")
    print("=" * 60)

    results = {}
    for topk in [20, 50, 100, 200]:
        print(f"\n--- TopK={topk} ---")
        metrics = run_single_experiment(returns_df, formation_period=best_formation, topk=topk,
                                        standardization='raw', rebalance='M')
        results[topk] = metrics
        print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  年化换手: {metrics['turnover']*100:.2f}%")

    return results


def run_rebalance_experiments(returns_df, best_formation, best_topk):
    print("\n" + "=" * 60)
    print("维度4: 再平衡频率探索")
    print(f"固定: 形成期={best_formation}, TopK={best_topk}, 标准化=Raw")
    print("=" * 60)

    results = {}
    for rebal in ['M', 'Q']:
        label = '月度' if rebal == 'M' else '季度'
        print(f"\n--- 再平衡: {label} ---")
        metrics = run_single_experiment(returns_df, formation_period=best_formation, topk=best_topk,
                                        standardization='raw', rebalance=rebal)
        results[rebal] = metrics
        print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  年化换手: {metrics['turnover']*100:.2f}%")

    return results


def write_exp_log(all_results, best_config_final):
    exp_log_path = PROJECT_ROOT / "exp_log.md"
    date_str = datetime.now().strftime('%Y-%m-%d')

    log_entries = []

    log_entries.append("\n" + "=" * 60)
    log_entries.append("## Week 2 实验：参数探索")
    log_entries.append("=" * 60)
    log_entries.append(f"- **日期**: {date_str}")
    log_entries.append("")

    formation_results = all_results['formation']
    best_formation, best_f_sharpe = find_best_config(formation_results)
    log_entries.append("### 维度1: 形成期探索")
    log_entries.append(f"- **固定参数**: TopK=50, 标准化=Raw, 再平衡=月度")
    log_entries.append(f"- **变量参数**: 形成期 K = {{1, 3, 6, 12}} 个月")
    log_entries.append("")
    log_entries.append("| 形成期 | 年化收益 | 夏普比率 | 最大回撤 | 年化换手 |")
    log_entries.append("|--------|----------|----------|----------|----------|")
    for k, m in sorted(formation_results.items()):
        log_entries.append(f"| {k}月 | {m['annual_return']*100:.2f}% | {m['sharpe_ratio']:.4f} | {m['max_drawdown']*100:.2f}% | {m['turnover']*100:.2f}% |")
    log_entries.append("")
    log_entries.append("**发现**:")
    log_entries.append(f"1. 最优形成期: {best_formation}个月 (夏普={best_f_sharpe:.4f})")
    log_entries.append("2. 短期(1月)反转效应存在但换手率高; 长期(12月)可能受动量效应干扰")
    log_entries.append("")

    std_results = all_results['standardization']
    best_std, best_s_sharpe = find_best_config(std_results)
    log_entries.append("### 维度2: 标准化探索")
    log_entries.append(f"- **固定参数**: 形成期={best_formation}, TopK=50, 再平衡=月度")
    log_entries.append(f"- **变量参数**: 标准化方法 = {{Raw, Rank, Z-score, Winsorization}}")
    log_entries.append("")
    log_entries.append("| 标准化 | 年化收益 | 夏普比率 | 最大回撤 | 年化换手 |")
    log_entries.append("|--------|----------|----------|----------|----------|")
    for std, m in std_results.items():
        log_entries.append(f"| {std} | {m['annual_return']*100:.2f}% | {m['sharpe_ratio']:.4f} | {m['max_drawdown']*100:.2f}% | {m['turnover']*100:.2f}% |")
    log_entries.append("")
    log_entries.append("**发现**:")
    log_entries.append("1. 对于TopK策略，标准化不改变选股结果(只关心排序)")
    log_entries.append("2. 不同标准化方法收益差异较小，主要影响信号数值分布")
    log_entries.append("")

    topk_results = all_results['topk']
    best_topk, best_t_sharpe = find_best_config(topk_results)
    log_entries.append("### 维度3: TopK探索")
    log_entries.append(f"- **固定参数**: 形成期={best_formation}, 标准化=Raw, 再平衡=月度")
    log_entries.append(f"- **变量参数**: TopK = {{20, 50, 100, 200}}")
    log_entries.append("")
    log_entries.append("| TopK | 年化收益 | 夏普比率 | 最大回撤 | 年化换手 |")
    log_entries.append("|------|----------|----------|----------|----------|")
    for tk, m in sorted(topk_results.items()):
        log_entries.append(f"| {tk} | {m['annual_return']*100:.2f}% | {m['sharpe_ratio']:.4f} | {m['max_drawdown']*100:.2f}% | {m['turnover']*100:.2f}% |")
    log_entries.append("")
    log_entries.append("**发现**:")
    log_entries.append(f"1. 最优TopK: {best_topk} (夏普={best_t_sharpe:.4f})")
    log_entries.append("2. TopK小集中度高但风险大; TopK大分散化好但可能稀释alpha")
    log_entries.append("")

    rebal_results = all_results['rebalance']
    best_rebal, best_r_sharpe = find_best_config(rebal_results)
    rebal_label = '月度' if best_rebal == 'M' else '季度'
    log_entries.append("### 维度4: 再平衡频率探索")
    log_entries.append(f"- **固定参数**: 形成期={best_formation}, TopK={best_topk}, 标准化=Raw")
    log_entries.append(f"- **变量参数**: 再平衡频率 = {{月度M, 季度Q}}")
    log_entries.append("")
    log_entries.append("| 再平衡 | 年化收益 | 夏普比率 | 最大回撤 | 年化换手 |")
    log_entries.append("|--------|----------|----------|----------|----------|")
    for rb, m in rebal_results.items():
        label = '月度' if rb == 'M' else '季度'
        log_entries.append(f"| {label} | {m['annual_return']*100:.2f}% | {m['sharpe_ratio']:.4f} | {m['max_drawdown']*100:.2f}% | {m['turnover']*100:.2f}% |")
    log_entries.append("")
    log_entries.append("**发现**:")
    log_entries.append(f"1. 最优再平衡: {rebal_label} (夏普={best_r_sharpe:.4f})")
    log_entries.append("2. 月度再平衡及时但换手高; 季度再平衡降低成本但可能错过信号")
    log_entries.append("")

    log_entries.append("### 最优参数组合")
    log_entries.append("=" * 40)
    log_entries.append(f"- **形成期**: {best_formation}个月")
    log_entries.append(f"- **标准化**: {best_std} (对TopK策略影响小)")
    log_entries.append(f"- **TopK**: {best_topk}")
    log_entries.append(f"- **再平衡**: {rebal_label}")
    log_entries.append(f"- **综合夏普比率**: {best_config_final['sharpe']:.4f}")
    log_entries.append("")
    log_entries.append("**选择理由**: 基于夏普比率综合评估，平衡收益与风险")
    log_entries.append("")
    log_entries.append("```bash")
    log_entries.append("python experiments/week2_all.py")
    log_entries.append("```")
    log_entries.append("发现:")

    with open(exp_log_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(log_entries))

    print(f"\n实验日志已更新: {exp_log_path}")

    return {
        'formation': best_formation,
        'standardization': best_std,
        'topk': best_topk,
        'rebalance': best_rebal
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Week 2: 参数探索实验")
    print("=" * 60)

    returns_df = load_monthly_returns()
    returns_df = clean_returns(returns_df)
    print(f"\n数据加载: {returns_df.shape}")

    all_results = {}

    formation_results = run_formation_period_experiments(returns_df)
    all_results['formation'] = formation_results

    best_formation, _ = find_best_config(formation_results)
    print(f"\n>>> 最优形成期: {best_formation}个月")

    std_results = run_standardization_experiments(returns_df, best_formation)
    all_results['standardization'] = std_results

    best_std, _ = find_best_config(std_results)

    topk_results = run_topk_experiments(returns_df, best_formation)
    all_results['topk'] = topk_results

    best_topk, _ = find_best_config(topk_results)
    print(f"\n>>> 最优TopK: {best_topk}")

    rebal_results = run_rebalance_experiments(returns_df, best_formation, best_topk)
    all_results['rebalance'] = rebal_results

    best_rebal, _ = find_best_config(rebal_results)

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
    print(f"\n最优配置:")
    print(f"  形成期: {best_formation}个月")
    print(f"  标准化: {best_std}")
    print(f"  TopK: {best_topk}")
    print(f"  再平衡: {'月度' if best_rebal == 'M' else '季度'}")

    best_config_final = {}
    metrics = run_single_experiment(returns_df, best_formation, best_topk, best_std, best_rebal)
    best_config_final['sharpe'] = metrics['sharpe_ratio']
    best_config_final['annual_return'] = metrics['annual_return']
    best_config_final['max_drawdown'] = metrics['max_drawdown']
    best_config_final['turnover'] = metrics['turnover']

    print(f"\n最优配置回测结果:")
    print(f"  年化收益: {best_config_final['annual_return']*100:.2f}%")
    print(f"  夏普比率: {best_config_final['sharpe']:.4f}")
    print(f"  最大回撤: {best_config_final['max_drawdown']*100:.2f}%")
    print(f"  年化换手: {best_config_final['turnover']*100:.2f}%")

    best_params = write_exp_log(all_results, best_config_final)
