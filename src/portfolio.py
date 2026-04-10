import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_monthly_returns, clean_returns
import signal as signal_module


def build_topk_portfolio(signal_df, topk=50, rebalance='M', weighting='EW'):
    """
    构建TopK等权组合

    组合构建步骤:
    1. 排序选股(TopK): 每月按信号从高到低排序，选择信号最高的K只股票
    2. 权重分配: 对选中的K只股票分配等权重(1/K)，未选中的权重为0
    3. 防泄露: 使用shift(1)确保t期权重应用于t+1期收益

    Parameters:
    -----------
    signal_df : pd.DataFrame
        信号矩阵 (日期 × 股票)
    topk : int
        选择前K只股票 (默认50)
    rebalance : str
        再平衡频率 ('M'=月度, 'Q'=季度)
    weighting : str
        权重方案 ('EW'=等权重)

    Returns:
    --------
    weights_df : pd.DataFrame
        权重矩阵 (日期 × 股票)
    """
    print(f"\n构建TopK组合: TopK={topk}, 再平衡={rebalance}, 权重={weighting}")
    print(f"信号维度: {signal_df.shape}")

    if weighting != 'EW':
        raise ValueError(f"目前只支持等权重(EW)，收到: {weighting}")

    if rebalance not in ['M', 'Q']:
        raise ValueError(f"再平衡频率只支持 M(月度) 或 Q(季度)，收到: {rebalance}")

    weights_df = signal_df.copy()
    weights_df = weights_df * 0.0

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

    print(f"权重矩阵维度: {weights_df.shape}")
    non_zero_weights = (weights_df > 0).sum(axis=1)
    print(f"每月持有股票数范围: {non_zero_weights.min()} - {non_zero_weights.max()}")

    return weights_df


def backtest_gross(weights_df, returns_df):
    """
    计算总收益(不考虑成本)

    关键: weights_df.shift(1) 确保 t 期的权重应用于 t+1 期的收益
    这符合防未来函数泄露原则：月末的权重基于当月信号，用于下月收益

    Parameters:
    -----------
    weights_df : pd.DataFrame
        权重矩阵 (日期 × 股票)
    returns_df : pd.DataFrame
        收益率矩阵 (日期 × 股票)

    Returns:
    --------
    portfolio_returns : pd.Series
        组合收益序列
    """
    aligned_weights = weights_df.shift(1)

    aligned_returns = returns_df.loc[aligned_weights.index]

    portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1)

    portfolio_returns = portfolio_returns.dropna()

    if len(portfolio_returns) == 0:
        print("[ERROR] 组合收益为空，请检查权重和收益率数据对齐")
        return portfolio_returns

    print(f"\n回测结果:")
    print(f"  起始日期: {portfolio_returns.index[0]}")
    print(f"  结束日期: {portfolio_returns.index[-1]}")
    print(f"  交易月数: {len(portfolio_returns)}")
    print(f"  月均收益: {portfolio_returns.mean()*100:.4f}%")
    print(f"  月收益标准差: {portfolio_returns.std()*100:.4f}%")

    return portfolio_returns


def calculate_turnover(weights_df):
    """
    计算换手率

    换手率 = |w_t - w_{t-1}| / 2 的均值

    Parameters:
    -----------
    weights_df : pd.DataFrame
        权重矩阵 (日期 × 股票)

    Returns:
    --------
    avg_turnover : float
        平均年化换手率
    """
    weight_diff = weights_df.diff().abs().sum(axis=1)
    weight_diff = weight_diff.dropna()

    avg_monthly_turnover = weight_diff.mean() / 2

    avg_annual_turnover = avg_monthly_turnover * 12

    return avg_annual_turnover


def validate_portfolio(weights_df, signal_df, returns_df):
    """
    验证组合构建是否正确

    检查项:
    1. 权重矩阵维度与信号矩阵一致
    2. 每行权重之和为1或0(当没有有效信号时)
    3. 权重为非负值
    4. TopK股票数正确

    Parameters:
    -----------
    weights_df : pd.DataFrame
        权重矩阵
    signal_df : pd.DataFrame
        信号矩阵
    returns_df : pd.DataFrame
        收益率矩阵

    Returns:
    --------
    is_valid : bool
        组合是否有效
    """
    print("\n" + "=" * 50)
    print("组合验证")
    print("=" * 50)

    print(f"\n1. 维度检查:")
    print(f"   权重维度: {weights_df.shape}")
    print(f"   信号维度: {signal_df.shape}")
    if weights_df.shape == signal_df.shape:
        print("   [OK] 维度一致")
    else:
        print("   [ERROR] 维度不一致")
        return False

    print(f"\n2. 权重和检查:")
    weight_sums = weights_df.sum(axis=1)
    valid_sums = weight_sums[weight_sums > 0]
    if len(valid_sums) > 0:
        print(f"   非零权重之和范围: [{valid_sums.min():.6f}, {valid_sums.max():.6f}]")
        if abs(valid_sums.mean() - 1.0) < 0.01:
            print("   [OK] 权重和接近1")
        else:
            print("   [WARN] 权重和偏离1")

    print(f"\n3. 权重非负检查:")
    negative_weights = (weights_df < 0).sum().sum()
    if negative_weights == 0:
        print("   [OK] 无负权重")
    else:
        print(f"   [ERROR] 存在 {negative_weights} 个负权重")
        return False

    print(f"\n4. TopK检查(抽查):")
    topk = 50
    for idx in range(1, min(5, len(weights_df))):
        row_weights = weights_df.iloc[idx]
        non_zero = (row_weights > 0).sum()
        if non_zero > 0:
            sample_date = weights_df.index[idx]
            selected_stocks = row_weights[row_weights > 0].index.tolist()
            sample_signals = signal_df.loc[sample_date, selected_stocks]
            print(f"   {sample_date}: 持有{non_zero}只, 信号范围[{sample_signals.min():.4f}, {sample_signals.max():.4f}]")
            break

    print("\n" + "=" * 50)
    print("组合验证通过")
    print("=" * 50)

    return True


if __name__ == '__main__':
    print("=" * 50)
    print("组合构建测试")
    print("=" * 50)

    returns_df = load_monthly_returns()
    returns_df = clean_returns(returns_df)
    print(f"收益率数据加载完成: {returns_df.shape}")

    signal_df = make_signal_reversal_1month(returns_df)
    print(f"信号生成完成: {signal_df.shape}")

    weights_df = build_topk_portfolio(signal_df, topk=50, rebalance='M', weighting='EW')

    validate_portfolio(weights_df, signal_df, returns_df)

    portfolio_returns = backtest_gross(weights_df, returns_df)

    turnover = calculate_turnover(weights_df)
    print(f"\n年化换手率: {turnover*100:.2f}%")

    print(f"\n组合收益序列预览(前10月):")
    print(portfolio_returns.head(10))
