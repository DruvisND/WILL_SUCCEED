import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_monthly_returns, clean_returns


def make_signal_reversal_1month(returns_df):
    """
    生成1个月反转信号

    信号定义: s_{i,t} = -r_{i,t-1}

    含义:
    - 上月跌得多的股票(r_{i,t-1}为负) -> 信号 s_{i,t} 为正 -> 买入
    - 上月涨得多的股票(r_{i,t-1}为正) -> 信号 s_{i,t} 为负 -> 卖出

    防泄露: 使用 shift(1) 确保只使用 t-1 及之前的数据

    Parameters:
    -----------
    returns_df : pd.DataFrame
        月度收益率矩阵 (日期 × 股票)

    Returns:
    --------
    signal_df : pd.DataFrame
        信号矩阵 (日期 × 股票)
    """
    signal_df = -returns_df.shift(1)
    return signal_df


def make_signal_reversal_kmonth(returns_df, k=3):
    """
    生成K个月反转信号

    信号定义: s_{i,t} = -sum_{j=1}^{K} r_{i,t-j}

    含义: 过去K个月累计收益的负值

    防泄露: 使用 shift(1) 确保信号只使用历史数据

    Parameters:
    -----------
    returns_df : pd.DataFrame
        月度收益率矩阵 (日期 × 股票)
    k : int
        形成期月数 (默认3)

    Returns:
    --------
    signal_df : pd.DataFrame
        信号矩阵 (日期 × 股票)
    """
    rolling_sum = returns_df.shift(1).rolling(window=k).sum()
    signal_df = -rolling_sum
    return signal_df


def standardize_signal(signal_df, method='raw'):
    """
    信号标准化

    Parameters:
    -----------
    signal_df : pd.DataFrame
        信号矩阵
    method : str
        标准化方法:
        - 'raw': 不标准化
        - 'rank': 横截面排名标准化 (0-1)
        - 'zscore': 横截面Z-score标准化
        - 'winsor': 极值压缩到5%-95%

    Returns:
    --------
    standardized_df : pd.DataFrame
        标准化后的信号矩阵
    """
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


def validate_signal(signal_df, returns_df):
    """
    验证信号生成是否正确（防泄露检查）

    检查项:
    1. 信号维度与收益率数据相同
    2. 第一个月的信号全为NaN（因为没有历史数据）
    3. 信号与收益率的关系正确（信号 = -上月收益）

    Parameters:
    -----------
    signal_df : pd.DataFrame
        信号矩阵
    returns_df : pd.DataFrame
        收益率矩阵

    Returns:
    --------
    is_valid : bool
        信号是否有效
    """
    print("\n" + "=" * 50)
    print("信号验证")
    print("=" * 50)

    print(f"\n1. 维度检查:")
    print(f"   信号维度: {signal_df.shape}")
    print(f"   收益率维度: {returns_df.shape}")
    if signal_df.shape == returns_df.shape:
        print("   [OK] 维度一致")
    else:
        print("   [ERROR] 维度不一致")
        return False

    print(f"\n2. 第一时间点检查:")
    first_row = signal_df.iloc[0]
    first_all_nan = first_row.isna().all()
    print(f"   第一月信号全为NaN: {first_all_nan}")
    if first_all_nan:
        print("   [OK] 符合预期（没有历史数据）")
    else:
        print("   [WARN] 第一月信号不应全为NaN")

    print(f"\n3. 信号与收益率关系检查（抽查）:")
    for idx in range(1, min(5, len(signal_df))):
        row = signal_df.iloc[idx]
        if row.notna().any():
            sample_date = signal_df.index[idx]
            sample_stkcd = row.dropna().index[0]
            actual_signal = signal_df.loc[sample_date, sample_stkcd]
            lagged_return = returns_df.shift(1).loc[sample_date, sample_stkcd]
            expected_signal = -lagged_return
            print(f"   抽查日期: {sample_date}, 股票: {sample_stkcd}")
            print(f"   t-1月收益率: {lagged_return:.6f}")
            print(f"   信号定义: -上期收益率 = {-lagged_return:.6f}")
            print(f"   实际信号: {actual_signal:.6f}")
            if abs(actual_signal - expected_signal) < 1e-6:
                print("   [OK] 信号 = -上期收益率，验证通过")
            else:
                print("   [ERROR] 信号与收益率关系可能有问题")
                return False
            break

    print("\n" + "=" * 50)
    print("信号验证通过")
    print("=" * 50)

    return True


if __name__ == '__main__':
    print("=" * 50)
    print("信号生成测试")
    print("=" * 50)

    returns_df = load_monthly_returns()
    returns_df = clean_returns(returns_df)
    print(f"收益率数据加载完成: {returns_df.shape}")

    print("\n--- 1个月反转信号 ---")
    signal_1m = make_signal_reversal_1month(returns_df)
    print(f"信号维度: {signal_1m.shape}")
    print(f"前3行预览:")
    print(signal_1m.head(3))

    validate_signal(signal_1m, returns_df)

    print("\n--- 3个月反转信号 ---")
    signal_3m = make_signal_reversal_kmonth(returns_df, k=3)
    print(f"信号维度: {signal_3m.shape}")
    print(f"前4行预览:")
    print(signal_3m.head(4))

    print("\n--- 信号标准化对比 ---")
    for method in ['raw', 'rank', 'zscore', 'winsor']:
        signal_std = standardize_signal(signal_1m, method=method)
        print(f"\n{method}标准化:")
        print(f"  范围: [{signal_std.min().min():.4f}, {signal_std.max().max():.4f}]")
        print(f"  均值: {signal_std.mean().mean():.4f}")
