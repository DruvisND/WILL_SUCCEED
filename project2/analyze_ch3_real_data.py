"""基于真实数据验证CH-3因子，特别关注论文时间段2000-2016"""
import pandas as pd
import numpy as np
from src.factors import construct_ch3_factor, validate_factors

print("=" * 70)
print("真实数据CH-3因子验证")
print("=" * 70)

# 加载数据
data = pd.read_csv('data/processed/monthly_data.csv')
rf = pd.read_csv('data/processed/rf_monthly.csv')

print(f"\n✓ 数据加载完成:")
print(f"  月度数据: {len(data)} 条")
print(f"  时间范围: {data['date'].min()} ~ {data['date'].max()}")
print(f"  无风险利率: {len(rf)} 条")

# ============================================================================
# 1️⃣ 全时期分析（2000-2025）
# ============================================================================
print("\n" + "=" * 70)
print("1️⃣  全时期分析（2000-2025）")
print("=" * 70)

factors_all = construct_ch3_factor(data, rf)
factors_all.to_csv('data/factors/ch3_monthly_full.csv', index=False)
stats_all = validate_factors(factors_all)

print(f"\n{len(factors_all)} 个月的因子数据")
print(f"{'因子':<6} {'月均收益':>12} {'标准差':>12} {'t统计':>10}")
print("-" * 70)
for col in ['MKT', 'SMB', 'VMG']:
    vals = stats_all['stats'][col]
    print(f"{col:<6} {vals['mean']:>11.4%}  {vals['std']:>11.4%}  {vals['t_stat']:>9.3f}")


# ============================================================================
# 2️⃣ 论文时期分析（2000-2016）
# ============================================================================
print("\n" + "=" * 70)
print("2️⃣  论文时期分析（2000-2016）")
print("=" * 70)

data_2000_2016 = data[data['date'].str[:7] <= '2016-12'].copy()
factors_2000_2016 = construct_ch3_factor(data_2000_2016, rf)
factors_2000_2016.to_csv('data/factors/ch3_monthly_2000_2016.csv', index=False)
stats_2000_2016 = validate_factors(factors_2000_2016)

print(f"\n{len(factors_2000_2016)} 个月（2000-2016）的因子数据")
print(f"{'因子':<6} {'月均收益':>12} {'标准差':>12} {'t统计':>10}")
print("-" * 70)
for col in ['MKT', 'SMB', 'VMG']:
    vals = stats_2000_2016['stats'][col]
    print(f"{col:<6} {vals['mean']:>11.4%}  {vals['std']:>11.4%}  {vals['t_stat']:>9.3f}")


# ============================================================================
# 3️⃣ 与论文对比
# ============================================================================
print("\n" + "=" * 70)
print("3️⃣  与论文对比（论文数据：2000-2016）")
print("=" * 70)

print(f"{'因子':<6} {'论文值(%)':>10} {'本期值(%)':>12} {'差异':>12} {'状态':>8}")
print("-" * 70)

paper_data = {
    'MKT': {'mean': 0.0066, 'std': 0.0809, 't_stat': 1.160},
    'SMB': {'mean': 0.0103, 'std': 0.0452, 't_stat': 3.250},
    'VMG': {'mean': 0.0114, 'std': 0.0375, 't_stat': 4.340}
}

for col in ['MKT', 'SMB', 'VMG']:
    vals = stats_2000_2016['stats'][col]
    paper_val = paper_data[col]['mean'] * 100
    our_val = vals['mean'] * 100
    diff = our_val - paper_val
    status = "✓" if abs(diff) < 0.5 else "✗"
    print(f"{col:<6} {paper_val:>9.3f}  {our_val:>11.3f}  {diff:>11.3f}  {status:>8}")


# ============================================================================
# 4️⃣ 诊断：按照有效样本量按日期统计
# ============================================================================
print("\n" + "=" * 70)
print("4️⃣  数据诊断")
print("=" * 70)

# 检查每月的样本量
sample_per_month = data.groupby('date').size()
print(f"\n每月样本数统计:")
print(f"  最小: {sample_per_month.min():4d} 只")
print(f"  中位数: {sample_per_month.median():4.0f} 只")
print(f"  平均: {sample_per_month.mean():4.1f} 只")
print(f"  最大: {sample_per_month.max():4d} 只")

# 检查EP数据
ep_stats = data[data['ep'] > 0]['ep'].describe()
print(f"\nEP数据统计（正值部分）:")
print(f"  均值: {ep_stats['mean']:.4f}")
print(f"  中位数: {ep_stats['50%']:.4f}")
print(f"  std: {ep_stats['std']:.4f}")

# 检查负EP比例
neg_ep_ratio = (data['ep'] <= 0).sum() / len(data)
print(f"\n负EP（亏损公司）比例: {neg_ep_ratio:.1%}")

# 检查市值分布
print(f"\n市值分布（单位：千元）:")
mcap_stats = data['mcap'].describe()
print(f"  中位数: {mcap_stats['50%']:.0f}")
print(f"  平均值: {mcap_stats['mean']:.0f}")

print("\n✓ 完成分析！")
