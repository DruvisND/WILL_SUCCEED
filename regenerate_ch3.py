"""重新生成修正后的CH-3因子"""
import pandas as pd
from src.factors import construct_ch3_factor, validate_factors

print("=" * 60)
print("重新生成修正版CH-3因子")
print("=" * 60)

# 加载数据
data = pd.read_csv('data/processed/monthly_data.csv')
rf = pd.read_csv('data/processed/rf_monthly.csv')

print(f"\n数据加载完成:")
print(f"  月度数据: {len(data)} 条记录")
print(f"  无风险利率: {len(rf)} 条记录")

# 构造因子
print("\n构造CH-3因子...")
factors = construct_ch3_factor(data, rf)

# 保存
factors.to_csv('data/factors/ch3_monthly.csv', index=False)
print(f"  保存至: data/factors/ch3_monthly.csv ({len(factors)} 月)")

# 验证
print("\n验证因子质量...")
stats = validate_factors(factors)

print("\n" + "=" * 60)
print("修正后的因子统计")
print("=" * 60)
print(f"{'因子':<6} {'月均收益':>10} {'标准差':>10} {'t统计量':>10}")
print("-" * 60)
for col in ['MKT', 'SMB', 'VMG']:
    vals = stats['stats'][col]
    print(f"{col:<6} {vals['mean']:>9.4%}  {vals['std']:>9.4%}  {vals['t_stat']:>9.3f}")

print("\n" + "=" * 60)
print("论文参考值（2000-2016）")
print("=" * 60)
print(f"{'因子':<6} {'月均收益':>10} {'标准差':>10} {'t统计量':>10}")
print("-" * 60)
print(f"{'MKT':<6} {'0.66%':>9}  {'8.09%':>9}  {'1.160':>9}")
print(f"{'SMB':<6} {'1.03%':>9}  {'4.52%':>9}  {'3.250':>9}")
print(f"{'VMG':<6} {'1.14%':>9}  {'3.75%':>9}  {'4.340':>9}")

print("\n" + "=" * 60)
print("因子相关系数矩阵")
print("=" * 60)
corr = factors[['MKT', 'SMB', 'VMG']].corr()
print(corr.to_string())

print("\n✓ 完成！")
