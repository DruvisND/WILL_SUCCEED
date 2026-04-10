"""CH-3因子构造实验"""
import pandas as pd
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factors import construct_ch3_factor, validate_factors

print("=" * 60)
print("CH-3 中国版三因子模型构造")
print("=" * 60)

# 加载数据
print("\n1. 加载数据...")
data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "monthly_data.csv")
rf = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "rf_monthly.csv")

print(f"   - 股票月度数据: {len(data)} 条记录")
print(f"   - 无风险利率: {len(rf)} 条记录")
print(f"   - 日期范围: {data['date'].min()} ~ {data['date'].max()}")

# 构造CH-3因子
print("\n2. 构造CH-3因子...")
factors = construct_ch3_factor(data, rf)

# 保存因子数据
factors.to_csv(PROJECT_ROOT / "data" / "factors" / "ch3_monthly.csv", index=False)
print(f"   - 保存至: {PROJECT_ROOT / 'data' / 'factors' / 'ch3_monthly.csv'}")

# 验证因子
print("\n3. 验证因子质量...")
validation = validate_factors(factors)

print("\n   因子统计特征:")
print("   " + "-" * 50)
print(f"   {'因子':<8} {'月均收益':>12} {'标准差':>12} {'t统计量':>12}")
print("   " + "-" * 50)
for col, v in validation['stats'].items():
    print(f"   {col:<8} {v['mean']*100:>11.2f}% {v['std']*100:>11.2f}% {v['t_stat']:>11.2f}")

print("\n   论文参考值 (Liu, Stambaugh & Yuan, 2019):")
print("   " + "-" * 50)
print(f"   {'因子':<8} {'月均收益':>12}")
print("   " + "-" * 50)
print(f"   {'MKT':<8} {'0.66%':>12}")
print(f"   {'SMB':<8} {'1.03%':>12}")
print(f"   {'VMG':<8} {'1.14%':>12}")

print("\n   因子相关系数:")
print("   " + "-" * 50)
corr_matrix = factors[['MKT', 'SMB', 'VMG']].corr()
print(corr_matrix.round(3).to_string())

# 累计收益
print("\n4. 计算累计收益...")
factors_cum = factors.copy()
for col in ['MKT', 'SMB', 'VMG']:
    factors_cum[f'{col}_cum'] = (1 + factors_cum[col]).cumprod()

# 保存结果
factors_cum.to_csv(PROJECT_ROOT / "results" / "ch3_with_cumsum.csv", index=False)
print(f"   - 保存至: {PROJECT_ROOT / 'results' / 'ch3_with_cumsum.csv'}")

print("\n" + "=" * 60)
print("CH-3因子构造完成!")
print("=" * 60)