"""
生成模拟数据用于CH-3因子构造
用户可用真实数据替换
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# 生成日期范围：2000-2025年月度数据
dates = pd.date_range(start='2000-01', end='2025-12', freq='ME')
dates = [d.strftime('%Y-%m') for d in dates]

# 模拟300只股票
n_stocks = 300
stock_ids = [f'{600000 + i:06d}' for i in range(n_stocks)]

print(f"生成日期: {len(dates)} 个月")
print(f"生成股票: {n_stocks} 只")

# 生成虚拟数据
data_records = []
for date in dates:
    year = int(date[:4])
    month = int(date[5:7])
    
    for stock_id in stock_ids:
        # 市值：对数正态分布，模拟小盘股效应
        mcap = np.random.lognormal(mean=10, sigma=1.5)
        
        # EP：正态分布，均值0.03，负值归入成长组
        ep = np.random.normal(0.03, 0.05)
        
        # 收益率：模拟三因子效应
        base_return = np.random.normal(0.01, 0.05)
        
        # ✓ 修正：小盘股应获得更高收益（小盘溢价为正）
        size_factor = 0.002 * (12 - np.log(mcap))  # 移除负号，小盘股溢价
        
        # ✓ 修正：高EP股票应获得更高收益（价值溢价为正）
        value_factor = 0.015 * ep  # 价值股溢价
        
        market_impact = np.random.normal(0, 0.08)
        
        ret = base_return + size_factor + value_factor + market_impact
        
        data_records.append({
            'date': date,
            'stock_id': stock_id,
            'mcap': mcap,
            'ep': ep,
            'return': ret
        })

df = pd.DataFrame(data_records)

# 保存处理后的数据
df.to_csv('project2/data/processed/monthly_data.csv', index=False)
print("保存: project2/data/processed/monthly_data.csv")

# 生成无风险利率数据（一年期存款利率，约2%）
rf_data = []
for date in dates:
    year = int(date[:4])
    # 2008年前较高，之后降低，近年略有回升
    if year < 2008:
        rf = 0.025
    elif year < 2015:
        rf = 0.0225
    elif year < 2020:
        rf = 0.015
    else:
        rf = 0.018
    
    rf_data.append({
        'date': date,
        'rf': rf / 12  # 月度化
    })

rf_df = pd.DataFrame(rf_data)
rf_df.to_csv('project2/data/processed/rf_monthly.csv', index=False)
print("保存: project2/data/processed/rf_monthly.csv")

print(f"\n模拟数据生成完成！")
print(f"总记录数: {len(df)}")