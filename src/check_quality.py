import pandas as pd
import numpy as np

data = pd.read_csv('../data/processed/monthly_data.csv')

print('='*60)
print('数据质量检查报告')
print('='*60)

print('\n1. 基本信息:')
print(f'   总记录数: {len(data):,}')
print(f'   股票数: {data["stock_id"].nunique():,}')
print(f'   日期范围: {data["date"].min()} ~ {data["date"].max()}')

print('\n2. 市值(mcap)统计:')
print(f'   缺失值: {data["mcap"].isna().sum():,}')
print(f'   均值: {data["mcap"].mean():,.0f} 千元')
print(f'   中位数: {data["mcap"].median():,.0f} 千元')
print(f'   最小值: {data["mcap"].min():,.0f} 千元')
print(f'   最大值: {data["mcap"].max():,.0f} 千元')

print('\n3. 收益率(return)统计:')
print(f'   缺失值: {data["return"].isna().sum():,}')
print(f'   均值: {data["return"].mean()*100:.2f}%')
print(f'   中位数: {data["return"].median()*100:.2f}%')
print(f'   最小值: {data["return"].min()*100:.2f}%')
print(f'   最大值: {data["return"].max()*100:.2f}%')

print('\n4. EP统计:')
print(f'   缺失值: {data["ep"].isna().sum():,}')
print(f'   均值: {data["ep"].mean():.4f}')
print(f'   中位数: {data["ep"].median():.4f}')
print(f'   最小值: {data["ep"].min():.4f}')
print(f'   最大值: {data["ep"].max():.4f}')

print('\n5. 极端值检查:')
print('\n   市值极端值 (前10大):')
top10_mcap = data.nlargest(10, 'mcap')[['date','stock_id','mcap']]
for _, row in top10_mcap.iterrows():
    print(f'   {row["date"]} {row["stock_id"]} {row["mcap"]:,.0f}')

print('\n   市值极端值 (前10小):')
bottom10_mcap = data.nsmallest(10, 'mcap')[['date','stock_id','mcap']]
for _, row in bottom10_mcap.iterrows():
    print(f'   {row["date"]} {row["stock_id"]} {row["mcap"]:,.0f}')

print('\n   收益率极端值 (最大10):')
top10_ret = data.nlargest(10, 'return')[['date','stock_id','return']]
for _, row in top10_ret.iterrows():
    print(f'   {row["date"]} {row["stock_id"]} {row["return"]*100:.2f}%')

print('\n   收益率极端值 (最小10):')
bottom10_ret = data.nsmallest(10, 'return')[['date','stock_id','return']]
for _, row in bottom10_ret.iterrows():
    print(f'   {row["date"]} {row["stock_id"]} {row["return"]*100:.2f}%')

print('\n   EP极端值 (最大10):')
top10_ep = data.nlargest(10, 'ep')[['date','stock_id','ep']]
for _, row in top10_ep.iterrows():
    print(f'   {row["date"]} {row["stock_id"]} {row["ep"]:.4f}')

print('\n   EP极端值 (最小10):')
bottom10_ep = data.nsmallest(10, 'ep')[['date','stock_id','ep']]
for _, row in bottom10_ep.iterrows():
    print(f'   {row["date"]} {row["stock_id"]} {row["ep"]:.4f}')

print('\n6. 分位数分布:')
print('\n   市值百分位:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'   {p}%: {data["mcap"].quantile(p/100):,.0f} 千元')

print('\n   收益率百分位:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'   {p}%: {data["return"].quantile(p/100)*100:.2f}%')

print('\n   EP百分位:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'   {p}%: {data["ep"].quantile(p/100):.4f}')

print('\n7. 异常值检测:')
print('\n   收益率绝对值 > 50% 的记录数:', len(data[data['return'].abs() > 0.5]))
print('   收益率绝对值 > 30% 的记录数:', len(data[data['return'].abs() > 0.3]))
print('   EP > 10 的记录数:', len(data[data['ep'] > 10]))
print('   EP < -5 的记录数:', len(data[data['ep'] < -5]))
