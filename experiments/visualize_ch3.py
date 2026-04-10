"""CH-3因子可视化"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
factors = pd.read_csv('project2/data/factors/ch3_monthly.csv')
factors['date'] = pd.to_datetime(factors['date'])
factors = factors.sort_values('date')

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 累计收益曲线
ax1 = axes[0, 0]
for col, color in [('MKT', '#1f77b4'), ('SMB', '#ff7f0e'), ('VMG', '#2ca02c')]:
    cum_ret = (1 + factors[col]).cumprod() - 1
    ax1.plot(factors['date'], cum_ret * 100, label=col, linewidth=1.5, color=color)

ax1.set_title('CH-3 Factor Cumulative Returns (2000-2025)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Return (%)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# 2. 因子相关性热力图
ax2 = axes[0, 1]
corr = factors[['MKT', 'SMB', 'VMG']].corr()
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f',
            ax=ax2, vmin=-1, vmax=1, square=True)
ax2.set_title('Factor Correlation Matrix', fontsize=12, fontweight='bold')

# 3. 因子收益分布
ax3 = axes[1, 0]
for col, color in [('MKT', '#1f77b4'), ('SMB', '#ff7f0e'), ('VMG', '#2ca02c')]:
    ax3.hist(factors[col] * 100, bins=50, alpha=0.5, label=col, color=color)
ax3.set_title('Factor Return Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Monthly Return (%)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

# 4. 年度收益对比
ax4 = axes[1, 1]
factors['year'] = factors['date'].dt.year
yearly = factors.groupby('year')[['MKT', 'SMB', 'VMG']].mean() * 12 * 100

x = np.arange(len(yearly))
width = 0.25

bars1 = ax4.bar(x - width, yearly['MKT'], width, label='MKT', color='#1f77b4')
bars2 = ax4.bar(x, yearly['SMB'], width, label='SMB', color='#ff7f0e')
bars3 = ax4.bar(x + width, yearly['VMG'], width, label='VMG', color='#2ca02c')

ax4.set_title('Annual Factor Returns', fontsize=12, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Annual Return (%)')
ax4.set_xticks(x[::5])
ax4.set_xticklabels(yearly.index[::5], rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('project2/results/ch3_analysis.png', dpi=150, bbox_inches='tight')
print("保存: project2/results/ch3_analysis.png")

# 单独保存累计收益图
fig2, ax = plt.subplots(figsize=(12, 6))
for col, color in [('MKT', '#1f77b4'), ('SMB', '#ff7f0e'), ('VMG', '#2ca02c')]:
    cum_ret = (1 + factors[col]).cumprod()
    ax.plot(factors['date'], cum_ret, label=f'{col} Factor', linewidth=1.5, color=color)

ax.set_title('CH-3 Factor Cumulative Returns (2000-2025)\nLiu, Stambaugh & Yuan (2019) Model', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Cumulative Return (1 = Initial)', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)

# 添加注释
ax.text(0.02, 0.98, 'Data: Simulated (Replace with real data)', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top', 
        style='italic', color='gray')

plt.tight_layout()
plt.savefig('project2/results/ch3_cumulative.png', dpi=150, bbox_inches='tight')
print("保存: project2/results/ch3_cumulative.png")

# 单独保存相关性热力图
fig3, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.3f',
            ax=ax, vmin=-1, vmax=1, square=True, 
            annot_kws={'size': 14, 'fontweight': 'bold'})
ax.set_title('CH-3 Factor Correlation Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('project2/results/ch3_correlation.png', dpi=150, bbox_inches='tight')
print("保存: project2/results/ch3_correlation.png")

print("\n可视化完成！")