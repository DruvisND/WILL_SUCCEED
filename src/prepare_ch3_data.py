"""
从真实raw数据构建CH-3因子所需数据
处理流程：
1. 加载TRD_Mnth.csv获取月度收益率和市值
2. 加载FS_Comins.csv获取净利润，计算EP = 净利润/总市值
3. 按财报披露规则对齐（避免前视偏差）
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def get_fiscal_period_mapping():
    """交易月份到财报期的映射（避免前视偏差）

    披露规则：
    - 年报（12月）：最晚次年4/30
    - 一季报（03月）：最晚4/30
    - 半年报（06月）：最晚8/31
    - 三季报（09月）：最晚10/31

    映射规则（选择最早可得的财报）：
    - 1-4月: 上年12月年报（次年4/30前披露）
    - 5-8月: 本年03月一季报（4/30前披露）
    - 9-10月: 本年06月半年报（8/31前披露）
    - 11-12月: 本年09月三季报（10/31前披露）
    """
    mapping = {}
    for year in range(1990, 2026):
        for month in range(1, 13):
            if month <= 4:
                fiscal_period = f"{year-1}-12"
            elif month <= 8:
                fiscal_period = f"{year}-03"
            elif month <= 10:
                fiscal_period = f"{year}-06"
            else:
                fiscal_period = f"{year}-09"
            trading_month = f"{year}-{month:02d}"
            mapping[trading_month] = fiscal_period
    return mapping

def standardize_fiscal_date(date_str):
    """将财务日期标准化为YYYY-MM格式

    处理非标准日期：
    - 01-01日期（如1999-01-01）转为上一年12月（1998-12）
    - 其他日期取月份部分
    """
    try:
        dt = pd.to_datetime(date_str)
        month = dt.month
        year = dt.year
        if month == 1 and dt.day == 1:
            return f"{year-1}-12"
        return f"{year}-{month:02d}"
    except:
        return None

def load_trd_mnth():
    """加载月度交易数据"""
    print("1. 加载TRD_Mnth.csv...")
    df = pd.read_csv(RAW_DIR / "TRD_Mnth.csv")
    df['Stkcd'] = df['Stkcd'].astype(str).str.zfill(6)
    df['Trdmnt'] = pd.to_datetime(df['Trdmnt'], format='%Y-%m')

    df = df[['Stkcd', 'Trdmnt', 'Mretwd', 'Msmvttl']].copy()
    df.columns = ['stock_id', 'date', 'return', 'mcap']

    a_share_mask = (
        df['stock_id'].str.startswith('600') |
        df['stock_id'].str.startswith('601') |
        df['stock_id'].str.startswith('603') |
        df['stock_id'].str.startswith('688') |
        df['stock_id'].str.startswith('000') |
        df['stock_id'].str.startswith('001') |
        df['stock_id'].str.startswith('002') |
        df['stock_id'].str.startswith('003') |
        df['stock_id'].str.startswith('300') |
        df['stock_id'].str.startswith('301')
    )
    df = df[a_share_mask]

    df['date'] = df['date'].dt.strftime('%Y-%m')
    df = df[df['return'].notna() & df['mcap'].notna()]
    df['mcap'] = pd.to_numeric(df['mcap'], errors='coerce')
    df['return'] = pd.to_numeric(df['return'], errors='coerce')
    df = df.dropna(subset=['mcap', 'return'])

    print(f"   加载完成: {len(df)} 条记录")
    print(f"   日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   股票数量: {df['stock_id'].nunique()}")
    return df

def load_financial_data():
    """加载财务数据：使用FS_Comins.csv的净利润"""
    print("\n2. 加载财务数据...")

    print("   加载 FS_Comins.csv (净利润)...")
    df = pd.read_csv(RAW_DIR / "FS_Comins.csv")
    df['Stkcd'] = df['Stkcd'].astype(str).str.zfill(6)
    df['B002000000'] = pd.to_numeric(df['B002000000'], errors='coerce')

    a_share_mask = (
        df['Stkcd'].str.startswith('600') |
        df['Stkcd'].str.startswith('601') |
        df['Stkcd'].str.startswith('603') |
        df['Stkcd'].str.startswith('688') |
        df['Stkcd'].str.startswith('000') |
        df['Stkcd'].str.startswith('001') |
        df['Stkcd'].str.startswith('002') |
        df['Stkcd'].str.startswith('003') |
        df['Stkcd'].str.startswith('300') |
        df['Stkcd'].str.startswith('301')
    )
    df = df[a_share_mask]

    df['fiscal_period'] = df['Accper'].apply(standardize_fiscal_date)
    df = df[df['fiscal_period'].notna()]

    df = df[['Stkcd', 'fiscal_period', 'B002000000']].rename(
        columns={'Stkcd': 'stock_id', 'B002000000': 'net_profit'}
    )
    df = df[df['net_profit'].notna()].copy()
    df = df.sort_values(['stock_id', 'fiscal_period'])

    print(f"   净利润记录: {len(df)} 条")
    print(f"   数据覆盖: {sorted(df['fiscal_period'].unique())[:10]} ...")
    print(f"   股票数量: {df['stock_id'].nunique()}")

    return df

def compute_ep_and_merge(trd_df, fin_data):
    """计算EP并对齐到交易月份"""
    print("\n3. 计算EP并对齐...")

    mapping = get_fiscal_period_mapping()

    trd_df = trd_df.copy()
    trd_df['fiscal_period'] = trd_df['date'].map(mapping)

    merged = trd_df.merge(
        fin_data[['stock_id', 'fiscal_period', 'net_profit']],
        on=['stock_id', 'fiscal_period'],
        how='left'
    )

    merged['ep'] = merged['net_profit'] / (merged['mcap'] * 1000)

    merged['ep'] = merged.groupby('stock_id')['ep'].ffill()

    print(f"   EP非空记录比例: {merged['ep'].notna().mean():.2%}")

    ep_valid = merged['ep'].dropna()
    if len(ep_valid) > 0:
        print(f"   EP统计: min={ep_valid.min():.4f}, median={ep_valid.median():.4f}, max={ep_valid.max():.4f}")

    return merged

def filter_and_save(df):
    """筛选有效数据并保存"""
    print("\n4. 筛选有效数据...")

    df = df[df['date'] >= '2005-01']
    df = df[df['date'] <= '2025-12']

    df = df[['date', 'stock_id', 'mcap', 'ep', 'return']].copy()
    df = df.dropna(subset=['mcap', 'return'])

    print(f"   清洗前记录: {len(df):,}")

    print("   清洗异常数据...")
    original_count = len(df)

    df = df[df['return'].abs() <= 0.50]
    print(f"   剔除收益率绝对值>50%: {original_count - len(df):,} 条")
    removed = original_count - len(df)
    original_count = len(df)

    df = df[df['ep'].abs() <= 10]
    print(f"   剔除EP绝对值>10: {original_count - len(df):,} 条")

    df = df.sort_values(['date', 'stock_id'])

    output_path = PROCESSED_DIR / "monthly_data.csv"
    df.to_csv(output_path, index=False)
    print(f"   保存至: {output_path}")
    print(f"   最终记录: {len(df):,}")
    print(f"   日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   股票数: {df['stock_id'].nunique()}")

    return df

def load_rf_data():
    """从真实CSV加载无风险利率数据"""
    print("   加载 SMRVR_Intrst.csv...")
    rf_raw = pd.read_csv(RAW_DIR / "SMRVR_Intrst.csv")
    rf_raw.columns = ['date', 'interest']
    rf_raw['interest'] = pd.to_numeric(rf_raw['interest'], errors='coerce') / 100 / 12
    rf_raw['date'] = pd.to_datetime(rf_raw['date'], format='%Y-%m').dt.strftime('%Y-%m')
    rf_raw = rf_raw[['date', 'interest']].rename(columns={'interest': 'rf'})
    print(f"   加载完成: {len(rf_raw)} 条记录")
    return rf_raw

def build_rf_data():
    """从真实数据构建无风险利率数据"""
    print("\n5. 构建无风险利率数据...")
    rf_df = load_rf_data()
    rf_path = PROCESSED_DIR / "rf_monthly.csv"
    rf_df.to_csv(rf_path, index=False)
    print(f"   保存至: {rf_path}")
    return rf_df

if __name__ == '__main__':
    print("=" * 60)
    print("从真实raw数据构建CH-3所需数据")
    print("净利润来源: FS_Comins.csv")
    print("=" * 60)

    trd_df = load_trd_mnth()
    fin_data = load_financial_data()
    merged = compute_ep_and_merge(trd_df, fin_data)
    filter_and_save(merged)
    build_rf_data()

    print("\n" + "=" * 60)
    print("✓ 数据处理完成（使用FS_Comins.csv净利润）！")
    print("=" * 60)