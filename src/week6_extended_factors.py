"""
Week 6: Extended Factor Pool and Factor Attribution
Tasks:
1. Construct FF-3 factors (HML using B/M)
2. Construct reversal factor (REV)
3. Factor attribution analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FACTORS_DIR = PROJECT_ROOT / "data" / "factors"

print("=" * 60)
print("Week 6: Extended Factor Pool & Attribution")
print("=" * 60)

print("\n1. Loading data...")

trd_df = pd.read_csv(RAW_DIR / "TRD_Mnth.csv")
trd_df['Stkcd'] = trd_df['Stkcd'].astype(str).str.zfill(6)
trd_df['Trdmnt'] = pd.to_datetime(trd_df['Trdmnt'], format='%Y-%m')
trd_df = trd_df[['Stkcd', 'Trdmnt', 'Mretwd', 'Msmvttl']].copy()
trd_df.columns = ['stock_id', 'date', 'return', 'mcap']
trd_df['date'] = trd_df['date'].dt.strftime('%Y-%m')

a_share_mask = (
    trd_df['stock_id'].str.startswith('600') |
    trd_df['stock_id'].str.startswith('601') |
    trd_df['stock_id'].str.startswith('603') |
    trd_df['stock_id'].str.startswith('688') |
    trd_df['stock_id'].str.startswith('000') |
    trd_df['stock_id'].str.startswith('001') |
    trd_df['stock_id'].str.startswith('002') |
    trd_df['stock_id'].str.startswith('003') |
    trd_df['stock_id'].str.startswith('300') |
    trd_df['stock_id'].str.startswith('301')
)
trd_df = trd_df[a_share_mask]
print(f"   Trading data: {len(trd_df):,} records, {trd_df['stock_id'].nunique():,} stocks")

equity_df = pd.read_csv(RAW_DIR / "FS_Combas.csv")
equity_df['Stkcd'] = equity_df['Stkcd'].astype(str).str.zfill(6)
equity_df['Accper'] = pd.to_datetime(equity_df['Accper'], format='%Y-%m-%d')
equity_df['fiscal_period'] = equity_df['Accper'].dt.strftime('%Y-%m')
equity_df['A003000000'] = pd.to_numeric(equity_df['A003000000'], errors='coerce')
equity_df = equity_df[equity_df['A003000000'].notna()]
equity_df = equity_df[['Stkcd', 'fiscal_period', 'A003000000']].rename(
    columns={'Stkcd': 'stock_id', 'A003000000': 'equity'}
)
equity_df = equity_df[equity_df['stock_id'].str.match(r'^[036]\d{5}$|^00[0-3]\d{3}$')]
print(f"   Equity data: {len(equity_df):,} records, {equity_df['stock_id'].nunique():,} stocks")

rf_df = pd.read_csv(RAW_DIR / "SMRVR_Intrst.csv")
rf_df.columns = ['date', 'interest']
rf_df['interest'] = pd.to_numeric(rf_df['interest'], errors='coerce') / 100 / 12
rf_df['date'] = pd.to_datetime(rf_df['date'], format='%Y-%m').dt.strftime('%Y-%m')
rf_df = rf_df[['date', 'interest']].rename(columns={'interest': 'rf'})
print(f"   Risk-free rate: {len(rf_df)} records")

print("\n2. Computing B/M and merging...")

trd_df = trd_df[trd_df['date'] >= '2005-01']
trd_df = trd_df[trd_df['date'] <= '2025-12']

def get_fiscal_period_mapping():
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

mapping = get_fiscal_period_mapping()
trd_df['fiscal_period'] = trd_df['date'].map(mapping)

merged = trd_df.merge(
    equity_df[['stock_id', 'fiscal_period', 'equity']],
    on=['stock_id', 'fiscal_period'],
    how='left'
)

merged['bm'] = merged['equity'] / (merged['mcap'] * 1000)
merged['bm'] = merged.groupby('stock_id')['bm'].ffill()

print(f"   B/M non-null ratio: {merged['bm'].notna().mean():.2%}")
print(f"   B/M stats: min={merged['bm'].min():.4f}, median={merged['bm'].median():.4f}, max={merged['bm'].max():.4f}")

merged_clean = merged[merged['return'].abs() <= 0.50]
merged_clean = merged_clean[merged_clean['bm'].abs() <= 10]
merged_clean = merged_clean.dropna(subset=['mcap', 'return', 'bm'])

print(f"   After cleaning: {len(merged_clean):,} records")

def exclude_shell_stocks(df, percentile=0.3):
    result = []
    for date in df['date'].unique():
        month_data = df[df['date'] == date].copy()
        threshold = month_data['mcap'].quantile(percentile)
        filtered = month_data[month_data['mcap'] > threshold]
        result.append(filtered)
    return pd.concat(result, ignore_index=True)

def calculate_value_weighted_return(returns, mcaps):
    valid = ~(np.isnan(returns) | np.isnan(mcaps))
    if valid.sum() == 0:
        return np.nan
    weights = mcaps[valid] / mcaps[valid].sum()
    return np.sum(weights * returns[valid])

def construct_ff3_factors(month_data):
    bm_data = month_data[month_data['bm'] > 0].copy()
    if len(bm_data) < 10:
        return {'HML': np.nan, 'SM': np.nan, 'BL': np.nan}

    median_mcap = bm_data['mcap'].median()
    bm_data['size_group'] = np.where(bm_data['mcap'] < median_mcap, 'S', 'B')

    bm_positive = bm_data[bm_data['bm'] > 0]['bm']
    if len(bm_positive) == 0:
        bm_data['bm_group'] = 'M'
    else:
        v_threshold = bm_positive.quantile(0.7)
        m_high = bm_positive.quantile(0.3)

        def assign_group(bm):
            if pd.isna(bm) or bm <= 0:
                return 'M'
            elif bm >= v_threshold:
                return 'H'
            elif bm <= m_high:
                return 'L'
            else:
                return 'M'

        bm_data['bm_group'] = bm_data['bm'].apply(assign_group)

    portfolios = {}
    for size in ['S', 'B']:
        for bm_grp in ['H', 'M', 'L']:
            subset = bm_data[(bm_data['size_group'] == size) & (bm_data['bm_group'] == bm_grp)]
            if len(subset) > 0:
                ret = calculate_value_weighted_return(subset['return'].values, subset['mcap'].values)
                portfolios[f'{size}_{bm_grp}'] = ret
            else:
                portfolios[f'{size}_{bm_grp}'] = np.nan

    HML = (portfolios.get('S_H', 0) + portfolios.get('B_H', 0)) / 2 - \
          (portfolios.get('S_L', 0) + portfolios.get('B_L', 0)) / 2
    SM = (portfolios.get('S_H', 0) + portfolios.get('S_M', 0) + portfolios.get('S_L', 0)) / 3 - \
         (portfolios.get('B_H', 0) + portfolios.get('B_M', 0) + portfolios.get('B_L', 0)) / 3

    return {'HML': HML, 'SM': SM}

def construct_reversal_factor(month_data, lookback=1):
    month_data = month_data.copy()
    if len(month_data) < 10:
        return np.nan

    median_mcap = month_data['mcap'].median()
    month_data['size_group'] = np.where(month_data['mcap'] < median_mcap, 'S', 'B')

    ret_neg = -month_data['return']
    v_threshold = ret_neg.quantile(0.7)
    m_high = ret_neg.quantile(0.3)

    def assign_group(ret):
        if pd.isna(ret):
            return 'M'
        elif ret >= v_threshold:
            return 'R'
        elif ret <= m_high:
            return 'P'
        else:
            return 'M'

    month_data['ret_group'] = month_data['return'].apply(lambda x: 'M' if pd.isna(x) else ('R' if x <= month_data['return'].quantile(0.3) else ('P' if x >= month_data['return'].quantile(0.7) else 'M')))

    winner = month_data[month_data['ret_group'] == 'R']
    loser = month_data[month_data['ret_group'] == 'P']

    if len(winner) > 0 and len(loser) > 0:
        ret_winner = calculate_value_weighted_return(winner['return'].values, winner['mcap'].values)
        ret_loser = calculate_value_weighted_return(loser['return'].values, loser['mcap'].values)
        return ret_loser - ret_winner
    return np.nan

print("\n3. Constructing FF-3 factors (HML, SM)...")

data_filtered = exclude_shell_stocks(merged_clean)
dates = sorted(data_filtered['date'].unique())

ff3_results = []
for date in dates:
    month_data = data_filtered[data_filtered['date'] == date]
    factors = construct_ff3_factors(month_data)

    rf_val = rf_df[rf_df['date'] == date]['rf'].values
    rf_val = rf_val[0] if len(rf_val) > 0 else 0

    mkt = calculate_value_weighted_return(month_data['return'].values, month_data['mcap'].values)
    mkt_excess = mkt - rf_val if not np.isnan(mkt) else np.nan

    ff3_results.append({
        'date': date,
        'MKT': mkt_excess,
        'SMB': factors.get('SM', np.nan),
        'HML': factors.get('HML', np.nan)
    })

ff3_factors = pd.DataFrame(ff3_results)
ff3_factors.to_csv(FACTORS_DIR / "ff3_monthly.csv", index=False)
print(f"   FF-3 factors saved: {len(ff3_factors)} months")
print(f"   MKT mean: {ff3_factors['MKT'].mean()*100:.2f}%")
print(f"   SMB mean: {ff3_factors['SMB'].mean()*100:.2f}%")
print(f"   HML mean: {ff3_factors['HML'].mean()*100:.2f}%")

print("\n4. Constructing reversal factor (REV)...")

rev_results = []
for date in dates:
    month_data = data_filtered[data_filtered['date'] == date]

    median_mcap = month_data['mcap'].median()
    month_data = month_data.copy()
    month_data['size_group'] = np.where(month_data['mcap'] < median_mcap, 'S', 'B')

    month_data['ret_neg'] = -month_data['return']

    v_threshold = month_data['ret_neg'].quantile(0.7)
    m_high = month_data['ret_neg'].quantile(0.3)

    month_data['rev_group'] = month_data['ret_neg'].apply(
        lambda x: 'R' if x >= v_threshold else ('P' if x <= m_high else 'M')
    )

    rev_factor = np.nan
    for size in ['S', 'B']:
        size_data = month_data[month_data['size_group'] == size]
        winner = size_data[size_data['rev_group'] == 'R']
        loser = size_data[size_data['rev_group'] == 'P']

        if len(winner) > 0 and len(loser) > 0:
            ret_w = calculate_value_weighted_return(winner['return'].values, winner['mcap'].values)
            ret_l = calculate_value_weighted_return(loser['return'].values, loser['mcap'].values)
            rev = (ret_l - ret_w) if not (np.isnan(ret_l) or np.isnan(ret_w)) else np.nan
            if not np.isnan(rev):
                if np.isnan(rev_factor):
                    rev_factor = rev
                else:
                    rev_factor = (rev_factor + rev) / 2

    rev_results.append({'date': date, 'REV': rev_factor})

rev_factors = pd.DataFrame(rev_results)
rev_factors.to_csv(FACTORS_DIR / "rev_monthly.csv", index=False)
print(f"   REV factor saved: {len(rev_factors)} months")
print(f"   REV mean: {rev_factors['REV'].mean()*100:.2f}%")

print("\n5. Loading CH-3 factors for comparison...")
ch3_factors = pd.read_csv(FACTORS_DIR / "ch3_monthly.csv")

print("\n" + "=" * 60)
print("Factor Summary (2005-2025)")
print("=" * 60)

print("\nCH-3 Factors:")
print(f"   MKT: {ch3_factors['MKT'].mean()*100:.2f}%/month")
print(f"   SMB: {ch3_factors['SMB'].mean()*100:.2f}%/month")
print(f"   VMG: {ch3_factors['VMG'].mean()*100:.2f}%/month")
print(f"   Corr(SMB,VMG): {ch3_factors['SMB'].corr(ch3_factors['VMG']):.3f}")

print("\nFF-3 Factors:")
print(f"   MKT: {ff3_factors['MKT'].mean()*100:.2f}%/month")
print(f"   SMB: {ff3_factors['SMB'].mean()*100:.2f}%/month")
print(f"   HML: {ff3_factors['HML'].mean()*100:.2f}%/month")
print(f"   Corr(SMB,HML): {ff3_factors['SMB'].corr(ff3_factors['HML']):.3f}")

print("\nREV Factor:")
print(f"   REV: {rev_factors['REV'].mean()*100:.2f}%/month")

print("\n" + "=" * 60)
print("Week 6 Factor Construction Complete!")
print("=" * 60)
