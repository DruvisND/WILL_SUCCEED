"""
CH-3因子构造模块
实现Liu, Stambaugh & Yuan (2019)的中国版三因子模型
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def exclude_shell_stocks(df: pd.DataFrame, percentile: float = 0.3) -> pd.DataFrame:
    """
    剔除市值最小的30%股票（壳资源）
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含 date, stock_id, mcap 列
    percentile : float
        剔除比例，默认0.3（30%）
    
    Returns:
    --------
    pd.DataFrame : 剔除后的数据
    """
    def exclude_per_month(group):
        threshold = group['mcap'].quantile(percentile)
        return group[group['mcap'] > threshold]
    
    result_列表 = []
    for date in df['date'].unique():
        month_数据 = df[df['date'] == date].copy()
        threshold = month_数据['mcap'].quantile(percentile)
        filtered = month_数据[month_数据['mcap'] > threshold]
        result_列表.append(filtered)
    return pd.concat(result_列表, ignore_index=True)


def calculate_value_weighted_return(returns: np.ndarray, mcaps: np.ndarray) -> float:
    """
    计算市值加权组合收益
    
    Parameters:
    -----------
    returns : np.ndarray
        股票收益率
    mcaps : np.ndarray
        对应市值
    
    Returns:
    --------
    float : 市值加权收益
    """
    valid = ~(np.isnan(returns) | np.isnan(mcaps))
    if valid.sum() == 0:
        return np.nan
    
    returns_valid = returns[valid]
    mcaps_valid = mcaps[valid]
    
    weights = mcaps_valid / mcaps_valid.sum()
    return np.sum(weights * returns_valid)


def construct_2x3_portfolios(df: pd.DataFrame) -> Dict[str, float]:
    """
    2x3分组构造6个市值加权组合
    
    分组方法:
    - 按市值中位数分为S(小盘)和B(大盘)
    - 按EP分为V(价值,前30%)、M(中性,中40%)、G(成长,后30%)
    - 负EP归入G组
    
    Parameters:
    -----------
    df : pd.DataFrame
        当月数据，包含 mcap, ep, return 列
    
    Returns:
    --------
    Dict[str, float] : 6个组合的收益率
        'S_V', 'S_M', 'S_G', 'B_V', 'B_M', 'B_G'
    """
    # 处理负EP：归入成长组
    df = df.copy()
    df.loc[df['ep'] <= 0, 'ep'] = np.nan
    
    # 按市值中位数分组
    median_mcap = df['mcap'].median()
    df['size_group'] = np.where(df['mcap'] < median_mcap, 'S', 'B')
    
    # 按EP分组（正EP部分）
    ep_valid = df[df['ep'] > 0]['ep']
    if len(ep_valid) == 0:
        # 所有EP都为负，全部归入成长组
        df['ep_group'] = 'G'
    else:
        v_threshold = ep_valid.quantile(0.7)  # 前30%为价值
        m_high = ep_valid.quantile(0.3)        # 后30%为成长
        
        def assign_ep_group(ep):
            if pd.isna(ep):
                return 'G'  # 负EP归入成长
            elif ep >= v_threshold:
                return 'V'  # 价值
            elif ep <= m_high:
                return 'G'  # 成长
            else:
                return 'M'  # 中性
        
        df['ep_group'] = df['ep'].apply(assign_ep_group)
    
    # 计算6个组合的市值加权收益
    portfolios = {}
    for size in ['S', 'B']:
        for ep in ['V', 'M', 'G']:
            subset = df[(df['size_group'] == size) & (df['ep_group'] == ep)]
            if len(subset) > 0:
                ret = calculate_value_weighted_return(
                    subset['return'].values,
                    subset['mcap'].values
                )
                portfolios[f'{size}_{ep}'] = ret
            else:
                portfolios[f'{size}_{ep}'] = np.nan
    
    return portfolios


def construct_smb(vmg_portfolios: Dict[str, float]) -> float:
    """
    构造SMB因子（小盘因子）
    SMB = (S_V + S_M + S_G)/3 - (B_V + B_M + B_G)/3
    
    Parameters:
    -----------
    vmg_portfolios : Dict[str, float]
        6个组合的收益率
    
    Returns:
    --------
    float : SMB因子收益
    """
    small = (vmg_portfolios.get('S_V', 0) + 
             vmg_portfolios.get('S_M', 0) + 
             vmg_portfolios.get('S_G', 0)) / 3
    
    big = (vmg_portfolios.get('B_V', 0) + 
           vmg_portfolios.get('B_M', 0) + 
           vmg_portfolios.get('B_G', 0)) / 3
    
    return small - big


def construct_vmg(vmg_portfolios: Dict[str, float]) -> float:
    """
    构造VMG因子（价值因子）
    VMG = (S_V + B_V)/2 - (S_G + B_G)/2
    
    Parameters:
    -----------
    vmg_portfolios : Dict[str, float]
        6个组合的收益率
    
    Returns:
    --------
    float : VMG因子收益
    """
    value = (vmg_portfolios.get('S_V', 0) + vmg_portfolios.get('B_V', 0)) / 2
    growth = (vmg_portfolios.get('S_G', 0) + vmg_portfolios.get('B_G', 0)) / 2
    
    return value - growth


def construct_market(df: pd.DataFrame) -> float:
    """
    构造市场因子（剩余70%股票的市值加权收益）
    
    Parameters:
    -----------
    df : pd.DataFrame
        剔除壳资源后的数据
    
    Returns:
    --------
    float : 市场因子收益
    """
    return calculate_value_weighted_return(
        df['return'].values,
        df['mcap'].values
    )


def construct_ch3_factor(data: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """
    构造CH-3三因子
    
    步骤：
    1. 每月剔除市值最小的30%股票
    2. 对剩余70%按市值和EP分组
    3. 计算6个市值加权组合
    4. 构造MKT, SMB, VMG因子
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含 date, stock_id, mcap, ep, return 列
    rf : pd.DataFrame
        包含 date, rf 列（月度无风险利率）
    
    Returns:
    --------
    pd.DataFrame : CH-3因子数据
        列: date, MKT, SMB, VMG
    """
    # 剔除壳资源股票
    data_filtered = exclude_shell_stocks(data)
    
    # 获取所有日期
    dates = sorted(data_filtered['date'].unique())
    
    results = []
    for date in dates:
        month_data = data_filtered[data_filtered['date'] == date]
        
        if len(month_data) == 0:
            continue
        
        # 构造6个组合
        portfolios = construct_2x3_portfolios(month_data)
        
        # 计算三个因子
        mkt = construct_market(month_data)
        smb = construct_smb(portfolios)
        vmg = construct_vmg(portfolios)
        
        # 获取无风险利率
        rf_value = rf[rf['date'] == date]['rf'].values
        rf_value = rf_value[0] if len(rf_value) > 0 else 0
        
        # 只对MKT减去无风险利率（市场超额收益）
        # SMB和VMG本身已是组合差值，不应再减rf
        mkt_excess = mkt - rf_value if not np.isnan(mkt) else np.nan
        smb_excess = smb  # 保持原值，不减rf
        vmg_excess = vmg  # 保持原值，不减rf
        
        results.append({
            'date': date,
            'MKT': mkt_excess,
            'SMB': smb_excess,
            'VMG': vmg_excess
        })
    
    return pd.DataFrame(results)


def validate_factors(factors: pd.DataFrame) -> Dict:
    """
    验证因子质量
    
    计算：
    - 均值、标准差、t统计量
    - 与论文对比
    - 相关系数
    
    Parameters:
    -----------
    factors : pd.DataFrame
        CH-3因子数据
    
    Returns:
    --------
    Dict : 验证结果
    """
    stats = {}
    for col in ['MKT', 'SMB', 'VMG']:
        series = factors[col].dropna()
        mean = series.mean()
        std = series.std()
        t_stat = mean / (std / np.sqrt(len(series)))
        
        stats[col] = {
            'mean': mean,
            'std': std,
            't_stat': t_stat,
            'n': len(series)
        }
    
    corr = factors[['MKT', 'SMB', 'VMG']].corr().to_dict()
    
    return {'stats': stats, 'corr': corr}


def construct_ff3_factor(data: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """
    构造FF-3三因子（HML使用B/M）
    
    FF-3 vs CH-3:
    - CH-3使用EP作为价值指标，剔除最小30%壳资源
    - FF-3使用B/M作为价值指标，使用全部股票
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含 date, stock_id, mcap, bm, return 列
    rf : pd.DataFrame
        包含 date, rf 列
    
    Returns:
    --------
    pd.DataFrame : FF-3因子数据
        列: date, MKT, SMB, HML
    """
    def construct_2x3_bm_portfolios(df: pd.DataFrame) -> Dict[str, float]:
        df = df.copy()
        df.loc[df['bm'] <= 0, 'bm'] = np.nan
        
        median_mcap = df['mcap'].median()
        df['size_group'] = np.where(df['mcap'] < median_mcap, 'S', 'B')
        
        bm_valid = df[df['bm'] > 0]['bm']
        if len(bm_valid) == 0:
            df['bm_group'] = 'M'
        else:
            v_threshold = bm_valid.quantile(0.7)
            m_high = bm_valid.quantile(0.3)
            
            def assign_bm_group(bm):
                if pd.isna(bm):
                    return 'M'
                elif bm >= v_threshold:
                    return 'H'
                elif bm <= m_high:
                    return 'L'
                else:
                    return 'M'
            
            df['bm_group'] = df['bm'].apply(assign_bm_group)
        
        portfolios = {}
        for size in ['S', 'B']:
            for bm_grp in ['H', 'M', 'L']:
                subset = df[(df['size_group'] == size) & (df['bm_group'] == bm_grp)]
                if len(subset) > 0:
                    ret = calculate_value_weighted_return(
                        subset['return'].values,
                        subset['mcap'].values
                    )
                    portfolios[f'{size}_{bm_grp}'] = ret
                else:
                    portfolios[f'{size}_{bm_grp}'] = np.nan
        
        return portfolios
    
    results = []
    dates = sorted(data['date'].unique())
    
    for date in dates:
        month_data = data[data['date'] == date].copy()
        
        if len(month_data) == 0:
            continue
        
        bm_data = month_data[month_data['bm'] > 0].copy()
        if len(bm_data) < 10:
            continue
        
        portfolios = construct_2x3_bm_portfolios(bm_data)
        
        HML = (portfolios.get('S_H', 0) + portfolios.get('B_H', 0)) / 2 - \
              (portfolios.get('S_L', 0) + portfolios.get('B_L', 0)) / 2
        SMB = (portfolios.get('S_H', 0) + portfolios.get('S_M', 0) + portfolios.get('S_L', 0)) / 3 - \
              (portfolios.get('B_H', 0) + portfolios.get('B_M', 0) + portfolios.get('B_L', 0)) / 3
        
        mkt = calculate_value_weighted_return(month_data['return'].values, month_data['mcap'].values)
        
        rf_value = rf[rf['date'] == date]['rf'].values
        rf_value = rf_value[0] if len(rf_value) > 0 else 0
        
        mkt_excess = mkt - rf_value if not np.isnan(mkt) else np.nan
        
        results.append({
            'date': date,
            'MKT': mkt_excess,
            'SMB': SMB,
            'HML': HML
        })
    
    return pd.DataFrame(results)


def construct_reversal_factor(data: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """
    构造反转因子（1个月反转）
    
    中国市场特点：
    - 没有动量效应，只有反转效应
    - 做多输家（过去1月表现差），做空赢家（过去1月表现好）
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含 date, stock_id, mcap, return 列
    lookback : int
        回看期数，默认1个月
    
    Returns:
    --------
    pd.DataFrame : 反转因子数据
        列: date, REV
    """
    pivot_returns = data.pivot(index='date', columns='stock_id', values='return')
    pivot_returns = pivot_returns.sort_index()
    
    pivot_mcap = data.pivot(index='date', columns='stock_id', values='mcap')
    pivot_mcap = pivot_mcap.sort_index()
    
    results = []
    dates = sorted(pivot_returns.index.unique())
    
    for i in range(lookback, len(dates)):
        current_date = dates[i]
        past_date = dates[i - lookback]
        
        current_returns = pivot_returns.loc[current_date]
        current_mcap = pivot_mcap.loc[current_date]
        past_returns = pivot_returns.loc[past_date]
        
        month_data = pd.DataFrame({
            'return': current_returns,
            'mcap': current_mcap,
            'past_return': past_returns
        }).dropna()
        
        if len(month_data) < 10:
            results.append({'date': current_date, 'REV': np.nan})
            continue
        
        median_mcap = month_data['mcap'].median()
        month_data['size_group'] = np.where(month_data['mcap'] < median_mcap, 'S', 'B')
        
        month_data['ret_neg'] = -month_data['past_return']
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
        
        results.append({'date': current_date, 'REV': rev_factor})
    
    return pd.DataFrame(results)