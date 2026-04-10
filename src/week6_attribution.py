"""
Week 6: Factor Attribution Analysis
Runs reversal strategy and attributes returns to CH-3 and FF-3 factors
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 60)
print("Week 6: Factor Attribution Analysis")
print("=" * 60)

def load_monthly_data():
    """Load and prepare monthly data for reversal strategy"""
    print("\n1. Loading data...")

    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "monthly_data.csv")

    df = df[['date', 'stock_id', 'return']].copy()
    df = df.drop_duplicates(subset=['date', 'stock_id'], keep='first')

    returns_df = df.pivot(index='date', columns='stock_id', values='return')
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df = returns_df.sort_index()

    returns_df = returns_df[returns_df.index >= '2005-01']
    returns_df = returns_df[returns_df.index <= '2025-12']

    returns_df[returns_df > 0.50] = np.nan
    returns_df[returns_df < -0.50] = np.nan

    returns_df = returns_df.dropna(axis=1, how='all')
    returns_df = returns_df.dropna(axis=0, how='all')

    print(f"   Returns: {returns_df.shape}")
    return returns_df

def make_signal_reversal(returns_df, k=1):
    """Generate reversal signal: negative of past k-month cumulative return"""
    rolling_sum = returns_df.shift(1).rolling(window=k).sum()
    signal_df = -rolling_sum
    return signal_df

def build_topk_portfolio(signal_df, topk=50):
    """Build equal-weighted TopK portfolio"""
    weights_df = signal_df.copy() * 0.0
    for idx in range(len(signal_df)):
        row = signal_df.iloc[idx]
        valid_signals = row.dropna()
        if len(valid_signals) == 0:
            continue
        topk_stocks = valid_signals.nlargest(topk).index.tolist()
        weight = 1.0 / topk
        for stk in topk_stocks:
            weights_df.iloc[idx, weights_df.columns.get_loc(stk)] = weight
    return weights_df

def backtest_gross(weights_df, returns_df):
    """Calculate gross portfolio returns"""
    aligned_weights = weights_df.shift(1)
    aligned_returns = returns_df.loc[aligned_weights.index]
    portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()
    return portfolio_returns

def calculate_metrics(returns_series):
    """Calculate performance metrics"""
    metrics = {}
    mean_monthly = returns_series.mean()
    std_monthly = returns_series.std()
    metrics['annual_return'] = (1 + mean_monthly) ** 12 - 1
    metrics['sharpe_ratio'] = mean_monthly / std_monthly * np.sqrt(12) if std_monthly > 0 else np.nan
    nav = (1 + returns_series).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    return metrics

def load_factors():
    """Load CH-3 and FF-3 factors"""
    ch3 = pd.read_csv(PROJECT_ROOT / "data" / "factors" / "ch3_monthly.csv")
    ff3 = pd.read_csv(PROJECT_ROOT / "data" / "factors" / "ff3_monthly.csv")
    ch3['date'] = pd.to_datetime(ch3['date'])
    ff3['date'] = pd.to_datetime(ff3['date'])
    return ch3, ff3

def run_attribution_regression(strategy_returns, factors_df, rf_df, model_name):
    """Run factor attribution regression using numpy"""
    try:
        strategy_returns = strategy_returns.copy()
        strategy_returns.index = pd.to_datetime(strategy_returns.index)

        factors_df = factors_df.copy()
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        factors_df = factors_df.set_index('date')

        rf_df = rf_df.copy()
        rf_df['date'] = pd.to_datetime(rf_df['date'])
        rf_df = rf_df.set_index('date')

        data = strategy_returns.to_frame(name='strategy')
        data = data.join(factors_df)
        data = data.join(rf_df.rename(columns={'rf': 'rf'}))

        data['excess_return'] = data['strategy'] - data['rf']

        data = data.dropna()

        if len(data) < 10:
            print(f"   Warning: Only {len(data)} observations after dropping NaN")
            return None, None

        y = data['excess_return'].values

        if model_name == 'CH-3':
            X = data[['MKT', 'SMB', 'VMG']].values
        else:
            X = data[['MKT', 'SMB', 'HML']].values

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print(f"   Warning: NaN values in X or y")
            return None, None

        X_const = np.column_stack([np.ones(len(y)), X])

        beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
        resid = y - X_const @ beta
        n = len(y)
        k = X_const.shape[1]
        sigma_sq = np.sum(resid**2) / (n - k)

        try:
            var_beta = sigma_sq * np.linalg.pinv(X_const.T @ X_const)
        except:
            var_beta = sigma_sq * np.linalg.inv(X_const.T @ X_const + 1e-8 * np.eye(k))

        se = np.sqrt(np.abs(np.diag(var_beta)))
        t_stats = beta / se

        r_squared = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

        results = {
            'alpha': beta[0],
            'alpha_t': t_stats[0],
            'mkt_beta': beta[1],
            'mkt_t': t_stats[1],
            'smb_beta': beta[2],
            'smb_t': t_stats[2],
            'r_squared': r_squared,
            'n_obs': n
        }

        if model_name == 'CH-3':
            results['vmg_beta'] = beta[3]
            results['vmg_t'] = t_stats[3]
        else:
            results['hml_beta'] = beta[3]
            results['hml_t'] = t_stats[3]

        return results, None

    except Exception as e:
        print(f"   Error in {model_name} regression: {e}")
        import traceback
        traceback.print_exc()
        return None, None

print("\n2. Running reversal strategy (k=1 month)...")
returns_df = load_monthly_data()

signal_df = make_signal_reversal(returns_df, k=1)
weights_df = build_topk_portfolio(signal_df, topk=50)
strategy_returns = backtest_gross(weights_df, returns_df)

metrics = calculate_metrics(strategy_returns)
print(f"   Strategy returns: {len(strategy_returns)} months")
print(f"   Annual return: {metrics['annual_return']*100:.2f}%")
print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
print(f"   Max drawdown: {metrics['max_drawdown']*100:.2f}%")

print("\n3. Loading factors...")
ch3_factors, ff3_factors = load_factors()
rf_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "rf_monthly.csv")
rf_df.columns = ['date', 'rf']
rf_df['date'] = pd.to_datetime(rf_df['date'])

print("\n4. Running CH-3 attribution...")
ch3_results, ch3_model = run_attribution_regression(
    strategy_returns, ch3_factors, rf_df, 'CH-3'
)

if ch3_results:
    print(f"   Alpha: {ch3_results['alpha']*100:.4f}%/month (t={ch3_results['alpha_t']:.2f})")
    print(f"   MKT beta: {ch3_results['mkt_beta']:.4f} (t={ch3_results['mkt_t']:.2f})")
    print(f"   SMB beta: {ch3_results['smb_beta']:.4f} (t={ch3_results['smb_t']:.2f})")
    print(f"   VMG beta: {ch3_results['vmg_beta']:.4f} (t={ch3_results['vmg_t']:.2f})")
    print(f"   R-squared: {ch3_results['r_squared']*100:.2f}%")

print("\n5. Running FF-3 attribution...")
ff3_results, ff3_model = run_attribution_regression(
    strategy_returns, ff3_factors, rf_df, 'FF-3'
)

if ff3_results:
    print(f"   Alpha: {ff3_results['alpha']*100:.4f}%/month (t={ff3_results['alpha_t']:.2f})")
    print(f"   MKT beta: {ff3_results['mkt_beta']:.4f} (t={ff3_results['mkt_t']:.2f})")
    print(f"   SMB beta: {ff3_results['smb_beta']:.4f} (t={ff3_results['smb_t']:.2f})")
    print(f"   HML beta: {ff3_results['hml_beta']:.4f} (t={ff3_results['hml_t']:.2f})")
    print(f"   R-squared: {ff3_results['r_squared']*100:.2f}%")

if ch3_results is None or ff3_results is None:
    print("   Error: Regression failed")
    print("=" * 60)
    sys.exit(1)

print("\n" + "=" * 60)
print("Attribution Summary")
print("=" * 60)

summary_data = []
summary_data.append("## Week 6 Attribution Analysis: 1-Month Reversal Strategy")
summary_data.append("")
summary_data.append("### Strategy Overview")
summary_data.append(f"- **Period**: 2005-01 ~ 2025-12")
summary_data.append(f"- **Formation**: 1 month reversal")
summary_data.append(f"- **TopK**: 50 stocks")
summary_data.append(f"- **Weighting**: Equal weight")
summary_data.append(f"- **Annual Return**: {metrics['annual_return']*100:.2f}%")
summary_data.append(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.4f}")
summary_data.append(f"- **Max Drawdown**: {metrics['max_drawdown']*100:.2f}%")
summary_data.append("")
summary_data.append("### CH-3 Attribution Results")
summary_data.append("")
summary_data.append("| Factor | Beta | t-stat |")
summary_data.append("|--------|------|--------|")
summary_data.append(f"| Alpha | {ch3_results['alpha']*100:.4f} | {ch3_results['alpha_t']:.2f} |")
summary_data.append(f"| MKT | {ch3_results['mkt_beta']:.4f} | {ch3_results['mkt_t']:.2f} |")
summary_data.append(f"| SMB | {ch3_results['smb_beta']:.4f} | {ch3_results['smb_t']:.2f} |")
summary_data.append(f"| VMG | {ch3_results['vmg_beta']:.4f} | {ch3_results['vmg_t']:.2f} |")
summary_data.append("")
summary_data.append(f"**R-squared**: {ch3_results['r_squared']*100:.2f}%")
summary_data.append("")
summary_data.append("### FF-3 Attribution Results")
summary_data.append("")
summary_data.append("| Factor | Beta | t-stat |")
summary_data.append("|--------|------|--------|")
summary_data.append(f"| Alpha | {ff3_results['alpha']*100:.4f} | {ff3_results['alpha_t']:.2f} |")
summary_data.append(f"| MKT | {ff3_results['mkt_beta']:.4f} | {ff3_results['mkt_t']:.2f} |")
summary_data.append(f"| SMB | {ff3_results['smb_beta']:.4f} | {ff3_results['smb_t']:.2f} |")
summary_data.append(f"| HML | {ff3_results['hml_beta']:.4f} | {ff3_results['hml_t']:.2f} |")
summary_data.append("")
summary_data.append(f"**R-squared**: {ff3_results['r_squared']*100:.2f}%")
summary_data.append("")
summary_data.append("### Model Comparison")
summary_data.append("")
summary_data.append("| Metric | CH-3 | FF-3 | Interpretation |")
summary_data.append("|--------|------|------|----------------|")
summary_data.append(f"| Alpha | {ch3_results['alpha']*100:.4f}% | {ff3_results['alpha']*100:.4f}% | |")
summary_data.append(f"| t-stat | {ch3_results['alpha_t']:.2f} | {ff3_results['alpha_t']:.2f} | |")
summary_data.append(f"| R-squared | {ch3_results['r_squared']*100:.2f}% | {ff3_results['r_squared']*100:.2f}% | |")
summary_data.append("")
summary_data.append("### Verifiable Conclusion")
summary_data.append("")
if ch3_results['alpha_t'] > 2:
    summary_data.append(f"Our 1-month reversal strategy achieved {metrics['annual_return']*100:.2f}% annual return with Sharpe {metrics['sharpe_ratio']:.2f}. ")
    summary_data.append(f"CH-3 attribution shows significant alpha of {ch3_results['alpha']*100:.4f}%/month (t={ch3_results['alpha_t']:.2f}). ")
    summary_data.append("The strategy has genuine alpha not explained by CH-3 factors.")
else:
    summary_data.append(f"Our 1-month reversal strategy achieved {metrics['annual_return']*100:.2f}% annual return with Sharpe {metrics['sharpe_ratio']:.2f}. ")
    summary_data.append(f"CH-3 attribution shows alpha of {ch3_results['alpha']*100:.4f}%/month (t={ch3_results['alpha_t']:.2f}) is not statistically significant. ")
    summary_data.append(f"Strategy has SMB beta of {ch3_results['smb_beta']:.2f} and VMG beta of {ch3_results['vmg_beta']:.2f}.")

summary_text = "\n".join(summary_data)
print(summary_text)

output_path = PROJECT_ROOT / "results" / "week6_attribution_summary.md"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"\nSummary saved to: {output_path}")

if ch3_model is not None:
    ch3_output = PROJECT_ROOT / "results" / "week6_ch3_regression.txt"
    with open(ch3_output, 'w', encoding='utf-8') as f:
        f.write(str(ch3_model.summary()))
    print(f"CH-3 regression details: {ch3_output}")

if ff3_model is not None:
    ff3_output = PROJECT_ROOT / "results" / "week6_ff3_regression.txt"
    with open(ff3_output, 'w', encoding='utf-8') as f:
        f.write(str(ff3_model.summary()))
    print(f"FF-3 regression details: {ff3_output}")

print("\n" + "=" * 60)
print("Week 6 Attribution Complete!")
print("=" * 60)
