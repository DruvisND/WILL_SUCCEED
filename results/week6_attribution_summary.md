## Week 6 Attribution Analysis: 1-Month Reversal Strategy

### Strategy Overview
- **Period**: 2005-01 ~ 2025-12
- **Formation**: 1 month reversal
- **TopK**: 50 stocks
- **Weighting**: Equal weight
- **Annual Return**: 9.22%
- **Sharpe Ratio**: 0.2642
- **Max Drawdown**: -82.08%

### CH-3 Attribution Results

| Factor | Beta | t-stat |
|--------|------|--------|
| Alpha | -0.8027 | -3.15 |
| MKT | 1.0209 | 33.43 |
| SMB | 0.8041 | 15.15 |
| VMG | -0.3407 | -4.79 |

**R-squared**: 88.05%

### FF-3 Attribution Results

| Factor | Beta | t-stat |
|--------|------|--------|
| Alpha | -0.9321 | -3.63 |
| MKT | 1.0302 | 34.23 |
| SMB | 0.8894 | 19.28 |
| HML | -0.1438 | -2.95 |

**R-squared**: 87.80%

### Model Comparison

| Metric | CH-3 | FF-3 | Interpretation |
|--------|------|------|----------------|
| Alpha | -0.8027% | -0.9321% | |
| t-stat | -3.15 | -3.63 | |
| R-squared | 88.05% | 87.80% | |

### Verifiable Conclusion

Our 1-month reversal strategy achieved 9.22% annual return with Sharpe 0.26. 
CH-3 attribution shows alpha of -0.8027%/month (t=-3.15) is not statistically significant. 
Strategy has SMB beta of 0.80 and VMG beta of -0.34.