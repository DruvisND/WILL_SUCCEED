"""策略评估模块
计算绩效指标和风险指标
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import numpy as np
import os
import json


def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12
) -> Dict: """计算策略绩效指标

Parameters:---------
returns : pd.Series
    策略收益率序列
risk_free_rate : float
    年化无风险利率
periods_per_year : int
    年化期数（月度为12，周度为52）

Returns:--------
Dict : 绩效指标
"""
    if len(returns) == 0:
        return {}

    returns = returns.dropna()

    mean_return = returns.mean()
    std_return = returns.std()

    annual_return = mean_return * periods_per_year
    annual_vol = std_return * np.sqrt(periods_per_year)

    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    win_rate = (returns > 0).sum() / len(returns)
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0

    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        "n_periods": len(returns)
    }


def calculate_risk_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict: """计算风险指标

Parameters:---------
returns : pd.Series
    策略收益率
benchmark_returns : pd.Series, optional
    基准收益率

Returns:--------
Dict : 风险指标
"""
    if len(returns) == 0:
        return {}

    returns = returns.dropna()

    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
    sortino = returns.mean() * 12 / downside_std if downside_std > 0 else 0

    result = {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "sortino_ratio": sortino
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        aligned_returns = returns.align(benchmark_returns, join="inner")
        strategy_ret = aligned_returns[0]
        bench_ret = aligned_returns[1]

        beta = np.cov(strategy_ret, bench_ret)[0, 1] / np.var(bench_ret) if np.var(bench_ret) > 0 else 1

        alpha = strategy_ret.mean() - beta * bench_ret.mean()
        alpha_annual = alpha * 12

        tracking_error = (strategy_ret - bench_ret).std() * np.sqrt(12)
        info_ratio = (strategy_ret.mean() - bench_ret.mean()) * 12 / tracking_error if tracking_error > 0 else 0

        result.update({
            "beta": beta,
            "alpha_annual": alpha_annual,
            "tracking_error": tracking_error,
            "info_ratio": info_ratio
        })

    return result


def generate_performance_report(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_name: str = "策略",
    save_path: Optional[str] = None
) -> str: """生成绩效报告

Parameters:---------
returns : pd.Series
    策略收益率
benchmark_returns : pd.Series, optional
    基准收益率
strategy_name : str
    策略名称
save_path : str, optional
    保存路径

Returns:--------
str : 报告文本
"""
    perf = calculate_performance_metrics(returns)
    risk = calculate_risk_metrics(returns, benchmark_returns)

    report = []
    report.append(f"{'='*50}")
    report.append(f"{strategy_name} 绩效报告")
    report.append(f"{'='*50}")

    report.append("\n【收益指标】")
    report.append(f"  总收益率: {perf['total_return']:.2%}")
    report.append(f"  年化收益率: {perf['annual_return']:.2%}")
    report.append(f"  年化波动率: {perf['annual_vol']:.2%}")
    report.append(f"  夏普比率: {perf['sharpe_ratio']:.3f}")

    report.append("\n【风险指标】")
    report.append(f"  最大回撤: {perf['max_drawdown']:.2%}")
    report.append(f"  Calmar比率: {perf['calmar_ratio']:.3f}")
    report.append(f"  VaR(95%): {risk.get('var_95', 0):.4f}")
    report.append(f"  CVaR(95%): {risk.get('cvar_95', 0):.4f}")
    report.append(f"  偏度: {risk.get('skewness', 0):.4f}")
    report.append(f"  峰度: {risk.get('kurtosis', 0):.4f}")
    report.append(f"  Sortino比率: {risk.get('sortino_ratio', 0):.3f}")

    report.append("\n【交易统计】")
    report.append(f"  胜率: {perf['win_rate']:.2%}")
    report.append(f"  平均盈利: {perf['avg_win']:.4f}")
    report.append(f"  平均亏损: {perf['avg_loss']:.4f}")
    report.append(f"  盈亏比: {perf['profit_loss_ratio']:.3f}")

    if benchmark_returns is not None:
        report.append("\n【相对基准】")
        report.append(f"  Beta: {risk.get('beta', 1):.4f}")
        report.append(f"  年化Alpha: {risk.get('alpha_annual', 0):.2%}")
        report.append(f"  跟踪误差: {risk.get('tracking_error', 0):.4f}")
        report.append(f"  信息比率: {risk.get('info_ratio', 0):.3f}")

    report.append(f"\n{'='*50}")
    report.append(f"数据期数: {perf['n_periods']} 个月")
    report.append(f"{'='*50}")

    report_text = "\n".join(report)
    print(report_text)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n报告已保存至 {save_path}")

    return report_text


def save_results(
    stats: Dict,
    equity_curve: list,
    dates: list,
    config: Dict,
    save_dir: str
) -> None: """保存回测结果

Parameters:---------
stats : Dict
    绩效统计
equity_curve : list
    权益曲线
dates : list
    日期
config : Dict
    配置
save_dir : str
    保存目录
"""
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "config": config,
        "performance": stats,
        "equity_curve": {
            "dates": dates,
            "values": equity_curve
        }
    }

    json_path = os.path.join(save_dir, "backtest_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存至 {json_path}")


class PerformanceEvaluator: """策略绩效评估器"""

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "策略"
    ): """初始化评估器

    Parameters:---------
    returns : pd.Series
        策略收益率
    benchmark_returns : pd.Series, optional
        基准收益率
    strategy_name : str
        策略名称
    """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.strategy_name = strategy_name
        self.metrics = {}

    def evaluate(self) -> Dict: """执行评估"""
        self.metrics["performance"] = calculate_performance_metrics(self.returns)
        self.metrics["risk"] = calculate_risk_metrics(
            self.returns, self.benchmark_returns
        )
        return self.metrics

    def report(self, save_path: Optional[str] = None) -> str: """生成报告"""
        return generate_performance_report(
            self.returns,
            self.benchmark_returns,
            self.strategy_name,
            save_path
        )


if __name__ == "__main__":
    print("评估模块测试")
    returns = pd.Series(np.random.randn(100) * 0.03 + 0.01)
    evaluator = PerformanceEvaluator(returns, strategy_name="测试策略")
    metrics = evaluator.evaluate()
    print(f"夏普比率: {metrics['performance']['sharpe_ratio']:.3f}")