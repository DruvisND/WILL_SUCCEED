"""回测框架模块
实现IRSF因子策略的回测
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


class Portfolio:
    """投资组合管理类"""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        leverage: float = 1.0
    ):
        """初始化组合

        Parameters:---------
        initial_capital : float
            初始资金
        leverage : float
            杠杆倍数
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_value = initial_capital

    def rebalance(
        self,
        target_stocks: List[str],
        target_weights: Dict[str, float],
        prices: pd.Series,
        n_long: int = 30,
        n_short: int = 30
    ) -> Tuple[float, Dict]:
        """调仓

        Parameters:---------
        target_stocks : List[str]
            目标股票列表（已排序，高分在前）
        target_weights : Dict
            目标权重
        prices : pd.Series
            股票价格
        n_long : int
            多头持仓数量
        n_short : int
            空头持仓数量

        Returns:
        -------- Tuple : (交易成本, 持仓)
        """
        long_stocks = target_stocks[:n_long]
        short_stocks = target_stocks[-n_short:] if n_short > 0 else []

        self.positions = {}

        long_weight = 0.5 / n_long if n_long > 0 else 0
        short_weight = -0.5 / n_short if n_short > 0 else 0

        transaction_cost = 0.0

        for stock in long_stocks:
            if stock in prices.index:
                self.positions[stock] = {
                    "shares": 0,
                    "weight": long_weight,
                    "direction": 1
                }

        for stock in short_stocks:
            if stock in prices.index:
                self.positions[stock] = {
                    "shares": 0,
                    "weight": short_weight,
                    "direction": -1
                }

        return transaction_cost, self.positions

    def get_positions_value(self, prices: pd.Series) -> float:
        """计算持仓市值"""
        total = 0.0
        for stock, pos in self.positions.items():
            if stock in prices.index:
                total += abs(pos["shares"] * prices[stock])
        return total

    def update_value(self, prices: pd.Series):
        """更新组合净值"""
        position_value = self.get_positions_value(prices)
        self.portfolio_value = self.cash + position_value


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        rebalance_frequency: str = "2W",
        commission_rate: float = 0.0015,
        stamp_tax: float = 0.001,
        slippage_bps: int = 10,
        leverage: float = 1.0
    ):
        """初始化回测引擎

        Parameters:---------
        initial_capital : float
            初始资金
        rebalance_frequency : str
            调仓频率: "1W"(周), "2W"(双周), "1M"(月)
        commission_rate : float
            佣金费率
        stamp_tax : float
            印花税率（卖出）
        slippage_bps : int
            滑点（基点）
        leverage : float
            杠杆
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.slippage = slippage_bps / 10000
        self.leverage = leverage

        self.portfolio = Portfolio(initial_capital, leverage)
        self.trades = []
        self.equity_curve = []

    def calculate_returns(
        self,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame,
        n_long: int = 30,
        n_short: int = 30,
        weight_method: str = "equal"
    ) -> pd.DataFrame:
        """计算策略收益

        Parameters:--------- 
        factor_data : pd.DataFrame
            因子数据，包含 date, ts_code, IRSF_score
        price_data : pd.DataFrame
            价格数据，包含 trade_date, ts_code, close
        n_long : int
            多头数量
        n_short : int
            空头数量
        weight_method : str
            权重方法: "equal", "factor_weighted"

        Returns:--------- 
        pd.DataFrame : 回测结果
        """
        price_data = price_data.copy()
        price_data["trade_date"] = pd.to_datetime(price_data["trade_date"])
        price_data = price_data.set_index(["trade_date", "ts_code"])['close'].unstack()

        factor_data = factor_data.copy()
        factor_data["date"] = pd.to_datetime(factor_data["date"])

        # 时间对齐检查
        min_factor_date = factor_data["date"].min()
        max_factor_date = factor_data["date"].max()
        min_price_date = price_data.index.min()
        max_price_date = price_data.index.max()
        
        print(f"因子数据时间范围: {min_factor_date} 到 {max_factor_date}")
        print(f"价格数据时间范围: {min_price_date} 到 {max_price_date}")
        
        # 检查因子数据是否早于价格数据
        if min_factor_date >= min_price_date:
            print("警告: 因子数据起始日期晚于或等于价格数据起始日期")
        
        # 检查因子数据是否覆盖价格数据
        if max_factor_date >= max_price_date:
            print("警告: 因子数据结束日期晚于或等于价格数据结束日期")

        if self.rebalance_frequency == "1W":
            factor_data["rebalance_date"] = factor_data["date"] + pd.to_timedelta(7, unit="d")
        elif self.rebalance_frequency == "2W":
            factor_data["rebalance_date"] = factor_data["date"] + pd.to_timedelta(14, unit="d")
        elif self.rebalance_frequency == "1M":
            factor_data["rebalance_date"] = factor_data["date"] + pd.DateOffset(months=1)
        else:
            factor_data["rebalance_date"] = factor_data["date"] + pd.DateOffset(months=1)
        
        # 检查调仓日期是否在价格数据范围内
        min_rebalance_date = factor_data["rebalance_date"].min()
        max_rebalance_date = factor_data["rebalance_date"].max()
        
        print(f"调仓日期范围: {min_rebalance_date} 到 {max_rebalance_date}")
        
        if min_rebalance_date < min_price_date:
            print("警告: 调仓起始日期早于价格数据起始日期")
        if max_rebalance_date > max_price_date:
            print("警告: 调仓结束日期晚于价格数据结束日期")

        results = []
        rebalance_dates = sorted(factor_data["rebalance_date"].unique())

        for i, rebal_date in enumerate(tqdm(rebalance_dates[:-1], desc="回测")):
            current_factor = factor_data[factor_data["rebalance_date"] == rebal_date].copy()

            if current_factor.empty:
                continue

            current_factor = current_factor.sort_values("IRSF_score", ascending=False)

            long_stocks = current_factor.head(n_long)["ts_code"].tolist()
            short_stocks = current_factor.tail(n_short)["ts_code"].tolist() if n_short > 0 else []

            next_rebal_date = rebalance_dates[i + 1]

            period_prices = price_data.loc[rebal_date:next_rebal_date]

            if period_prices.empty:
                continue

            long_returns = []
            short_returns = []

            for stock in long_stocks:
                if stock in period_prices.columns:
                    prices = period_prices[stock].dropna()
                    if len(prices) > 1:
                        ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                        long_returns.append(ret)

            for stock in short_stocks:
                if stock in period_prices.columns:
                    prices = period_prices[stock].dropna()
                    if len(prices) > 1:
                        ret = 1 - (prices.iloc[-1] / prices.iloc[0])
                        short_returns.append(ret)

            long_ret = np.mean(long_returns) if long_returns else 0
            short_ret = np.mean(short_returns) if short_returns else 0

            strategy_ret = (long_ret + short_ret) / 2

            trading_cost = (self.commission_rate + self.slippage) * 2
            net_ret = strategy_ret - trading_cost

            results.append({
                "date": rebal_date,
                "long_return": long_ret,
                "short_return": short_ret,
                "gross_return": strategy_ret,
                "net_return": net_ret,
                "n_long": len(long_stocks),
                "n_short": len(short_stocks)
            })

        return pd.DataFrame(results)

    def run_backtest(
        self,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame,
        n_long: int = 30,
        n_short: int = 30,
        benchmark_code: Optional[str] = "000905.SH"
    ) -> Dict:
        """运行完整回测

        Parameters:---------
        factor_data : pd.DataFrame
            因子数据
        price_data : pd.DataFrame
            价格数据
        n_long : int
            多头数量
        n_short : int
            空头数量
        benchmark_code : str
            基准代码

        Returns:---------
        Dict : 回测绩效指标
        """
        print("开始回测...")

        returns_df = self.calculate_returns(
            factor_data, price_data, n_long, n_short
        )

        if returns_df.empty:
            return {}

        cumulative = (1 + returns_df["net_return"]).cumprod()
        equity_curve = self.initial_capital * cumulative

        returns_df["cumulative_return"] = cumulative
        returns_df["equity"] = equity_curve

        if benchmark_code and benchmark_code in price_data["ts_code"].values:
            benchmark_returns = self._calculate_benchmark_returns(
                price_data, benchmark_code, returns_df["date"].values
            )
        else:
            benchmark_returns = returns_df["net_return"]

        excess_returns = returns_df["net_return"] - benchmark_returns

        stats = self._calculate_statistics(returns_df, benchmark_returns, excess_returns)

        stats["equity_curve"] = equity_curve.tolist()
        stats["dates"] = returns_df["date"].dt.strftime("%Y-%m-%d").tolist()

        self.equity_curve = returns_df

        return stats

    def _calculate_benchmark_returns(
        self,
        price_data: pd.DataFrame,
        benchmark_code: str,
        dates: np.ndarray
    ) -> pd.Series:
        """计算基准收益"""
        bench_data = price_data[price_data["ts_code"] == benchmark_code].copy()
        bench_data["trade_date"] = pd.to_datetime(bench_data["trade_date"])
        bench_data = bench_data.sort_values("trade_date")
        bench_data["return"] = bench_data["close"].pct_change()

        return bench_data.set_index("trade_date")["return"]

    def _calculate_statistics(
        self,
        returns_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        excess_returns: pd.Series
    ) -> Dict:
        """计算绩效指标"""
        net_returns = returns_df["net_return"]

        annual_return = net_returns.mean() * 12
        annual_vol = net_returns.std() * np.sqrt(12)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cumulative = (1 + net_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        win_rate = (net_returns > 0).sum() / len(net_returns)
        avg_win = net_returns[net_returns > 0].mean() if (net_returns > 0).any() else 0
        avg_loss = net_returns[net_returns < 0].mean() if (net_returns < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_return": cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0,
            "n_periods": len(returns_df)
        }

    def plot_equity_curve(
        self,
        stats: Dict,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> None:
        """绘制权益曲线"""
        if not self.equity_curve.empty:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            dates = pd.to_datetime(stats["dates"])
            equity = stats["equity_curve"]

            axes[0].plot(dates, equity, label="策略", linewidth=2)
            if benchmark_returns is not None:
                bench_cumulative = (1 + benchmark_returns).cumprod() * self.initial_capital
                axes[0].plot(dates[:len(bench_cumulative)], bench_cumulative, label="基准", linewidth=1.5)

            axes[0].set_title("策略权益曲线", fontsize=14)
            axes[0].set_xlabel("日期")
            axes[0].set_ylabel("净值")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            net_returns = self.equity_curve["net_return"]
            axes[1].bar(range(len(net_returns)), net_returns * 100)
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].set_title("月度收益", fontsize=14)
            axes[1].set_xlabel("期数")
            axes[1].set_ylabel("收益率(%)")
            axes[1].grid(True, alpha=0.3)

            plt.tight_plt()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"图表已保存至 {save_path}")

            plt.show()


class BacktestRunner:
    """回测运行器封装类"""

    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """初始化回测运行器

        Parameters:---------
        config : Dict
            回测配置
        """
        self.config = config or {}
        self.engine = BacktestEngine(
            initial_capital=self.config.get("initial_capital", 1000000),
            rebalance_frequency=self.config.get("rebalance_frequency", "2W"),
            commission_rate=self.config.get("commission_rate", 0.0015),
            stamp_tax=self.config.get("stamp_tax", 0.001),
            slippage_bps=self.config.get("slippage_bps", 10)
        )

    def run(
        self,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame,
        n_long: int = 30,
        n_short: int = 30
    ) -> Dict:
        """运行回测"""
        return self.engine.run_backtest(
            factor_data, price_data, n_long, n_short
        )

    def run_grid_search(
        self,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame,
        n_long_range: List[int] = [20, 30, 50],
        n_short_range: List[int] = [20, 30, 50]
    ) -> pd.DataFrame:
        """参数网格搜索"""
        results = []

        for n_long in n_long_range:
            for n_short in n_short_range:
                print(f"\n测试: n_long={n_long}, n_short={n_short}")

                stats = self.run(factor_data, price_data, n_long, n_short)

                results.append({
                    "n_long": n_long,
                    "n_short": n_short,
                    "annual_return": stats.get("annual_return", 0),
                    "sharpe_ratio": stats.get("sharpe_ratio", 0),
                    "max_drawdown": stats.get("max_drawdown", 0)
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("sharpe_ratio", ascending=False)

        print("\n最佳参数:")
        print(results_df.head())

        return results_df


if __name__ == "__main__":
    print("回测框架测试")

    price_data = pd.DataFrame({
        "ts_code": sum([["00" + str(i).zfill(4) + ".SZ"] * 100 for i in range(1, 51)], [])
    }).assign(
        trade_date=pd.date_range("2020-01-01", periods=100),
        close=np.random.uniform(10, 50, (50, 100)).mean(axis=1)
    })

    factor_data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=50),
        "ts_code": np.repeat(["00" + str(i).zfill(4) + ".SZ" for i in range(1, 51)], 50),
        "IRSF_score": np.random.randn(2500)
    })

    runner = BacktestRunner()
    print("回测初始化完成")