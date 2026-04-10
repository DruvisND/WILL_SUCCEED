# -*- coding: utf-8 -*-
"""
反转策略核心模块 (Reversal Strategy Core)
- 不使用未来数据：策略制定与回测仅使用 2025 年及以前的数据
- 形成期、标准化、TopK、再平衡频率 可配置
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Literal

# 数据路径：优先项目根目录下的 week2-data.xlsx
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "week2-data.xlsx")
if not os.path.isfile(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "week2-data.xlsx")
PROCESSED_MONTHLY_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "monthly_returns.csv")

# 禁止使用未来数据：回测与信号生成仅使用该日期之前的数据
CUTOFF_YEAR = 2026


def _make_synthetic_data(obs_per_stock: int = 120, n_stocks: int = 300) -> pd.DataFrame:
    """生成合成面板数据用于测试（仅当缺少 week2-data.xlsx 时）。"""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=obs_per_stock, freq="B")
    codes = [f"S{i:05d}" for i in range(n_stocks)]
    rows = []
    for code in codes:
        ret = np.random.randn(obs_per_stock) * 0.01
        for i, d in enumerate(dates):
            if d.year >= CUTOFF_YEAR:
                continue
            rows.append({"date": d, "code": code, "return": ret[i]})
    df = pd.DataFrame(rows)
    df = df[["date", "code", "return"]]
    return df


def load_week2_data(path: str | None = None, allow_synthetic: bool = True) -> pd.DataFrame:
    """加载 week2-data.xlsx，统一列名并限制在 CUTOFF_YEAR 之前。"""
    # 若已存在预处理后的月度宽表，优先使用（显著加速 Week2 实验）
    if os.path.isfile(PROCESSED_MONTHLY_CSV):
        wide = pd.read_csv(PROCESSED_MONTHLY_CSV, index_col=0, parse_dates=True)
        wide = wide[wide.index.year < CUTOFF_YEAR].copy()
        # 数据单位自检：若收益率像“百分数”被保存（例如 2 表示 2% 或 200%），做一次稳健缩放
        q_hi = wide.quantile(0.99).max()
        q_lo = wide.quantile(0.01).min()
        if (q_hi > 1.0) or (q_lo < -1.0):
            wide = wide / 100.0
        long = (
            wide.stack(dropna=True)
            .rename("return")
            .reset_index()
            .rename(columns={"level_1": "code", "index": "date"})
        )
        long["date"] = pd.to_datetime(long["date"])
        long["code"] = long["code"].astype(str).str.strip()
        return long[["date", "code", "return"]]

    p = path or DATA_PATH
    if not os.path.isfile(p):
        if allow_synthetic:
            import warnings
            warnings.warn(
                f"未找到 {p}，使用合成数据运行（结果仅作结构测试）。"
                "请将 week2-data.xlsx 放在项目根或 data/ 下以使用真实数据。"
            )
            return _make_synthetic_data()
        raise FileNotFoundError(
            f"未找到数据文件: {p}\n请将 week2-data.xlsx 放在项目根目录或 data/ 目录下。"
        )
    raw = pd.read_excel(p)
    # 兼容 TRD_Mnth（月度）字段：Stkcd / Trdmnt / Mretwd
    if {"Stkcd", "Trdmnt", "Mretwd"}.issubset(set(raw.columns)):
        df = raw[["Stkcd", "Trdmnt", "Mretwd"]].copy()
        df["code"] = df["Stkcd"].astype(str).str.strip()
        trd = df["Trdmnt"].astype(str).str.strip()
        trd = trd.str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)  # 兼容 YYYYMM
        df["date"] = pd.to_datetime(trd + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
        df = df.dropna(subset=["date"])
        df["return"] = pd.to_numeric(df["Mretwd"], errors="coerce")
        df = df.dropna(subset=["return"])
        df = df[df["date"].dt.year < CUTOFF_YEAR].copy()  # 禁止使用 2026
        return df[["date", "code", "return"]].copy()

    # 列名标准化：兼容 date/trade_date, code/ts_code/symbol, close/price/return
    col_map = {}
    for c in raw.columns:
        c_lower = str(c).strip().lower()
        if "date" in c_lower or c_lower == "date":
            col_map[c] = "date"
        elif "code" in c_lower or "ts_code" in c_lower or "symbol" in c_lower or "stock" in c_lower:
            col_map[c] = "code"
        elif "close" in c_lower or "price" in c_lower or c_lower == "close":
            col_map[c] = "close"
        elif "return" in c_lower or "ret" in c_lower:
            col_map[c] = "return"
    raw = raw.rename(columns=col_map)
    if "date" not in raw.columns:
        raise ValueError("数据中需包含日期列 (date / trade_date)")
    if "code" not in raw.columns:
        raise ValueError("数据中需包含股票代码列 (code / ts_code / symbol)")
    raw["date"] = pd.to_datetime(raw["date"])
    # 禁止使用未来数据：只保留 CUTOFF_YEAR 之前
    raw = raw[raw["date"].dt.year < CUTOFF_YEAR].copy()
    if "close" in raw.columns and "return" not in raw.columns:
        raw = raw.sort_values(["code", "date"])
        raw["return"] = raw.groupby("code")["close"].pct_change()
        raw = raw.dropna(subset=["return"])
    if "return" not in raw.columns:
        raise ValueError("数据中需包含收益率列 (return) 或价格列 (close) 以计算收益")
    return raw[["date", "code", "return"]].copy()


def to_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """将日频收益转为月频：按 (code, 年月) 聚合为几何收益 (1+r).prod()-1。"""
    df = df.copy()
    # 若已是月末日期且同一 (code,ym) 只有一条记录，则认为已是月频
    df["ym"] = df["date"].dt.to_period("M")
    dup = df.duplicated(subset=["code", "ym"]).any()
    if not dup:
        return df[["date", "code", "return"]].copy()
    monthly = (
        df.groupby(["code", "ym"])["return"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index()
    )
    monthly["date"] = monthly["ym"].dt.to_timestamp(how="end")
    return monthly[["date", "code", "return"]]


def _to_wide(monthly: pd.DataFrame) -> pd.DataFrame:
    """长表(date,code,return) -> 宽表(date index, code columns)。"""
    wide = monthly.pivot_table(index="date", columns="code", values="return", aggfunc="first")
    wide = wide.sort_index()
    return wide


def _formation_signal_wide(returns_wide: pd.DataFrame, formation_months: int) -> pd.DataFrame:
    """
    形成期累计收益（几何累计）：(1+r).rolling(K).prod()-1
    用 log1p + rolling sum 实现，速度比逐股票循环快得多。
    """
    log_r = np.log1p(returns_wide)
    sum_log = log_r.rolling(formation_months).sum()
    form_ret = np.expm1(sum_log)
    return form_ret


def _standardize_wide(signal_wide: pd.DataFrame, method: Literal["raw", "rank", "zscore", "winsor"]) -> pd.DataFrame:
    if method == "raw":
        return signal_wide
    if method == "rank":
        return signal_wide.rank(axis=1, pct=True)
    if method == "zscore":
        mean = signal_wide.mean(axis=1)
        std = signal_wide.std(axis=1).replace(0, np.nan)
        return (signal_wide.sub(mean, axis=0)).div(std, axis=0)
    if method == "winsor":
        lo = signal_wide.quantile(0.01, axis=1)
        hi = signal_wide.quantile(0.99, axis=1)
        return signal_wide.clip(lower=lo, upper=hi, axis=0)
    raise ValueError(f"unknown method: {method}")


def _weights_from_signal(
    score_wide: pd.DataFrame,
    topk: int,
    rebal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    反转：选 score 最小的 topk（输家），等权。
    仅在 rebal_dates 计算权重，其余日期 forward-fill。
    """
    rebal_dates = pd.DatetimeIndex(pd.to_datetime(rebal_dates)).sort_values()
    w = pd.DataFrame(0.0, index=score_wide.index, columns=score_wide.columns)
    for d in rebal_dates:
        if d not in score_wide.index:
            continue
        s = score_wide.loc[d].dropna()
        if s.empty:
            continue
        k = min(topk, len(s))
        losers = s.nsmallest(k).index
        w.loc[d, losers] = 1.0 / k
    # forward fill 持仓（非再平衡月沿用上一期）
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w


def _backtest_wide(weights_wide: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.Series:
    """月度组合收益：sum_i w_{t-1,i} * r_{t,i}（权重滞后一月）。"""
    aligned = weights_wide.shift(1).reindex_like(returns_wide).fillna(0.0)
    port = (aligned * returns_wide).sum(axis=1)
    return port


def formation_return(
    monthly: pd.DataFrame,
    formation_months: int,
) -> pd.DataFrame:
    """
    计算形成期收益（过去 formation_months 个月的累计收益）。
    在每月末仅使用该月及之前的数据，无未来函数。
    """
    monthly = monthly.sort_values(["code", "date"])
    out_list = []
    for code, g in monthly.groupby("code"):
        g = g.set_index("date").sort_index()
        g["cumret"] = (1 + g["return"]).rolling(formation_months).apply(
            lambda x: x.prod() - 1 if len(x) == formation_months else np.nan,
            raw=True,
        )
        out_list.append(g.reset_index()[["date", "code", "cumret"]])
    form = pd.concat(out_list, ignore_index=True)
    form = form.rename(columns={"cumret": "form_ret"})
    return form.dropna(subset=["form_ret"])


def standardize(
    form: pd.DataFrame,
    method: Literal["raw", "rank", "zscore", "winsor"],
) -> pd.DataFrame:
    """对 form_ret 做截面标准化。"""
    form = form.copy()
    if method == "raw":
        form["score"] = form["form_ret"]
        return form
    if method == "rank":
        form["score"] = form.groupby("date")["form_ret"].rank(pct=True)
        return form
    if method == "zscore":
        mean = form.groupby("date")["form_ret"].transform("mean")
        std = form.groupby("date")["form_ret"].transform("std")
        form["score"] = (form["form_ret"] - mean) / std.replace(0, np.nan)
        return form
    if method == "winsor":
        lo = form.groupby("date")["form_ret"].transform(lambda x: x.quantile(0.01))
        hi = form.groupby("date")["form_ret"].transform(lambda x: x.quantile(0.99))
        form["score"] = form["form_ret"].clip(lo, hi)
        return form
    raise ValueError(f"unknown method: {method}")


def select_reversal_topk(
    form: pd.DataFrame,
    topk: int,
    rebal_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    反转策略：选择 score 最小的 topk 只股票（过去表现最差）。
    score 为 form_ret 或标准化后的值，越小越“输家”。
    若提供 rebal_dates，仅在这些日期选股。
    """
    form = form.copy()
    form["rank_asc"] = form.groupby("date")["score"].rank(ascending=True)
    selected = form[form["rank_asc"] <= topk][["date", "code", "rank_asc"]].copy()
    if rebal_dates is not None and len(rebal_dates) > 0:
        rebal_set = set(pd.to_datetime(rebal_dates).normalize())
        selected = selected[selected["date"].dt.normalize().isin(rebal_set)]
    return selected


def rebalance_dates(monthly: pd.DataFrame, freq: Literal["M", "Q"]) -> pd.DatetimeIndex:
    """获取再平衡日（月末或季末）。"""
    dates = monthly["date"].drop_duplicates().sort_values()
    if freq == "M":
        return pd.DatetimeIndex(dates)
    if freq == "Q":
        return pd.DatetimeIndex(dates[dates.dt.month.isin([3, 6, 9, 12])])
    raise ValueError(f"freq must be M or Q, got {freq}")


def backtest(
    monthly: pd.DataFrame,
    selected: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    等权持有 selected 中的股票，在 rebal_dates 再平衡；非再平衡月沿用上一期组合。
    每月收益 = 当月持有组合的等权平均收益。
    """
    monthly = monthly.copy()
    monthly["ym"] = monthly["date"].dt.to_period("M")
    selected = selected.copy()
    rebal_dates = pd.DatetimeIndex(pd.to_datetime(rebal_dates).normalize()).sort_values()
    ret_list = []
    for ym in monthly["ym"].drop_duplicates().sort_values():
        month_end = ym.to_timestamp(how="end")
        # 本月持有的组合 = 最近一次再平衡日（<= 上月末）选出的股票
        prev_month_end = month_end - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
        if prev_month_end < rebal_dates.min():
            ret_list.append({"ym": ym, "ret": np.nan})
            continue
        prev_rebal = rebal_dates[rebal_dates <= prev_month_end]
        if len(prev_rebal) == 0:
            ret_list.append({"ym": ym, "ret": np.nan})
            continue
        last_rebal = prev_rebal.max()
        hold = selected[selected["date"].dt.normalize() == last_rebal][["code"]].drop_duplicates()
        if hold.empty:
            ret_list.append({"ym": ym, "ret": np.nan})
            continue
        next_ret = monthly[monthly["ym"] == ym][["code", "return"]]
        merge = hold.merge(next_ret, on="code", how="inner")
        if merge.empty:
            ret_list.append({"ym": ym, "ret": np.nan})
            continue
        ret_list.append({"ym": ym, "ret": merge["return"].mean()})
    ret_df = pd.DataFrame(ret_list).set_index("ym")
    ret_df.index = ret_df.index.to_timestamp(how="end")
    return ret_df["ret"]


def run_reversal(
    formation_months: int = 6,
    standardization: Literal["raw", "rank", "zscore", "winsor"] = "rank",
    topk: int = 50,
    rebalance_freq: Literal["M", "Q"] = "M",
    data_path: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    运行反转策略，返回 (策略收益序列, 形成期数据用于记录)。
    不使用 2026 年数据。
    """
    raw = load_week2_data(data_path)
    monthly = to_monthly_returns(raw)
    returns_wide = _to_wide(monthly)
    form_ret_wide = _formation_signal_wide(returns_wide, formation_months)
    score_wide = _standardize_wide(form_ret_wide, standardization)
    rebal_dates = rebalance_dates(monthly, rebalance_freq)
    weights_wide = _weights_from_signal(score_wide, topk, rebal_dates)
    strategy_ret = _backtest_wide(weights_wide, returns_wide)
    # 形成期明细在大样本下体量巨大，实验脚本不依赖该输出，这里返回空表以提升运行速度
    return strategy_ret, pd.DataFrame()


def metrics(series: pd.Series, ann: float = 12.0) -> dict:
    """计算年化收益、波动、夏普（无风险利率 0）。"""
    series = series.dropna()
    if series.empty or len(series) < 2:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    n = len(series)
    growth = (1 + series).prod()
    total_ret = growth - 1
    ann_ret = (growth ** (ann / n) - 1) if growth > 0 else np.nan
    ann_vol = series.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "total_ret": total_ret}
