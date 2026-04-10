"""
数据加载模块
加载各类金融数据用于IRSF因子策略
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DB_DIR = os.path.join(DATA_DIR, "database")
BACKUP_DIR = os.path.join(DATA_DIR, "backup")


def load_research_records(
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    加载机构调研记录数据

    Parameters:
    -----------
    stock_code : str, optional
        股票代码筛选，如 "000001" 或 "000001.SZ"
    start_date : str, optional
        起始日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
    end_date : str, optional
        结束日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
    db_path : str, optional
        SQLite数据库路径，默认 data/database/research.db

    Returns:
    --------
    pd.DataFrame
        包含列: stock_code, stock_name,调研日期, institution, q_content, a_content
    """
    if db_path is None:
        db_path = os.path.join(DB_DIR, "research.db")

    if not os.path.isfile(db_path):
        csv_path = os.path.join(RAW_DIR, "research_records.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["调研日期"])
        else:
            raise FileNotFoundError(f"未找到调研数据: {db_path} 或 {csv_path}")
    else:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM research_records WHERE 1=1"
        params = []

        if stock_code:
            query += " AND stock_code = ?"
            params.append(stock_code)
        if start_date:
            query += " AND 调研日期 >= ?"
            params.append(start_date)
        if end_date:
            query += " AND 调研日期 <= ?"
            params.append(end_date)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

    return df


def load_price_data(
    stock_codes: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq"
) -> pd.DataFrame:
    """
    加载日行情数据

    Parameters:
    -----------
    stock_codes : list, optional
        股票代码列表，如 ["000001.SZ", "600000.SH"]
    start_date : str, optional
        起始日期，格式 "YYYYMMDD"
    end_date : str, optional
        结束日期，格式 "YYYYMMDD"
    adjust : str
        复权类型: "qfq"前复权，"hfq"后复权，"None"不复权

    Returns:
    --------
    pd.DataFrame
        包含列: ts_code, trade_date, open, high, low, close, vol, amount, adj_factor
    """
    csv_path = os.path.join(PROCESSED_DIR, "daily_price.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"未找到行情数据: {csv_path}\n请先运行数据收集脚本"
        )

    df = pd.read_csv(csv_path, parse_dates=["trade_date"])

    if stock_codes:
        df = df[df["ts_code"].isin(stock_codes)]
    if start_date:
        df = df[df["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["trade_date"] <= pd.to_datetime(end_date)]

    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def load_index_constituents(
    index_code: str = "000300.SH"
) -> pd.DataFrame:
    """
    加载指数成分股

    Parameters:
    -----------
    index_code : str
        指数代码，默认沪深300 "000300.SH"，中证500为 "000905.SH"

    Returns:
    --------
    pd.DataFrame
        包含列: ts_code, weight, trade_date
    """
    csv_path = os.path.join(PROCESSED_DIR, f"index_{index_code.replace('.', '_')}_constituents.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到成分股数据: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["trade_date"])
    return df


def load_financial_fundamentals(
    stock_codes: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    加载财务基本面数据

    Parameters:
    -----------
    stock_codes : list, optional
        股票代码列表
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含列: ts_code, report_date, mcap, pe, pb, bm, roe, roa
    """
    csv_path = os.path.join(PROCESSED_DIR, "financial_fundamentals.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到财务数据: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["report_date"])

    if stock_codes:
        df = df[df["ts_code"].isin(stock_codes)]
    if start_date:
        df = df[df["report_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["report_date"] <= pd.to_datetime(end_date)]

    return df


def load_risk_free_rate(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    加载无风险利率（月度）

    Parameters:
    -----------
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含列: date, rf (月度无风险利率)
    """
    csv_path = os.path.join(PROCESSED_DIR, "risk_free_rate.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到无风险利率数据: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df


def get_universe(
    include_hs300: bool = True,
    include_zz500: bool = True
) -> list:
    """
    获取选股池（沪深300 + 中证500）

    Parameters:
    -----------
    include_hs300 : bool
        是否包含沪深300
    include_zz500 : bool
        是否包含中证500

    Returns:
    --------
    list : 股票代码列表
    """
    stocks = set()

    if include_hs300:
        hs300 = load_index_constituents("000300.SH")
        stocks.update(hs300["ts_code"].tolist())

    if include_zz500:
        zz500 = load_index_constituents("000905.SH")
        stocks.update(zz500["ts_code"].tolist())

    return sorted(list(stocks))


def list_backup_files(file_type: str = "all") -> list:
    """
    列出备份文件

    Parameters:
    -----------
    file_type : str
        文件类型: "all", "csv", "db"

    Returns:
    --------
    list : 备份文件列表
    """
    if not os.path.exists(BACKUP_DIR):
        return []
    
    backup_files = []
    for filename in os.listdir(BACKUP_DIR):
        if file_type == "csv" and filename.endswith(".csv"):
            backup_files.append(filename)
        elif file_type == "db" and filename.endswith(".db"):
            backup_files.append(filename)
        elif file_type == "all":
            backup_files.append(filename)
    
    # 按时间戳排序，最新的在前
    backup_files.sort(reverse=True)
    return backup_files


def load_backup_data(backup_filename: str) -> pd.DataFrame:
    """
    加载备份数据

    Parameters:
    -----------
    backup_filename : str
        备份文件名

    Returns:
    --------
    pd.DataFrame : 加载的数据
    """
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"备份文件不存在: {backup_path}")
    
    if backup_filename.endswith(".csv"):
        return pd.read_csv(backup_path)
    elif backup_filename.endswith(".db"):
        import sqlite3
        conn = sqlite3.connect(backup_path)
        df = pd.read_sql_query("SELECT * FROM research_records", conn)
        conn.close()
        return df
    else:
        raise ValueError(f"不支持的文件类型: {backup_filename}")


def merge_factor_data(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    control_factors: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    合并因子数据与收益数据、控制变量

    Parameters:
    -----------
    factor_df : pd.DataFrame
        因子数据，包含 ts_code, date, factor columns
    price_df : pd.DataFrame
        行情数据，包含 ts_code, trade_date, close
    control_factors : dict, optional
        控制变量列名映射，如 {"size": "mcap", "bm": "bm"}

    Returns:
    --------
    pd.DataFrame
        合并后的数据，包含因子、收益、控制变量
    """
    price_df = price_df.copy()
    price_df["date"] = price_df["trade_date"]

    price_df["return"] = price_df.groupby("ts_code")["close"].pct_change()

    next_month = price_df.copy()
    next_month["date"] = next_month["date"] + pd.DateOffset(months=1)
    next_month["return_next"] = next_month.groupby("ts_code")["return"].shift(-1)

    merged = pd.merge(factor_df, next_month, on=["ts_code", "date"], how="inner")

    if control_factors:
        fund_df = load_financial_fundamentals()
        fund_df["date"] = fund_df["report_date"]
        merged = pd.merge(merged, fund_df, on=["ts_code", "date"], how="left")

    return merged.dropna()