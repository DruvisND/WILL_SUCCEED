"""数据获取模块
从Tushare、AKShare等渠道获取金融数据
"""
from __future__ import annotations
import os
import time
import sqlite3
from typing import Optional, List
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DB_DIR = os.path.join(DATA_DIR, "database")
BACKUP_DIR = os.path.join(DATA_DIR, "backup")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


def init_tushare(token: Optional[str] = None):
    """初始化Tushare Pro API"""
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("请安装tushare: pip install tushare")

    if token is None:
        token = os.environ.get("TUSHARE_TOKEN")
    if token:
        ts.set_token(token)
    return ts.pro_api()


def fetch_index_constituents(
    index_code: str = "000300.SH",
    trade_date: Optional[str] = None,
    api=None
) -> pd.DataFrame:
    """获取指数成分股"""
    if api is None:
        api = init_tushare()

    df = api.index_weight(index_code=index_code, trade_date=trade_date)
    df = df[["ts_code", "weight", "trade_date"]].copy()
    df["weight"] = df["weight"].astype(float)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df


def fetch_daily_price(
    ts_code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    api=None
) -> pd.DataFrame:
    """获取个股日行情"""
    if api is None:
        api = init_tushare()

    df = api.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount,adj_factor"
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    for col in ["open", "high", "low", "close", "vol", "amount", "adj_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("trade_date").reset_index(drop=True)


def fetch_bulk_price(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    save_path: Optional[str] = None,
    delay: float = 0.15
) -> pd.DataFrame:
    """批量获取多只股票日行情"""
    all_data = []

    for i, code in enumerate(ts_codes):
        try:
            df = fetch_daily_price(code, start_date, end_date, adjust)
            if not df.empty:
                all_data.append(df)
            time.sleep(delay)

            if (i + 1) % 50 == 0:
                print(f"已处理 {i+1}/{len(ts_codes)} 只股票...")

        except Exception as e:
            print(f"获取 {code} 数据失败: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    if save_path:
        # 备份原有文件
        backup_file(save_path)
        result.to_csv(save_path, index=False)
        print(f"数据已保存至 {save_path}")

    return result


def fetch_financial_indicators(
    ts_code: str,
    start_date: str,
    end_date: str,
    api=None
) -> pd.DataFrame:
    """获取财务指标"""
    if api is None:
        api = init_tushare()

    df = api.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        return pd.DataFrame()

    select_cols = ["ts_code", "end_date", "market_cap", "pe", "pb", "bm", "roe", "roa"]
    available_cols = [c for c in select_cols if c in df.columns]
    df = df[available_cols].copy()

    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d")
    df = df.rename(columns={"end_date": "report_date", "market_cap": "mcap"})

    return df


def fetch_research_records_from_akshare(symbol: str) -> pd.DataFrame:
    """从AKShare获取机构调研记录"""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请安装akshare: pip install akshare")

    df = ak.stock_irm_cninfo(symbol=symbol)

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [
        "stock_code", "stock_name", "调研日期", "institution",
        "q_type", "q_content", "a_type", "a_content"
    ]

    return df


def backup_file(file_path: str) -> None:
    """备份文件到backup目录
    
    Parameters:
    -----------
    file_path : str
        要备份的文件路径
    """
    if not os.path.isfile(file_path):
        return
    
    filename = os.path.basename(file_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    import shutil
    shutil.copy2(file_path, backup_path)
    print(f"文件已备份至: {backup_path}")


def save_to_database(
    df: pd.DataFrame,
    table_name: str,
    db_path: Optional[str] = None,
    if_exists: str = "append"
) -> None:
    """保存数据到SQLite数据库"""
    if db_path is None:
        db_path = os.path.join(DB_DIR, "research.db")
    
    # 备份数据库文件
    if os.path.isfile(db_path):
        backup_file(db_path)

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"数据已保存至 {db_path} 的 {table_name} 表")


def fetch_all_research_records(
    stock_codes: List[str],
    save_path: Optional[str] = None,
    db_path: Optional[str] = None,
    delay: float = 2.0
) -> pd.DataFrame:
    """批量获取多只股票的调研记录"""
    all_records = []

    for i, code in enumerate(stock_codes):
        try:
            df = fetch_research_records_from_akshare(code)
            if not df.empty:
                all_records.append(df)
            time.sleep(delay)

            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(stock_codes)} 只股票...")

        except Exception as e:
            print(f"获取 {code} 调研记录失败: {e}")
            continue

    if not all_records:
        return pd.DataFrame()

    result = pd.concat(all_records, ignore_index=True)

    if save_path:
        # 备份原有文件
        backup_file(save_path)
        result.to_csv(save_path, index=False, encoding="utf-8-sig")

    if db_path:
        save_to_database(result, "research_records", db_path=db_path, if_exists="replace")

    return result


if __name__ == "__main__":
    api = init_tushare()
    hs300 = fetch_index_constituents("000300.SH", api=api)
    print(f"沪深300成分股数量: {len(hs300)}")