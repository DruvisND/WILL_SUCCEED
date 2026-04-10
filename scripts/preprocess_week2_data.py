# -*- coding: utf-8 -*-
"""
任务 1：数据获取与预处理
从 week2-data.xlsx（TRD_Mnth 月个股回报率）生成 data/processed/monthly_returns.csv
宽格式：行=日期，列=股票代码，值=收益率
字段说明见 TRD_Mnth[DES][xlsx].txt：Stkcd, Trdmnt, Mretwd
"""
from __future__ import annotations

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "monthly_returns.csv")

# 数据文件：优先项目根目录
EXCEL_PATHS = [
    os.path.join(PROJECT_ROOT, "week2-data.xlsx"),
    os.path.join(DATA_DIR, "week2-data.xlsx"),
]

# TRD_Mnth 字段：英文与常见中文列名映射
COL_MAP = {
    "Stkcd": "Stkcd",
    "证券代码": "Stkcd",
    "Trdmnt": "Trdmnt",
    "交易月份": "Trdmnt",
    "Mretwd": "Mretwd",
    "考虑现金红利再投资的月个股回报率": "Mretwd",
}


def find_excel() -> str:
    for p in EXCEL_PATHS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"未找到 week2-data.xlsx。请将 TRD_Mnth 数据保存为 week2-data.xlsx，"
        f"放在项目根目录或 data/ 目录下。\n尝试路径: {EXCEL_PATHS}"
    )


def load_raw(excel_path: str) -> pd.DataFrame:
    """读取 Excel，统一为 Stkcd, Trdmnt, Mretwd。若首行非表头，可改为 header=1 等。"""
    raw = pd.read_excel(excel_path, header=0)
    # 统一列名
    rename = {}
    for c in raw.columns:
        c_str = str(c).strip()
        if c_str in COL_MAP:
            rename[c] = COL_MAP[c_str]
    raw = raw.rename(columns=rename)
    for req in ["Stkcd", "Trdmnt", "Mretwd"]:
        if req not in raw.columns:
            raise ValueError(f"数据需包含列 {req}（或其中文名）。当前列: {list(raw.columns)}")
    return raw[["Stkcd", "Trdmnt", "Mretwd"]].copy()


def to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    长格式 -> 宽格式
    行 = 月末日期，列 = 股票代码，值 = Mretwd
    """
    df = df.copy()
    df["Stkcd"] = df["Stkcd"].astype(str).str.strip()
    # Trdmnt 为 YYYY-MM，转为月末日期
    df["date"] = pd.to_datetime(df["Trdmnt"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
    wide = df.pivot_table(index="date", columns="Stkcd", values="Mretwd")
    wide.sort_index(inplace=True)
    return wide


def main():
    excel_path = find_excel()
    print(f"读取: {excel_path}")
    raw = load_raw(excel_path)
    # 数值清洗
    raw["Mretwd"] = pd.to_numeric(raw["Mretwd"], errors="coerce")
    raw = raw.dropna(subset=["Mretwd"])
    # 时间范围 2005-2025（课程要求）
    trd = raw["Trdmnt"].astype(str).str.strip()
    # 支持 YYYY-MM 或 YYYYMM
    trd = trd.str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)
    raw["date_tmp"] = pd.to_datetime(trd + "-01", errors="coerce")
    raw = raw.dropna(subset=["date_tmp"])
    raw = raw[(raw["date_tmp"].dt.year >= 2005) & (raw["date_tmp"].dt.year <= 2025)]
    raw = raw.drop(columns=["date_tmp"])
    wide = to_wide_format(raw)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    wide.to_csv(OUT_PATH)
    print(f"已保存: {OUT_PATH}")
    print(f"数据维度: {wide.shape}")
    print(f"时间范围: {wide.index.min()} 到 {wide.index.max()}")
    print(f"股票数量: {wide.shape[1]}")
    # 检查清单
    r_min, r_max = wide.min().min(), wide.max().max()
    print(f"收益率范围: [{r_min:.4f}, {r_max:.4f}]（应在约 -0.5 ~ 0.5）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
