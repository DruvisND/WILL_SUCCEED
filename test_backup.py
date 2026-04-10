"""测试备份功能"""
import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_collector import backup_file, BACKUP_DIR
from src.data_loader import list_backup_files, load_backup_data


print("=" * 60)
print("测试备份功能")
print("=" * 60)

# 创建测试数据
test_data = pd.DataFrame({
    "stock_code": ["000001", "000002", "000003"],
    "stock_name": ["平安银行", "万科A", "国农科技"],
    "price": [15.2, 20.5, 8.7]
})

# 测试文件路径
test_file = os.path.join(PROJECT_ROOT, "data", "test_backup.csv")

# 保存测试数据
test_data.to_csv(test_file, index=False)
print(f"创建测试文件: {test_file}")
print(f"测试数据:\n{test_data}")

# 测试备份功能
print("\n测试备份功能...")
backup_file(test_file)

# 测试列出备份文件
print("\n测试列出备份文件...")
backup_files = list_backup_files("csv")
print(f"备份文件列表: {backup_files}")

if backup_files:
    # 测试加载备份数据
    print("\n测试加载备份数据...")
    latest_backup = backup_files[0]
    print(f"加载最新备份: {latest_backup}")
    backup_data = load_backup_data(latest_backup)
    print(f"备份数据:\n{backup_data}")

# 清理测试文件
if os.path.exists(test_file):
    os.remove(test_file)
    print(f"\n清理测试文件: {test_file}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)