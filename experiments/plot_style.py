# -*- coding: utf-8 -*-
"""
统一 Matplotlib 绘图风格（解决中文方块问题）。

在 Windows 上优先使用：
- Microsoft YaHei（微软雅黑）
- SimHei（黑体）

注意：需要系统已安装对应字体；若都没有，会回退到 DejaVu Sans（可能缺中文字形）。
"""
from __future__ import annotations

from matplotlib import pyplot as plt


def apply_cn_font() -> None:
    # 常见中文字体优先级（按可用性自动回退）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

