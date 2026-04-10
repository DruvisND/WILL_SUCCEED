# Week03：成本、换手率与证据标准（总结）

来源：`https://quant-suibe.netlify.app/week03`

---

## 1. 为什么要从 Gross 到 Net

Week01–02 计算的是 **Gross Return（总收益）**：假设无交易成本。\n\n现实交易会产生成本（买卖价差、冲击、佣金、印花税等），高换手策略可能在扣除成本后失效。\n\n因此 Week03 要求：\n- 计算换手率\n- 用简化成本模型得到净收益\n- 做成本敏感性与可交易性评估\n- 用 Strategy Card v1 统一记录“证据”\n\n---\n\n## 2. 换手率 Turnover\n\n定义：\n\n\\[\nTurnover_t = \\frac{1}{2} \\sum_{i=1}^N |w_{i,t} - w_{i,t-1}|\n\\]\n\n关键点：\n- 乘 1/2：买+卖合计变化除以 2 才是实际交易量比例\n- 年化换手率：月度换手率均值 × 12\n\n---\n\n## 3. 简化成本模型\n\n净收益：\n\n\\[\nr^{net}_t = r^{gross}_t - Turnover_t \\times c\n\\]\n\n其中：\n- `c = cost_bps / 10000`\n- 常用测试成本：0、10、20、50 bps\n\n---\n\n## 4. 成本敏感性分析（0/10/20/50 bps）\n\n建议输出表：\n- Gross 指标（年化收益/夏普/回撤/换手）\n- Net 指标（同上）\n- 成本拖累：Gross - Net\n\n可视化建议（课程 Week03 交付物）：\n- 不同成本下净值曲线对比\n- 成本 vs 收益/夏普\n- 成本拖累柱状图\n- 可交易性总结面板\n\n---\n\n## 5. 可交易性评估清单\n\n- **临界成本（Break-even Cost）**：净收益降到 0% 或 10% 时的成本水平\n- **换手率**：\n  - >1000% 年化：高换手（成本敏感）\n  - 500–1000%：中等\n  - <500%：较低\n- **容量（Capacity）**：与流动性、TopK、持仓市值分布相关\n- **现实约束**：T+1、涨跌停、停牌、ST、低流动性过滤等\n\n---\n\n## 6. 证据标准：Strategy Card v1\n\nWeek03 的“证据”要求用标准化模板记录：\n- 信号、组合、对齐验证、Gross/Net 指标、成本敏感性、可交易性、复现命令\n\n模板文件：`reports/strategy_card_v1.md`\n\n*** End Patch"}Oops ApplyPatch freeform; need correct. We'll call ApplyPatch properly.
