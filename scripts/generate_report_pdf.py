# -*- coding: utf-8 -*-
"""
Generate a submission-ready PDF report from markdown-like content.

Output:
  reports/reversal_strategy_report.pdf
"""
from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = PROJECT_ROOT / "reports" / "reversal_strategy_report.md"
OUT_PDF = PROJECT_ROOT / "reports" / "reversal_strategy_report.pdf"
RESULTS_DIR = PROJECT_ROOT / "results"


def _read_md_text() -> str:
    return REPORT_MD.read_text(encoding="utf-8")


def _img(path: Path, width_cm: float, max_height_cm: float = 22.0) -> Image | None:
    if not path.exists():
        return None
    img = Image(str(path))
    # scale by width first
    img.drawWidth = width_cm * cm
    img.drawHeight = img.imageHeight * (img.drawWidth / img.imageWidth)
    # cap height to avoid LayoutError
    max_h = max_height_cm * cm
    if img.drawHeight > max_h:
        scale = max_h / img.drawHeight
        img.drawHeight = max_h
        img.drawWidth = img.drawWidth * scale
    return img


def build_pdf():
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"missing: {REPORT_MD}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1C", parent=styles["Heading1"], spaceAfter=10))
    styles.add(ParagraphStyle(name="H2C", parent=styles["Heading2"], spaceAfter=8))
    styles.add(ParagraphStyle(name="BodyC", parent=styles["BodyText"], leading=14, spaceAfter=6))

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    story: list = []

    md = _read_md_text().splitlines()
    for line in md:
        s = line.strip()
        if not s:
            story.append(Spacer(1, 6))
            continue
        if s.startswith("# "):
            story.append(Paragraph(s[2:], styles["H1C"]))
            continue
        if s.startswith("## "):
            story.append(Paragraph(s[3:], styles["H2C"]))
            continue
        if s.startswith("```"):
            # skip code fences; keep report concise
            continue
        if s.startswith("> "):
            story.append(Paragraph(s[2:], styles["Italic"]))
            continue
        # simple bullet rendering
        if s.startswith("- "):
            story.append(Paragraph(f"• {s[2:]}", styles["BodyC"]))
            continue
        story.append(Paragraph(s, styles["BodyC"]))

    # Append figures (if present)
    figs = [
        ("基线净值曲线", RESULTS_DIR / "week1_nav.png"),
        ("Week2 形成期对比", RESULTS_DIR / "week2_formation_period.png"),
        ("Week2 标准化方法对比", RESULTS_DIR / "week2_standardization.png"),
        ("Week2 TopK 对比", RESULTS_DIR / "week2_topk.png"),
        ("Week2 再平衡频率对比", RESULTS_DIR / "week2_rebalance.png"),
        ("Week3 成本敏感性", RESULTS_DIR / "week3_cost_sensitivity.png"),
    ]
    story.append(PageBreak())
    story.append(Paragraph("附录：关键图表", styles["H1C"]))
    for title, p in figs:
        story.append(Paragraph(title, styles["H2C"]))
        im = _img(p, width_cm=16.5, max_height_cm=21.5)
        if im is None:
            story.append(Paragraph(f"(缺失图表文件：{p.name})", styles["BodyC"]))
        else:
            story.append(im)
        story.append(Spacer(1, 10))

    doc.build(story)


def main():
    build_pdf()
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()

