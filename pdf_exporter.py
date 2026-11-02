"""
PDF Report Generator for Smart Review Analyzer v3.0
Uses ReportLab to generate professional, paginated PDF reports of batch sentiment analysis.
"""

import io
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw 'Page X of Y' footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header rule and title (pages after page 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, "Smart Review Analyzer v3.0 — Batch Sentiment Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 750, letter[0] - 36, 750)

        # Footer rule and page info
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 40, letter[0] - 36, 40)

        self.drawString(36, 28, "Confidential & Automated Sentiment Intelligence")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 36, 28, page_str)

        self.restoreState()


def generate_batch_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generates a polished PDF report from batch analysis output.
    Returns PDF file bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#312e81"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        'CellRegular',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    cell_style_bold = ParagraphStyle(
        'CellBold',
        parent=cell_style,
        fontName='Helvetica-Bold'
    )

    review_text_style = ParagraphStyle(
        'ReviewText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Header & Title Banner
    story.append(Paragraph("Smart Review Analyzer v3.0", title_style))
    story.append(Paragraph("Batch Sentiment Analysis & Multi-Format Document Intelligence Report", subtitle_style))
    story.append(Spacer(1, 4))

    # 2. Metadata Box
    diagnostics = data.get("diagnostics", {})
    filename = diagnostics.get("filename") or "Pasted Batch Reviews"
    file_ext = diagnostics.get("file_extension") or "N/A"
    total_reviews = data.get("total_reviews", 0)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    meta_data = [
        [
            Paragraph("<b>Source:</b> " + str(filename), cell_style),
            Paragraph("<b>Format:</b> " + str(file_ext).upper(), cell_style),
            Paragraph("<b>Generated:</b> " + now_str, cell_style)
        ],
        [
            Paragraph("<b>Total Reviews:</b> " + str(total_reviews), cell_style),
            Paragraph("<b>Characters:</b> " + str(diagnostics.get("character_count", "N/A")), cell_style),
            Paragraph("<b>Words:</b> " + str(diagnostics.get("word_count", "N/A")), cell_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[200, 140, 200])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Sentiment Scorecard Table
    story.append(Paragraph("Executive Sentiment Scorecard", section_heading))
    dist = data.get("sentiment_distribution", {})
    pos_cnt = dist.get("Positive", 0)
    neg_cnt = dist.get("Negative", 0)
    neu_cnt = dist.get("Neutral", 0)

    pos_pct = dist.get("Positive_Percentage", round((pos_cnt / total_reviews * 100), 1) if total_reviews else 0.0)
    neg_pct = dist.get("Negative_Percentage", round((neg_cnt / total_reviews * 100), 1) if total_reviews else 0.0)
    neu_pct = dist.get("Neutral_Percentage", round((neu_cnt / total_reviews * 100), 1) if total_reviews else 0.0)

    scorecard_data = [
        ["Total Processed", "Positive Reviews", "Negative Reviews", "Neutral Reviews"],
        [
            f"{total_reviews}",
            f"{pos_cnt} ({pos_pct}%)",
            f"{neg_cnt} ({neg_pct}%)",
            f"{neu_cnt} ({neu_pct}%)"
        ]
    ]

    scorecard_table = Table(scorecard_data, colWidths=[135, 135, 135, 135])
    scorecard_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#e0e7ff")),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#dcfce7")),
        ('BACKGROUND', (2, 0), (2, 0), colors.HexColor("#fee2e2")),
        ('BACKGROUND', (3, 0), (3, 0), colors.HexColor("#fef3c7")),

        ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor("#3730a3")),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor("#166534")),
        ('TEXTCOLOR', (2, 0), (2, 0), colors.HexColor("#991b1b")),
        ('TEXTCOLOR', (3, 0), (3, 0), colors.HexColor("#b45309")),

        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 12),
        ('TEXTCOLOR', (0, 1), (0, 1), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor("#15803d")),
        ('TEXTCOLOR', (2, 1), (2, 1), colors.HexColor("#b91c1c")),
        ('TEXTCOLOR', (3, 1), (3, 1), colors.HexColor("#b45309")),

        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(scorecard_table)
    story.append(Spacer(1, 12))

    # 4. Aspect Distribution
    aspects = data.get("aspect_distribution", {})
    if aspects:
        story.append(Paragraph("Aspect Category Breakdown", section_heading))
        aspect_items = sorted(aspects.items(), key=lambda x: x[1], reverse=True)
        aspect_cols = []
        for cat, cnt in aspect_items[:8]:
            aspect_cols.append(f"<b>{cat}:</b> {cnt} ({round(cnt/total_reviews*100, 1) if total_reviews else 0}%)")
        
        # Format into a 3 or 4-column sub-table
        rows = [aspect_cols[i:i+4] for i in range(0, len(aspect_cols), 4)]
        if rows:
            formatted_rows = []
            for r in rows:
                while len(r) < 4:
                    r.append("")
                formatted_rows.append([Paragraph(c, cell_style) for c in r])
            asp_table = Table(formatted_rows, colWidths=[135, 135, 135, 135])
            asp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(asp_table)
            story.append(Spacer(1, 12))

    # 5. Detailed Reviews Table
    story.append(Paragraph("Detailed Review Sentiment Classifications", section_heading))

    reviews = data.get("detailed_results", [])
    
    # Table headers
    table_headers = [
        Paragraph("<b># / ID</b>", cell_style_bold),
        Paragraph("<b>Review Text Snippet</b>", cell_style_bold),
        Paragraph("<b>Sentiment</b>", cell_style_bold),
        Paragraph("<b>ML Conf</b>", cell_style_bold),
        Paragraph("<b>VADER</b>", cell_style_bold),
        Paragraph("<b>TextBlob</b>", cell_style_bold),
        Paragraph("<b>ML (LR)</b>", cell_style_bold),
        Paragraph("<b>Aspects</b>", cell_style_bold)
    ]
    
    table_rows = [table_headers]

    for idx, r in enumerate(reviews):
        source_id = r.get("source_id") or str(idx + 1)
        rev_text = r.get("original_review", "")
        # Truncate very long review text for PDF readability
        if len(rev_text) > 220:
            rev_text = rev_text[:217] + "..."
        
        sent = r.get("sentiment", "Neutral")
        conf_val = r.get("confidence_percentage")
        conf = f"{conf_val}%" if conf_val is not None else "N/A"
        vader_comp = r.get("overall_compound_score") if r.get("overall_compound_score") is not None else r.get("vader_scores", {}).get("compound_score", "0.0")
        tb_pol = r.get("textblob_metrics", {}).get("polarity", "0.0")
        ml_pred = r.get("ml_model_predictions", {}).get("logistic_regression", "N/A")
        cats = ", ".join(r.get("categories", [])) or "General"

        # Sentiment badge styling
        sent_color = "#15803d" if sent == "Positive" else ("#b91c1c" if sent == "Negative" else "#b45309")
        sent_p = Paragraph(f"<font color='{sent_color}'><b>{sent}</b></font>", cell_style)

        table_rows.append([
            Paragraph(str(source_id), cell_style),
            Paragraph(rev_text, review_text_style),
            sent_p,
            Paragraph(conf, cell_style),
            Paragraph(str(vader_comp), cell_style),
            Paragraph(str(tb_pol), cell_style),
            Paragraph(str(ml_pred), cell_style),
            Paragraph(cats, cell_style)
        ])

    # Widths sum to 540 (letter width is 612 - 72 = 540)
    col_widths = [45, 215, 52, 48, 44, 44, 46, 46]
    results_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    
    t_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]

    # Alternate row background
    for row_i in range(1, len(table_rows)):
        if row_i % 2 == 0:
            t_style.append(('BACKGROUND', (0, row_i), (-1, row_i), colors.HexColor("#f8fafc")))

    results_table.setStyle(TableStyle(t_style))
    story.append(results_table)

    # Build document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()

