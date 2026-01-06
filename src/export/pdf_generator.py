"""Generate PDF reports from Markdown."""

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils.logger import setup_logger

logger = setup_logger("PDFGenerator")


def markdown_to_pdf(
    markdown_text: str, title: Optional[str] = None, metadata: Optional[dict] = None
) -> bytes:
    """Convert Markdown to PDF using ReportLab.

    Args:
        markdown_text: Markdown content
        title: Optional report title
        metadata: Optional metadata

    Returns:
        PDF as bytes
    """
    logger.info("Generating PDF with ReportLab")

    buffer = io.BytesIO()

    # Create PDF
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    # Build content
    story = []
    styles = getSampleStyleSheet()

    # Add custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#667eea"),
        spaceAfter=30,
        alignment=1,  # Center
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#667eea"),
        spaceAfter=12,
        spaceBefore=20,
    )

    # Title
    if not title:
        title = "TaskFlow Sentiment Analysis Report"

    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Metadata table
    if metadata:
        data = [
            ["Quality Score", f"{metadata.get('quality_score', 'N/A')}/100"],
            ["Word Count", f"{metadata.get('word_count', 'N/A'):,}"],
            ["Processing Time", f"{metadata.get('total_time', 'N/A')}s"],
            ["Cost", f"${metadata.get('cost', 'N/A')}"],
            ["Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
        ]

        t = Table(data, colWidths=[2 * inch, 3 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4ff")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#667eea")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.white),
                ]
            )
        )

        story.append(t)
        story.append(Spacer(1, 0.5 * inch))

    # Convert Markdown to simple text (strip formatting)
    # Simple parser - split by lines
    lines = markdown_text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            story.append(Spacer(1, 0.1 * inch))
            continue

        # Headers
        if line.startswith("# "):
            text = line[2:].strip()
            story.append(Paragraph(text, title_style))
        elif line.startswith("## "):
            text = line[3:].strip()
            story.append(Paragraph(text, heading_style))
        elif line.startswith("### "):
            text = line[4:].strip()
            sub_style = ParagraphStyle(
                "SubHeading",
                parent=styles["Heading3"],
                fontSize=14,
                textColor=colors.HexColor("#764ba2"),
                spaceAfter=10,
                spaceBefore=15,
            )
            story.append(Paragraph(text, sub_style))
        # Bold
        elif line.startswith("**") and line.endswith("**"):
            text = line[2:-2]
            story.append(Paragraph(f"<b>{text}</b>", styles["Normal"]))
        # Regular paragraph
        else:
            # Clean markdown syntax
            text = line.replace("**", "<b>").replace("**", "</b>")
            text = text.replace("*", "<i>").replace("*", "</i>")
            story.append(Paragraph(text, styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

    # Build PDF
    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"PDF generated: {len(pdf_bytes)} bytes")
    return pdf_bytes
