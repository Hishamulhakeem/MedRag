import io
import json
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from backend.database import get_db
from backend.models import Report, User, AbnormalValue
from backend.schemas import AbnormalValueResponse
from backend.middleware.auth import get_current_user
from backend.services.summarizer import SummarizerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])
summarizer_service = SummarizerService()

@router.post("/{report_id}/summarize")
async def summarize_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Streams the summary for a specific medical report and persists it to the database
    once the streaming finishes.
    """
    # 1. Fetch report
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )

    # 2. Get abnormalities list
    abnormalities = db.query(AbnormalValue).filter(
        AbnormalValue.report_id == report_id
    ).all()
    
    abnormalities_data = [
        {
            "test_name": a.test_name,
            "value": a.value,
            "unit": a.unit,
            "normal_range": a.normal_range,
            "status": a.status,
            "explanation": f"Value detected outside normal limits."  # default fallback
        }
        for a in abnormalities
    ]

    async def sse_summary_generator():
        full_summary_text = ""
        try:
            async for data_str in summarizer_service.generate_summary_stream(
                extracted_text=report.extracted_text or "",
                abnormalities=abnormalities_data
            ):
                if data_str.startswith("data: "):
                    content_json = data_str[6:].strip()
                    try:
                        parsed = json.loads(content_json)
                        if "token" in parsed:
                            full_summary_text += parsed["token"]
                    except json.JSONDecodeError:
                        pass
                yield data_str

            # Save summary to DB when streaming finishes successfully
            from backend.database import SessionLocal
            db_session = SessionLocal()
            try:
                report_record = db_session.query(Report).filter(Report.id == report_id).first()
                if report_record:
                    report_record.summary = full_summary_text
                    db_session.commit()
            except Exception as db_ex:
                logger.error(f"Failed to persist summary to report DB: {str(db_ex)}")
            finally:
                db_session.close()

        except Exception as e:
            logger.error(f"Error in summary stream: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(sse_summary_generator(), media_type="text/event-stream")

@router.post("/{report_id}/abnormalities", response_model=List[AbnormalValueResponse])
def get_abnormalities(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns a list of flagged abnormalities for a report."""
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )

    # Re-run normal range explanations mapping
    from backend.services.abnormality import REFERENCE_RANGES
    
    abnormalities = db.query(AbnormalValue).filter(
        AbnormalValue.report_id == report_id
    ).all()

    res = []
    for a in abnormalities:
        # Retrieve base explanation from reference mapping
        ref = REFERENCE_RANGES.get(a.test_name, {})
        explanation = f"Your {a.test_name} is {a.status.lower()} ({a.value} {a.unit or ''}). normal range is {a.normal_range}."
        if ref:
            explanation += f" {ref.get('description', '')}"

        res.append(AbnormalValueResponse(
            test_name=a.test_name,
            value=a.value,
            unit=a.unit or "",
            normal_range=a.normal_range,
            status=a.status,
            explanation=explanation
        ))

    return res

def convert_md_to_html(md_text: str) -> str:
    """Basic helper to translate Markdown styles to ReportLab HTML tags."""
    if not md_text:
        return ""
    # Convert bold **text** to <b>text</b>
    html = re_sub_bold = re_sub_bold = re_sub_bold = re_sub_bold = re_sub_bold = re_sub_bold = re_sub_bold = md_text
    import re
    html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html)
    html = re.sub(r"\*(.*?)\*", r"<i>\1</i>", html)
    # Convert list markers
    html = re.sub(r"^\s*-\s+(.*?)$", r"&bull; \1", html, flags=re.MULTILINE)
    # Convert headers
    html = re.sub(r"^###\s+(.*?)$", r"<font size=11><b>\1</b></font>", html, flags=re.MULTILINE)
    html = re.sub(r"^##\s+(.*?)$", r"<font size=12><b>\1</b></font>", html, flags=re.MULTILINE)
    html = re.sub(r"^#\s+(.*?)$", r"<font size=14><b>\1</b></font>", html, flags=re.MULTILINE)
    # Convert blockquotes
    html = re.sub(r"^>\s+(.*?)$", r"<i>\1</i>", html, flags=re.MULTILINE)
    # Line breaks
    html = html.replace("\n", "<br/>")
    return html

@router.get("/{report_id}/export")
def export_summary_pdf(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generates a professional patient-ready PDF summary for download."""
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )

    # Use existing summary, or build a fast mock one if empty
    summary_text = report.summary
    if not summary_text:
        summary_text = (
            "### Report Summary\n\n"
            "This report is waiting for a summary generation. Please run the AI summary first.\n\n"
            "**Primary Findings**:\n"
        )
        abns = db.query(AbnormalValue).filter(AbnormalValue.report_id == report_id).all()
        for ab in abns:
            summary_text += f"- {ab.test_name}: {ab.value} {ab.unit or ''} [{ab.status}]\n"
        summary_text += "\nConsult a doctor for diagnostic assessments."

    # Build PDF
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#0f172a")  # Slate-900
    teal_color = colors.HexColor("#0d9488")     # Teal-600
    light_bg = colors.HexColor("#f8fafc")       # Slate-50
    border_color = colors.HexColor("#cbd5e1")   # Slate-300

    # Modify style objects
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=primary_color,
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=teal_color,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # 1. Title
    story.append(Paragraph("MedRAG — AI Medical Report Summary", title_style))
    story.append(Spacer(1, 10))

    # 2. Metadata Info
    metadata_data = [
        [
            Paragraph("<b>File Name:</b>", body_style), Paragraph(report.filename, body_style),
            Paragraph("<b>Analyzed On:</b>", body_style), Paragraph(report.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"), body_style)
        ],
        [
            Paragraph("<b>Mime Type:</b>", body_style), Paragraph(report.mime_type, body_style),
            Paragraph("<b>Page Count:</b>", body_style), Paragraph(str(report.page_count), body_style)
        ]
    ]
    meta_table = Table(metadata_data, colWidths=[80, 200, 80, 172])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('PADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, border_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # 3. Abnormalities Section
    story.append(Paragraph("Biomarker Findings Table", h2_style))
    abnormals = db.query(AbnormalValue).filter(AbnormalValue.report_id == report_id).all()
    
    if abnormals:
        # Table columns: Test Name | Detected Value | Unit | Reference Range | Status
        table_data = [[
            Paragraph("Test Name", table_header_style),
            Paragraph("Value", table_header_style),
            Paragraph("Unit", table_header_style),
            Paragraph("Reference Range", table_header_style),
            Paragraph("Status", table_header_style)
        ]]

        for ab in abnormals:
            # Color code status
            status_color = "#22c55e" # normal green
            if ab.status == "CRITICAL":
                status_color = "#ef4444" # red
            elif ab.status in ["HIGH", "LOW"]:
                status_color = "#f97316" # orange
            elif ab.status == "BORDERLINE":
                status_color = "#eab308" # yellow

            status_cell = Paragraph(f"<font color='{status_color}'><b>{ab.status}</b></font>", table_cell_style)
            
            table_data.append([
                Paragraph(ab.test_name, table_cell_style),
                Paragraph(str(ab.value), table_cell_style),
                Paragraph(ab.unit or "-", table_cell_style),
                Paragraph(ab.normal_range, table_cell_style),
                status_cell
            ])

        ab_table = Table(table_data, colWidths=[140, 70, 70, 152, 100])
        ab_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), teal_color),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(ab_table)
    else:
        story.append(Paragraph("No specific biomarkers or abnormalities were parsed numerically in the report table.", body_style))
    
    story.append(Spacer(1, 15))

    # 4. Detailed Narrative Summary
    story.append(Paragraph("AI Narrative Clinical Summary", h2_style))
    formatted_summary = convert_md_to_html(summary_text)
    story.append(Paragraph(formatted_summary, body_style))
    story.append(Spacer(1, 20))

    # 5. Disclaimer Box
    disclaimer_box = Table([[
        Paragraph(
            "<b>CLINICAL DISCLAIMER:</b> This summary is compiled using an AI analysis engine based on the text contents of the uploaded report. "
            "It does not replace clinical validation, professional medical diagnostics, or a doctor's consultation. "
            "Please check with a primary physician before making any changes to your medication or health treatment plan.",
            disclaimer_style
        )
    ]], colWidths=[532])
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef2f2")),  # soft red bg
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fca5a5")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(disclaimer_box)

    # Build document
    doc.build(story)
    
    pdf_buffer.seek(0)
    
    return StreamingResponse(
        pdf_buffer, 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_summary_{report_id}.pdf"}
    )
