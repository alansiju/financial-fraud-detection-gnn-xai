"""api/reports.py - PDF, CSV, and Excel export endpoints"""
import io
import csv
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from server.auth.dependencies import get_current_user
from server.ml import loader, predictor

router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.get("/csv")
def download_csv(
    limit: int = Query(1000, ge=1, le=50000),
    fraud_only: bool = Query(False),
    current_user=Depends(get_current_user),
):
    """Export predictions as CSV."""
    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probs  = predictor.get_all_predictions()
    labels = data.y.cpu().numpy()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "node_id", "transaction_id", "fraud_probability", "is_fraud", "true_label", "risk_level"
    ])
    writer.writeheader()

    count = 0
    for i in range(len(probs)):
        if count >= limit:
            break
        prob    = float(probs[i])
        is_frd  = prob >= loader.get_threshold()
        if fraud_only and not is_frd:
            continue
        real_tx = loader.get_tx_id_by_node(i)
        writer.writerow({
            "node_id":           i,
            "transaction_id":    real_tx if real_tx is not None else i,
            "fraud_probability": round(prob, 6),
            "is_fraud":          is_frd,
            "true_label":        int(labels[i]),
            "risk_level":        predictor._risk_level(prob, loader.get_threshold()),
        })
        count += 1

    output.seek(0)
    filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/excel")
def download_excel(limit: int = Query(500, le=2000), current_user=Depends(get_current_user)):
    """Export predictions as styled Excel sheet."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl not installed. Run: pip install openpyxl")

    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probs  = predictor.get_all_predictions()
    labels = data.y.cpu().numpy()

    wb  = openpyxl.Workbook()
    ws  = wb.active
    ws.title = "Fraud Predictions"

    # Header styling
    headers = ["Node ID", "Transaction ID", "Fraud Probability", "Is Fraud", "True Label", "Risk Level"]
    header_fill = PatternFill("solid", fgColor="1E293B")
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font      = Font(bold=True, color="FFFFFF")
        cell.fill      = header_fill
        cell.alignment = Alignment(horizontal="center")

    red_fill   = PatternFill("solid", fgColor="FEE2E2")
    green_fill = PatternFill("solid", fgColor="DCFCE7")

    for i in range(min(limit, len(probs))):
        prob   = float(probs[i])
        is_frd = prob >= loader.get_threshold()
        fill   = red_fill if is_frd else green_fill
        real_tx = loader.get_tx_id_by_node(i)
        tx_val = real_tx if real_tx is not None else i
        row_data = [i, tx_val, round(prob, 6), is_frd, int(labels[i]), predictor._risk_level(prob, loader.get_threshold())]
        for col, val in enumerate(row_data, 1):
            cell = ws.cell(row=i + 2, column=col, value=val)
            cell.fill = fill

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/pdf")
def download_pdf(current_user=Depends(get_current_user)):
    """Generate a summary PDF report."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=40)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Heading1"], alignment=TA_CENTER,
                                 fontSize=20, textColor=colors.HexColor("#1E40AF"))
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER,
                                 fontSize=11, textColor=colors.HexColor("#64748B"))
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#1E293B"))

    story = []
    story.append(Paragraph("FraudShield AI — Detection Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", sub_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Model Overview", h2_style))
    model_data = [
        ["Parameter", "Value"],
        ["Model Architecture", "GraphSAGE (3-Layer)"],
        ["Dataset", "IEEE-CIS Fraud Detection"],
        ["Validation ROC-AUC", "0.8967"],
        ["Test Accuracy", "97.4%"],
        ["Precision", "82.0%"],
        ["Recall", "71.0%"],
        ["F1 Score", "76.1%"],
    ]
    t = Table(model_data, colWidths=[3 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Summary stats
    story.append(Paragraph("Dataset Statistics", h2_style))
    data = loader.get_data()
    if data is not None:
        labels = data.y.cpu().numpy()
        total  = len(labels)
        fraud  = int(labels.sum())
        stats  = [
            ["Metric", "Count"],
            ["Total Transactions", f"{total:,}"],
            ["Fraud Transactions",  f"{fraud:,}"],
            ["Legitimate",          f"{total - fraud:,}"],
            ["Fraud Rate",          f"{fraud/total*100:.2f}%"],
        ]
    else:
        stats = [["Metric", "Count"], ["Total", "590,540"], ["Fraud", "20,663"], ["Legit", "569,877"]]
    t2 = Table(stats, colWidths=[3 * inch, 3.5 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E40AF")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EFF6FF"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)

    doc.build(story)
    buf.seek(0)
    filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
