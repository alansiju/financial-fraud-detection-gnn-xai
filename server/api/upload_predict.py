"""
api/upload_predict.py — Upload a real transaction CSV and get fraud predictions.

Endpoints:
    POST /api/upload-predict          Full pipeline: upload → preprocess → graph → predict
    POST /api/upload-predict/validate Validate-only: check columns and data quality
    GET  /api/upload-predict/sample   Download a sample IEEE-CIS CSV file
"""

import io
import csv
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from server.auth.dependencies import get_current_user
from server.ml import loader
from server.ml.pipeline import run_full_pipeline, validate_csv, MAX_UPLOAD_ROWS
from server.config import DEFAULT_FRAUD_THRESHOLD, ROOT
from typing import Optional
import pandas as pd

router = APIRouter(prefix="/api/upload-predict", tags=["Upload & Predict"])


@router.post("")
async def upload_and_predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.01, le=0.99),
    current_user=Depends(get_current_user),
):
    """
    Upload a transaction CSV (IEEE-CIS format) and run the full
    preprocessing → graph construction → GraphSAGE inference pipeline.

    The CSV should contain transaction-level columns like TransactionAmt,
    card1, ProductCD, DeviceInfo, etc. Missing columns are filled with
    default values automatically.

    Returns fraud predictions for every row in the uploaded file.
    """
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if loader.get_preprocessor() is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded — cannot process transactions")

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    print("[SCAN API] Request received")
    print(f"[SCAN API] Filename: {file.filename}")
    print(f"[SCAN API] Threshold: {threshold}")

    # Read and parse CSV
    try:
        content = await file.read()
        print("[SCAN API] File saved")
        text = content.decode("utf-8-sig")  # utf-8-sig handles BOM
        df = pd.read_csv(io.StringIO(text), low_memory=False)
        print("[SCAN API] CSV parsed")
        print(f"[SCAN API] Rows: {len(df)}")
    except UnicodeDecodeError as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="File encoding error — please use UTF-8")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV is empty")
    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(df):,} rows — maximum is {MAX_UPLOAD_ROWS:,}"
        )

    # Run the full pipeline in a thread executor so it doesn't block the
    # async event loop (run_full_pipeline is CPU-bound / synchronous).
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_full_pipeline, df, threshold)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scan execution error: {str(e)}")

    if not result.get("success"):
        raise HTTPException(status_code=422, detail={
            "message": "Validation failed",
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
        })

    print(f"[SCAN COMPLETED] rows={result['total_rows']}, fraud={result['fraud_count']}, time={result['processing_time_ms']}ms")
    result["filename"] = file.filename
    result["uploaded_by"] = current_user.username
    print("[SCAN API] Response returned")
    return result


@router.post("/validate")
async def validate_only(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    """
    Validate a transaction CSV without running predictions.
    Returns a column-level quality report.
    """
    if loader.get_preprocessor() is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    report = validate_csv(df)

    # Add column-level detail
    preprocessor = loader.get_preprocessor()
    expected = set(preprocessor.all_cols) if preprocessor else set()
    uploaded = set(df.columns)

    report["column_detail"] = {
        "matched":  sorted(uploaded & expected),
        "missing":  sorted(expected - uploaded),
        "extra":    sorted(uploaded - expected - {"TransactionID", "isFraud"}),
    }
    report["filename"] = file.filename
    report["preview"] = df.head(5).fillna("").to_dict(orient="records")

    return report


@router.get("/sample")
def download_sample(current_user=Depends(get_current_user)):
    """
    Download a sample transaction CSV that can be uploaded for prediction.
    Uses 10 rows from the training data (if available) or generates synthetic data.
    """
    sample_path = ROOT / "sample_transactions.csv"

    if sample_path.exists():
        with open(sample_path, "r") as f:
            content = f.read()
    else:
        # Generate a minimal synthetic sample
        content = _generate_synthetic_sample()

    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"},
    )


@router.get("/{job_id}/download")
def download_results(
    job_id: str,
    current_user=Depends(get_current_user),
):
    """
    Download prediction results as CSV.
    Note: This is a placeholder — in production you'd store results by job_id.
    For now, the frontend sends the results directly from the upload response.
    """
    raise HTTPException(
        status_code=404,
        detail="Results not found. Use the download button in the frontend which exports directly from the response data."
    )


def _generate_synthetic_sample() -> str:
    """
    A downloadable sample CSV: real rows from the dataset, not invented ones.

    This used to fabricate every value -- 19 V-columns of rng.uniform(0, 1)
    rounded to 4 decimals, D-columns of rng.uniform(0, 100), and nothing else
    of the 339 V-columns the model was trained on. Real V1 is exactly 1.0, real
    D1 is a whole number of days. A user who downloaded the sample and uploaded
    it got predictions driven entirely by imputed medians and concluded the
    model was broken. (synthetic_corrected_2500.csv in the repo root has the
    same fingerprint and is almost certainly its output.)

    Real rows guarantee the sample round-trips through the pipeline correctly.
    """
    import pandas as pd

    n = 10
    tx_path = ROOT / "train_transaction.csv"
    idt_path = ROOT / "train_identity.csv"

    if tx_path.exists():
        tx = pd.read_csv(tx_path, nrows=n, low_memory=False)
        if idt_path.exists():
            idt = pd.read_csv(idt_path, low_memory=False)
            tx = tx.merge(idt, on="TransactionID", how="left")
        # Ship it as an unlabelled sample: the label is what we are predicting.
        tx = tx.drop(columns=["isFraud"], errors="ignore")
        buf = io.StringIO()
        tx.to_csv(buf, index=False)
        return buf.getvalue()

    # Without the raw CSVs there is nothing honest to hand back.
    raise HTTPException(
        status_code=503,
        detail="Sample unavailable: train_transaction.csv not found. The "
               "sample is drawn from real data rather than generated, so the "
               "source file must be present.",
    )
