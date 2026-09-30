"""api/predict.py - Single, batch, and CSV upload prediction endpoints"""
import io
import json
import csv as csv_module
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.auth.dependencies import get_current_user
from server.database import get_db, PredictionLog
from server.ml import loader, predictor
from server.config import DEFAULT_FRAUD_THRESHOLD

# None -> server.ml.loader.get_threshold(), the value train.py tuned on
# validation and stored in best_model.pt. Callers may still override.

router = APIRouter(prefix="/api/predict", tags=["Prediction"])


class PredictNodeRequest(BaseModel):
    node_id:   int
    threshold: Optional[float] = None


class BatchPredictRequest(BaseModel):
    node_ids:  List[int]
    threshold: Optional[float] = None


@router.post("/node")
def predict_single(
    req: PredictNodeRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Model/graph not loaded")

    node_id = loader.resolve_node_id(req.node_id)
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Transaction or node '{req.node_id}' not found in graph")

    try:
        result = predictor.predict_node(node_id, req.threshold)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    real_tx = loader.get_tx_id_by_node(node_id)
    result["transaction_id"] = real_tx if real_tx is not None else req.node_id

    # Log to DB
    db.add(PredictionLog(
        node_id       = node_id,
        fraud_prob    = result["fraud_probability"],
        is_fraud      = result["is_fraud"],
        confidence    = result["confidence"],
        threshold_used = result["threshold"],
        username      = current_user.username,
    ))
    db.commit()
    return result


@router.post("/batch")
def predict_batch(
    req: BatchPredictRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Model/graph not loaded")

    if len(req.node_ids) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 nodes per batch")

    resolved_ids = []
    for nid in req.node_ids:
        rnid = loader.resolve_node_id(nid)
        if 0 <= rnid < data.num_nodes:
            resolved_ids.append(rnid)

    results = predictor.predict_batch(resolved_ids, req.threshold)
    for r in results:
        r["transaction_id"] = loader.get_tx_id_by_node(r["node_id"])
        db.add(PredictionLog(
            node_id        = r["node_id"],
            fraud_prob     = r["fraud_probability"],
            is_fraud       = r["is_fraud"],
            confidence     = r["confidence"],
            threshold_used = r["threshold"],
            username       = current_user.username,
        ))
    db.commit()
    fraud_count = sum(1 for r in results if r["is_fraud"])
    return {
        "total":        len(results),
        "fraud_count":  fraud_count,
        "legit_count":  len(results) - fraud_count,
        "results":      results,
    }


@router.post("/upload")
async def predict_csv(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Accept a CSV with a 'TransactionID' or 'node_id' column and predict for each row."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Model/graph not loaded")

    content = await file.read()
    text    = content.decode("utf-8-sig")
    reader  = csv_module.DictReader(io.StringIO(text))

    node_ids = []
    for row in reader:
        nid = (
            row.get("TransactionID")
            or row.get("transaction_id")
            or row.get("node_id")
            or row.get("id")
            or row.get("NodeId")
        )
        if nid:
            try:
                resolved = loader.resolve_node_id(int(float(nid)))
                if 0 <= resolved < data.num_nodes:
                    node_ids.append(resolved)
            except (ValueError, TypeError):
                pass

    if not node_ids:
        raise HTTPException(status_code=400, detail="CSV must contain a 'TransactionID' or 'node_id' column matching graph transactions")
    if len(node_ids) > 5000:
        raise HTTPException(status_code=400, detail="Max 5000 rows per upload")

    results     = predictor.predict_batch(node_ids, threshold)
    for r in results:
        r["transaction_id"] = loader.get_tx_id_by_node(r["node_id"])

    fraud_count = sum(1 for r in results if r["is_fraud"])
    return {
        "filename":    file.filename,
        "rows_parsed": len(node_ids),
        "total":       len(results),
        "fraud_count": fraud_count,
        "legit_count": len(results) - fraud_count,
        "results":     results,
    }


@router.get("/history")
def prediction_history(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":           r.id,
            "node_id":      r.node_id,
            "fraud_prob":   r.fraud_prob,
            "is_fraud":     r.is_fraud,
            "confidence":   r.confidence,
            "created_at":   r.created_at,
            "username":     r.username,
        }
        for r in rows
    ]
