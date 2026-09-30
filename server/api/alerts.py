"""
api/alerts.py — the analyst review queue.

The app could score transactions but had nowhere to *work* them: a ranked list
of what to look at next, and a record of what a human decided. Without that,
the model's output has no feedback loop and no audit trail.

The queue is the flagged transactions ordered by score, minus anything already
dispositioned. Dispositions are persisted, so the precision the analysts
actually observe can be compared against the precision the test split
predicted — which is the only way to find out whether the offline number
survives contact with real reviewers.
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth.dependencies import get_current_user
from server.database import AlertReview, get_db
from server.ml import loader, predictor

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

DISPOSITIONS = {"confirmed_fraud", "dismissed", "escalated"}


class DispositionRequest(BaseModel):
    disposition: str
    note: Optional[str] = None


def _meta_value(meta, col, idx, default=None):
    if meta is None or col not in meta.columns:
        return default
    v = meta[col].iloc[idx]
    if v is None or (isinstance(v, float) and v != v):
        return default
    return v


@router.get("/queue")
def get_queue(
    limit: int = Query(50, ge=1, le=500),
    min_score: Optional[float] = Query(None, ge=0, le=1),
    include_reviewed: bool = Query(False),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Highest-risk unreviewed transactions, ranked by score.

    Ranking is by probability rather than by the flag, so lowering the
    threshold never reshuffles the top of the queue — an analyst working down
    from the riskiest item sees a stable order.
    """
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model or graph not loaded.")

    probs = predictor.get_all_predictions()
    threshold = loader.get_threshold()
    cut = min_score if min_score is not None else threshold

    reviewed = {r.node_id for r in db.query(AlertReview.node_id).all()}

    # argpartition over the flagged set: sorting all 590k scores to show 50
    # rows is wasted work.
    above = np.flatnonzero(probs >= cut)
    if above.size == 0:
        return {"threshold": round(threshold, 4), "cut": round(cut, 4),
                "total_flagged": 0, "total_reviewed": len(reviewed),
                "returned": 0, "items": []}

    order = above[np.argsort(probs[above])[::-1]]
    meta = loader.get_node_meta()
    y = data.y.cpu().numpy()

    items = []
    for node_id in order:
        nid = int(node_id)
        if not include_reviewed and nid in reviewed:
            continue
        ts = _meta_value(meta, "timestamp", nid)
        items.append({
            "node_id": nid,
            "transaction_id": loader.get_tx_id_by_node(nid),
            "fraud_probability": round(float(probs[nid]), 6),
            "risk_level": predictor._risk_level(float(probs[nid]), threshold),
            "amount": (round(float(_meta_value(meta, "TransactionAmt", nid, 0.0)), 2)),
            "card_type": str(_meta_value(meta, "card4", nid, "(unknown)")),
            "device": str(_meta_value(meta, "DeviceType", nid, "(unknown)")),
            "email_domain": str(_meta_value(meta, "P_emaildomain", nid, "(unknown)")),
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts is not None else None,
            "true_label": int(y[nid]),
            "reviewed": nid in reviewed,
        })
        if len(items) >= limit:
            break

    return {
        "threshold": round(threshold, 4),
        "cut": round(cut, 4),
        "model": loader.get_active_model(),
        "total_flagged": int(above.size),
        "total_reviewed": len(reviewed),
        "remaining": int(above.size) - len(reviewed & set(above.tolist())),
        "returned": len(items),
        "items": items,
    }


@router.post("/{node_id}/disposition")
def set_disposition(
    node_id: int,
    body: DispositionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Record an analyst's decision on one alert."""
    if body.disposition not in DISPOSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"disposition must be one of {sorted(DISPOSITIONS)}")

    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Graph not loaded.")
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"node {node_id} not found")

    probs = predictor.get_all_predictions()
    row = db.query(AlertReview).filter_by(node_id=node_id).first()
    if row is None:
        row = AlertReview(node_id=node_id)
        db.add(row)

    row.transaction_id = loader.get_tx_id_by_node(node_id)
    row.disposition = body.disposition
    row.note = body.note
    row.reviewer = current_user.username
    row.fraud_prob = float(probs[node_id])
    row.true_label = int(data.y[node_id].item())
    row.threshold_used = float(loader.get_threshold())
    row.model_used = loader.get_active_model()
    row.reviewed_at = datetime.utcnow()
    db.commit()

    return {
        "node_id": node_id,
        "transaction_id": row.transaction_id,
        "disposition": row.disposition,
        "reviewer": row.reviewer,
        "true_label": row.true_label,
        "agrees_with_label": (
            (row.disposition == "confirmed_fraud" and row.true_label == 1)
            or (row.disposition == "dismissed" and row.true_label == 0)
        ),
    }


@router.delete("/{node_id}/disposition")
def clear_disposition(
    node_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Undo a decision and return the alert to the queue."""
    row = db.query(AlertReview).filter_by(node_id=node_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="no disposition to clear")
    db.delete(row)
    db.commit()
    return {"node_id": node_id, "cleared": True}


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Review throughput, and the precision analysts actually observed.

    Observed precision is the interesting number: it is what the model delivers
    to the people using it, and it can be compared against the precision the
    test split promised.
    """
    rows = db.query(AlertReview).all()
    confirmed = [r for r in rows if r.disposition == "confirmed_fraud"]
    dismissed = [r for r in rows if r.disposition == "dismissed"]
    escalated = [r for r in rows if r.disposition == "escalated"]

    decided = confirmed + dismissed
    observed_precision = (len(confirmed) / len(decided)) if decided else None

    # How often the analyst's call matched the ground-truth label.
    agree = sum(
        1 for r in rows
        if (r.disposition == "confirmed_fraud" and r.true_label == 1)
        or (r.disposition == "dismissed" and r.true_label == 0)
    )

    by_reviewer = {}
    for r in rows:
        by_reviewer.setdefault(r.reviewer or "unknown", 0)
        by_reviewer[r.reviewer or "unknown"] += 1

    return {
        "total_reviewed": len(rows),
        "confirmed_fraud": len(confirmed),
        "dismissed": len(dismissed),
        "escalated": len(escalated),
        "observed_precision": (round(observed_precision, 4)
                               if observed_precision is not None else None),
        "agreement_with_labels": (round(agree / len(rows), 4) if rows else None),
        "by_reviewer": by_reviewer,
        "recent": [
            {
                "node_id": r.node_id,
                "transaction_id": r.transaction_id,
                "disposition": r.disposition,
                "fraud_prob": round(r.fraud_prob or 0, 4),
                "true_label": r.true_label,
                "reviewer": r.reviewer,
                "reviewed_at": r.reviewed_at,
            }
            for r in sorted(rows, key=lambda x: x.reviewed_at or datetime.min,
                            reverse=True)[:20]
        ],
    }
