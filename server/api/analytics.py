"""
api/analytics.py - Chart data for the analytics dashboard.

What changed
------------
Every breakdown on this endpoint used to be invented:

    fraud_by_month  : 0.03 + 0.02 * sin(i * pi / 6)      -- a sine wave
    fraud_by_device : rng.uniform(0.02, 0.08)            -- random
    fraud_by_card   : rng.uniform(0.02, 0.06)            -- random
    fraud_by_email  : rng.uniform(0.02, 0.05)            -- random
    amounts         : abs(x[:, 0]) * 500 + 10            -- a standardised
                                                            feature read as a
                                                            dollar amount
    DEVICE_TYPES    : ["Mobile", "Desktop", "Tablet", "POS Terminal", "ATM",
                       "Web Browser"] -- categories that do not exist in
                       IEEE-CIS, which only has 'desktop' and 'mobile'

It looked like analysis and was noise. The endpoint now reads the real columns
from outputs/models/node_meta.pkl (written by save_node_meta.py) and reports
actual distributions. If that artifact is missing it says so instead of
fabricating a replacement.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from server.auth.dependencies import get_current_user
from server.ml import loader, predictor

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

AMOUNT_BUCKETS = [0, 50, 100, 250, 500, 1000, 5000, np.inf]
AMOUNT_NAMES = ["<$50", "$50-100", "$100-250", "$250-500",
                "$500-1k", "$1k-5k", ">$5k"]
TOP_N = 8


def _breakdown(meta: pd.DataFrame, labels: np.ndarray, flagged: np.ndarray,
               col: str, key: str, top_n: int = TOP_N) -> list:
    """Real counts per category, most frequent first."""
    if col not in meta.columns:
        return []
    series = meta[col].fillna("(missing)").astype(str)
    order = series.value_counts().head(top_n).index
    out = []
    for val in order:
        mask = (series == val).to_numpy()
        out.append({
            key: val,
            "total": int(mask.sum()),
            "fraud": int(labels[mask].sum()),
            "flagged": int(flagged[mask].sum()),
            "fraud_rate": round(float(labels[mask].mean() * 100), 3),
        })
    return out


@router.get("/summary")
def get_summary(current_user=Depends(get_current_user)):
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(status_code=503,
                            detail="Model or graph not loaded yet.")

    meta = loader.get_node_meta()
    if meta is None:
        raise HTTPException(
            status_code=503,
            detail="node_meta.pkl not found - analytics needs the raw "
                   "transaction columns. Run: py save_node_meta.py",
        )

    labels = data.y.cpu().numpy()
    n = len(labels)
    if len(meta) != n:
        raise HTTPException(
            status_code=503,
            detail=f"node_meta has {len(meta):,} rows but the graph has "
                   f"{n:,} nodes. Rebuild with: py save_node_meta.py",
        )

    probs = predictor.get_all_predictions()
    flagged = probs >= loader.get_threshold()

    # ── Fraud over time (real timestamps) ──────────────────────────────────
    fraud_by_month = []
    if "month" in meta.columns:
        grp = meta.groupby("month", sort=False)
        # Keep chronological order rather than alphabetical.
        months = meta.drop_duplicates("month").sort_values("TransactionDT")["month"]
        for m in months:
            idx = (meta["month"] == m).to_numpy()
            total = int(idx.sum())
            fraud = int(labels[idx].sum())
            fraud_by_month.append({
                "month": m, "total": total,
                "fraud": fraud, "legit": total - fraud,
                "fraud_rate": round(100 * fraud / max(total, 1), 3),
            })

    # ── Fraud by real transaction amount ───────────────────────────────────
    fraud_by_amount = []
    if "TransactionAmt" in meta.columns:
        amounts = pd.to_numeric(meta["TransactionAmt"],
                                errors="coerce").fillna(0).to_numpy()
        for i, name in enumerate(AMOUNT_NAMES):
            mask = (amounts >= AMOUNT_BUCKETS[i]) & (amounts < AMOUNT_BUCKETS[i + 1])
            total = int(mask.sum())
            fraud_by_amount.append({
                "bucket": name, "total": total,
                "fraud": int(labels[mask].sum()),
                "fraud_rate": round(float(labels[mask].mean() * 100), 3)
                if total else 0.0,
            })

    total_fraud = int(labels.sum())

    return {
        "source": "node_meta.pkl (real transaction columns)",
        "threshold": round(loader.get_threshold(), 4),
        "fraud_by_month": fraud_by_month,
        "fraud_by_amount": fraud_by_amount,
        "fraud_by_device": _breakdown(meta, labels, flagged, "DeviceType", "device"),
        "fraud_by_card": _breakdown(meta, labels, flagged, "card4", "card_type"),
        "fraud_by_email": _breakdown(meta, labels, flagged, "P_emaildomain", "domain"),
        "fraud_by_product": _breakdown(meta, labels, flagged, "ProductCD", "product"),
        "fraud_by_hour": [
            {
                "hour": int(h),
                "total": int((meta["hour"] == h).sum()),
                "fraud": int(labels[(meta["hour"] == h).to_numpy()].sum()),
            }
            for h in range(24)
        ] if "hour" in meta.columns else [],
        "class_dist": [
            {"name": "Legitimate", "value": n - total_fraud},
            {"name": "Fraud", "value": total_fraud},
        ],
    }
