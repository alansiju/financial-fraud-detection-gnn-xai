"""
api/transactions.py - Paginated transaction list + detail.

What changed
------------
Rows in this table were mostly invented:

    amount       : abs(x[i, 0]) * 500 + 10   -- x is standardised features,
                                                not dollars
    device       : DEVICE_TYPES[rng.integers(...)]   -- random, from a list of
                   categories ("Tablet", "POS Terminal", "ATM") that do not
                   exist in IEEE-CIS
    email_domain : random
    card_type    : random
    product_cat  : random, from invented names ("Electronics", "Travel", ...)
                   rather than the real ProductCD codes
    timestamp    : _fake_ts() -- a random date in 2019

So the analyst-facing transaction table showed a fabricated card type, device
and date against every real fraud probability, and the search and filter
controls operated on that noise. It now reads outputs/models/node_meta.pkl.

`is_fraud` also used a hardcoded 0.5 rather than the tuned threshold.
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from server.auth.dependencies import get_current_user
from server.ml import loader, predictor

router = APIRouter(prefix="/api/transactions", tags=["Transactions"])

_tx_cache = None


def _build_cache():
    global _tx_cache
    if _tx_cache is not None:
        return
    data = loader.get_data()
    if data is None:
        _tx_cache = []
        return

    meta = loader.get_node_meta()
    if meta is None:
        raise HTTPException(
            status_code=503,
            detail="node_meta.pkl not found - the transaction list needs the "
                   "raw columns. Run: py save_node_meta.py",
        )

    probs = predictor.get_all_predictions()
    labels = data.y.cpu().numpy()
    threshold = loader.get_threshold()
    n = data.num_nodes

    amt = pd.to_numeric(meta.get("TransactionAmt"), errors="coerce").fillna(0.0)
    ts = meta["timestamp"] if "timestamp" in meta.columns else None

    def col(name, default="(unknown)"):
        if name in meta.columns:
            return meta[name].fillna(default).astype(str).tolist()
        return [default] * n

    devices = col("DeviceType")
    emails = col("P_emaildomain")
    cards = col("card4")
    products = col("ProductCD")
    amounts = amt.tolist()
    stamps = (ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
              if ts is not None else [None] * n)

    rows = []
    for i in range(n):
        prob = float(probs[i])
        real_tx = loader.get_tx_id_by_node(i)
        rows.append({
            "id": i,
            "transaction_id": str(real_tx) if real_tx is not None else str(i),
            "amount": round(float(amounts[i]), 2),
            "fraud_probability": round(prob, 4),
            "is_fraud": bool(prob >= threshold),
            "true_label": int(labels[i]),
            "risk_level": predictor._risk_level(prob, threshold),
            "device": devices[i],
            "email_domain": emails[i],
            "card_type": cards[i],
            "product_cat": products[i],
            "timestamp": stamps[i],
        })
    _tx_cache = rows


@router.get("")
def list_transactions(
    page:    int   = Query(1, ge=1),
    limit:   int   = Query(20, ge=1, le=200),
    fraud:   str   = Query(None, description="all|fraud|legit"),
    search:  str   = Query(None),
    sort:    str   = Query("id"),
    order:   str   = Query("asc"),
    device:  str   = Query(None),
    email:   str   = Query(None),
    min_amt: float = Query(None),
    max_amt: float = Query(None),
    current_user=Depends(get_current_user),
):
    _build_cache()
    rows = list(_tx_cache)

    # Filters
    if fraud == "fraud":
        rows = [r for r in rows if r["is_fraud"]]
    elif fraud == "legit":
        rows = [r for r in rows if not r["is_fraud"]]
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r["transaction_id"].lower() or s in r["device"].lower()]
    if device:
        rows = [r for r in rows if r["device"].lower() == device.lower()]
    if email:
        rows = [r for r in rows if r["email_domain"].lower() == email.lower()]
    if min_amt is not None:
        rows = [r for r in rows if r["amount"] >= min_amt]
    if max_amt is not None:
        rows = [r for r in rows if r["amount"] <= max_amt]

    # Sort
    reverse = order.lower() == "desc"
    try:
        rows.sort(key=lambda r: r.get(sort, r["id"]), reverse=reverse)
    except TypeError:
        pass

    total = len(rows)
    start = (page - 1) * limit
    rows  = rows[start: start + limit]

    return {
        "total":       total,
        "page":        page,
        "limit":       limit,
        "total_pages": max(1, (total + limit - 1) // limit),
        "data":        rows,
    }


@router.get("/facets")
def get_facets(current_user=Depends(get_current_user)):
    """
    Distinct values available for the filter dropdowns.

    The UI used to hardcode ['Mobile','Desktop','Tablet','POS Terminal','ATM',
    'Web Browser'] for the device filter. IEEE-CIS only has 'desktop' and
    'mobile', so once the table started serving real values every one of those
    options matched nothing. Serving the facets keeps the filters honest.
    """
    _build_cache()
    meta = loader.get_node_meta()
    if meta is None:
        raise HTTPException(status_code=503, detail="node_meta.pkl not found")

    def distinct(col, limit=25):
        if col not in meta.columns:
            return []
        vc = meta[col].fillna("(missing)").astype(str).value_counts().head(limit)
        return [{"value": v, "count": int(c)} for v, c in vc.items()]

    return {
        "devices": distinct("DeviceType"),
        "email_domains": distinct("P_emaildomain"),
        "card_types": distinct("card4"),
        "products": distinct("ProductCD"),
        "amount_range": {
            "min": float(meta["TransactionAmt"].min()),
            "max": float(meta["TransactionAmt"].max()),
        } if "TransactionAmt" in meta.columns else None,
        "threshold": round(loader.get_threshold(), 4),
    }


@router.get("/{identifier}")
def get_transaction(identifier: int, current_user=Depends(get_current_user)):
    _build_cache()
    data = loader.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="Graph not loaded")

    node_id = loader.resolve_node_id(identifier)
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail="Transaction not found")

    result = predictor.predict_node(node_id)
    x_vec  = data.x[node_id].cpu().numpy().tolist()

    # Neighbours, read from the cached CSR adjacency. Comparing the full
    # 4.5M-column edge_index array twice per request was the slowest thing on
    # this endpoint.
    csr = loader.get_full_csr()
    if csr is not None:
        lo, hi = csr.rowptr[node_id], csr.rowptr[node_id + 1]
        neighbours = [int(n) for n in csr.col[lo:hi][:20]]
    else:
        neighbours = []

    real_tx = loader.get_tx_id_by_node(node_id)
    tx_str  = str(real_tx) if real_tx is not None else str(identifier)

    # Real attributes, same source as the list view. This endpoint previously
    # drew device/email/card/product from np.random.default_rng(node_id) and
    # reconstructed "amount" as abs(x[0]) * 500 + 10 from a standardised
    # feature, so the detail page for a transaction disagreed with reality and
    # with nothing in particular.
    meta = loader.get_node_meta()

    def mv(col, default=None):
        if meta is None or col not in meta.columns:
            return default
        v = meta[col].iloc[node_id]
        if v is None or (isinstance(v, float) and v != v):
            return default
        return v

    ts = mv("timestamp")
    row = {
        **result,
        "transaction_id": tx_str,
        "device":         str(mv("DeviceType", "(unknown)")),
        "email_domain":   str(mv("P_emaildomain", "(unknown)")),
        "card_type":      str(mv("card4", "(unknown)")),
        "card_network":   str(mv("card6", "(unknown)")),
        "product_cat":    str(mv("ProductCD", "(unknown)")),
        "device_info":    str(mv("DeviceInfo", "(unknown)")),
        "billing_zip":    (int(mv("addr1")) if mv("addr1") is not None else None),
        "amount":         (round(float(mv("TransactionAmt", 0.0)), 2)),
        "timestamp":      (ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts is not None else None),
        "feature_vector": x_vec[:20],
        "feature_names":  loader.get_feature_names()[:20],
        "neighbours":     neighbours,
        "degree":         len(neighbours),
    }
    return row
