"""
api/graph.py - Ego-subgraph JSON for the interactive force-graph view.

What changed
------------
* Each node carried a `device` and `email_domain` drawn from
  `np.random.default_rng(n)` over invented category lists ("Tablet",
  "POS Terminal") that do not exist in IEEE-CIS. The graph viewer therefore
  labelled every node with a fabricated device and email. Both now come from
  outputs/models/node_meta.pkl, alongside the real amount and timestamp.

* Neighbour expansion compared the full `edge_index` array once per frontier
  node, and edge collection then looped over all 4.5M directed edges in
  Python. On the rebuilt graph a single depth-2 request did hundreds of
  millions of comparisons. Both now use the cached CSR adjacency, so a request
  touches only the nodes in the neighbourhood.

* Fraud/legit colouring used a hardcoded 0.5 cut-off instead of the tuned
  threshold.
"""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from server.auth.dependencies import get_current_user
from server.ml import loader, predictor

router = APIRouter(prefix="/api/graph", tags=["Graph"])

MAX_SUBGRAPH_NODES = 150


@router.get("/{identifier}")
def get_subgraph(
    identifier: int,
    depth: int = Query(1, ge=1, le=2),
    current_user=Depends(get_current_user),
):
    """Ego-subgraph around a node, as JSON for the force-graph component."""
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Graph not loaded")

    node_id = loader.resolve_node_id(identifier)
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction or node {identifier} not found")

    csr = loader.get_full_csr()
    if csr is None:
        raise HTTPException(status_code=503, detail="Adjacency not built")

    probs = predictor.get_all_predictions()
    labels = data.y.cpu().numpy()
    threshold = loader.get_threshold()
    meta = loader.get_node_meta()

    # ── Neighbourhood expansion over CSR ───────────────────────────────────
    visited = {node_id}
    frontier = [node_id]
    for _ in range(depth):
        nxt = []
        for n in frontier:
            lo, hi = csr.rowptr[n], csr.rowptr[n + 1]
            for nb in csr.col[lo:hi]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
                    if len(visited) >= MAX_SUBGRAPH_NODES:
                        break
            if len(visited) >= MAX_SUBGRAPH_NODES:
                break
        frontier = nxt
        if not frontier or len(visited) >= MAX_SUBGRAPH_NODES:
            break

    def meta_val(col, idx, default="(unknown)"):
        if meta is None or col not in meta.columns:
            return default
        v = meta[col].iloc[idx]
        return default if v is None or (isinstance(v, float) and v != v) else str(v)

    nodes = []
    for n in sorted(visited):
        prob = float(probs[n]) if n < len(probs) else 0.0
        real_tx = loader.get_tx_id_by_node(n)
        nodes.append({
            "id": n,
            "transaction_id": real_tx if real_tx is not None else n,
            "label": str(real_tx) if real_tx is not None else f"Node {n}",
            "type": ("target" if n == node_id
                     else ("fraud" if prob >= threshold else "legit")),
            "fraud_prob": round(prob, 4),
            "true_label": int(labels[n]) if n < len(labels) else -1,
            "device": meta_val("DeviceType", n),
            "email_domain": meta_val("P_emaildomain", n),
            "card_type": meta_val("card4", n),
            "amount": (round(float(meta["TransactionAmt"].iloc[n]), 2)
                       if meta is not None and "TransactionAmt" in meta.columns
                       else None),
        })

    # ── Edges induced on the visited set, read from CSR ─────────────────────
    edges = []
    seen = set()
    for n in visited:
        lo, hi = csr.rowptr[n], csr.rowptr[n + 1]
        for off in range(lo, hi):
            nb = int(csr.col[off])
            if nb not in visited:
                continue
            key = (min(n, nb), max(n, nb))
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "source": key[0],
                "target": key[1],
                "weight": round(float(csr.weight[off]), 4),
            })

    return {
        "center_node": node_id,
        "center_transaction_id": loader.get_tx_id_by_node(node_id) or identifier,
        "threshold": round(threshold, 4),
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "truncated": len(visited) >= MAX_SUBGRAPH_NODES,
    }
