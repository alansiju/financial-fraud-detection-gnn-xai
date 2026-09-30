"""
ml/predictor.py - Inference for nodes that already exist in the training graph.

What changed
------------
* Every call used to run `model(data.x, data.edge_index)` — a full-graph forward
  pass over 590,540 nodes — to read back a single node's probability. On the
  rebuilt graph message passing materialises a [num_edges, num_features] tensor
  that does not fit in memory at all, and even on the old graph it made a
  single-node request cost a full inference sweep. Probabilities are now
  computed once by loader.get_all_node_probs() and cached.

* `threshold` defaults to the validation-tuned value from the checkpoint rather
  than a hardcoded 0.5.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from server.ml import loader


def _resolve_threshold(threshold: Optional[float]) -> float:
    return loader.get_threshold() if threshold is None else float(threshold)


def _probs() -> np.ndarray:
    probs = loader.get_all_node_probs()
    if probs is None:
        raise RuntimeError("Model not loaded")
    return probs


def _result(node_id: int, prob: float, threshold: float) -> Dict[str, Any]:
    data = loader.get_data()
    is_fraud = bool(prob >= threshold)
    label = int(data.y[node_id].item()) if hasattr(data, "y") else None
    return {
        "node_id": int(node_id),
        "fraud_probability": round(float(prob), 6),
        "is_fraud": is_fraud,
        "confidence": round(float(prob if is_fraud else 1.0 - prob), 6),
        "threshold": round(threshold, 4),
        "true_label": label,
        "risk_level": _risk_level(prob, threshold),
    }


def predict_node(node_id: int, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Inference for a single graph node."""
    data = loader.get_data()
    if data is None or loader.get_model() is None:
        raise RuntimeError("Model not loaded")

    n = data.num_nodes
    if node_id < 0 or node_id >= n:
        raise ValueError(f"node_id {node_id} out of range [0, {n})")

    return _result(node_id, _probs()[node_id], _resolve_threshold(threshold))


def predict_batch(node_ids: List[int],
                  threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Inference for many nodes."""
    data = loader.get_data()
    if data is None or loader.get_model() is None:
        raise RuntimeError("Model not loaded")

    thr = _resolve_threshold(threshold)
    probs = _probs()
    return [_result(nid, probs[nid], thr)
            for nid in node_ids if 0 <= nid < data.num_nodes]


def get_all_predictions(threshold: Optional[float] = None) -> np.ndarray:
    """Probability array for every node (used by metrics + analytics)."""
    return _probs()


def _risk_level(prob: float, threshold: Optional[float] = None) -> str:
    """
    Risk bands anchored to the operating threshold.

    Fixed cut-offs at 0.8/0.6/0.4 labelled essentially everything "LOW", because
    this model's probabilities are concentrated well below 0.5. Scaling the
    bands by the tuned threshold keeps the labels meaningful whatever the
    model's calibration turns out to be.
    """
    thr = loader.get_threshold() if threshold is None else threshold
    # Bands split the interval between the threshold and certainty, so they
    # stay meaningful wherever the threshold sits. Scaling the threshold
    # instead (thr * 2.0) put CRITICAL above 1.0 for any threshold over 0.5,
    # making the top band unreachable.
    headroom = max(1.0 - thr, 1e-6)
    if prob >= thr + headroom * 0.66:
        return "CRITICAL"
    if prob >= thr + headroom * 0.33:
        return "HIGH"
    if prob >= thr:
        return "MEDIUM"
    if prob >= thr * 0.5:
        return "LOW"
    return "MINIMAL"
