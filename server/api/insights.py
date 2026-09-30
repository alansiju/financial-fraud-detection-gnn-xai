"""
api/insights.py — operating-point sweep, drift monitoring, global explainability.

Three things the app could not answer before:

1. "What happens if I move the threshold?"  The Settings page had a slider and
   the Performance page had one confusion matrix, but nothing connected them.
   /threshold-sweep returns the full confusion matrix across the threshold
   range in one call, so the UI can show the trade-off live while dragging
   rather than round-tripping per step.

2. "What does a false negative actually cost?"  Fraud has asymmetric costs —
   a missed fraud loses the transaction value, a false alarm costs review
   minutes. Best-F1 implicitly assumes they are equal, which they never are.
   The sweep carries the counts needed to price any cost pair client-side.

3. "Is the model still valid?"  The chronological split exposed real drift
   (validation PR-AUC 0.473 against test 0.415). /drift quantifies it with PSI
   over the score distribution instead of leaving it as a caveat in a report.
"""

from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from server.auth.dependencies import get_current_user
from server.ml import loader, predictor

router = APIRouter(prefix="/api/insights", tags=["Insights"])


def _test_arrays():
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model or graph not loaded.")
    probs = predictor.get_all_predictions()
    mask = data.test_mask.cpu().numpy()
    return data, probs[mask], data.y.cpu().numpy()[mask].astype(int)


# ═══════════════════════════════════════════════════════════════════════════
#  1. THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/threshold-sweep")
def threshold_sweep(
    steps: int = Query(99, ge=20, le=400),
    current_user=Depends(get_current_user),
):
    """
    Confusion matrix at every threshold across the range, on the test split.

    Returned as raw counts rather than pre-computed rates so the client can
    derive precision, recall, F1 and any cost model without another request.
    """
    data, y_prob, y_true = _test_arrays()

    order = np.argsort(y_prob)
    sorted_probs = y_prob[order]
    sorted_true = y_true[order]

    total = len(y_true)
    total_pos = int(sorted_true.sum())
    total_neg = total - total_pos

    # Cumulative counts below each cut let the whole sweep be one pass instead
    # of one full comparison per threshold.
    cum_pos = np.concatenate([[0], np.cumsum(sorted_true)])
    thresholds = np.linspace(0.01, 0.99, steps)
    idx = np.searchsorted(sorted_probs, thresholds, side="left")

    fn = cum_pos[idx]                 # positives scored below the cut
    tn = idx - fn                     # negatives below the cut
    tp = total_pos - fn
    fp = total_neg - tn

    points: List[Dict[str, Any]] = []
    for i, t in enumerate(thresholds):
        TP, FP, FN, TN = int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i])
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        points.append({
            "threshold": round(float(t), 4),
            "tp": TP, "fp": FP, "fn": FN, "tn": TN,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(2 * prec * rec / (prec + rec), 6) if (prec + rec) else 0.0,
            "flagged": TP + FP,
            "flagged_pct": round((TP + FP) / total * 100, 4),
        })

    return {
        "split_mode": getattr(data, "split_mode", "unknown"),
        "n_test": total,
        "n_fraud": total_pos,
        "base_rate": round(total_pos / total, 6),
        "current_threshold": round(loader.get_threshold(), 4),
        "model_tuned_threshold": round(loader.get_tuned_threshold(), 4),
        "points": points,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  2. DRIFT
# ═══════════════════════════════════════════════════════════════════════════

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two score distributions.

    Convention: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant.
    Bin edges come from the expected (reference) distribution's quantiles so
    that an even reference spread is guaranteed.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    # Floor the shares so an empty bin does not send the log to infinity.
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


@router.get("/drift")
def drift(current_user=Depends(get_current_user)):
    """Score-distribution drift between train, validation and test."""
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model or graph not loaded.")

    probs = predictor.get_all_predictions()
    y = data.y.cpu().numpy().astype(int)
    tr = data.train_mask.cpu().numpy()
    va = data.val_mask.cpu().numpy()
    te = data.test_mask.cpu().numpy()

    def hist(p, bins=20):
        counts, edges = np.histogram(p, bins=bins, range=(0, 1))
        share = counts / max(counts.sum(), 1)
        return [
            {"bin": round(float((edges[i] + edges[i + 1]) / 2), 4),
             "share": round(float(share[i]), 6),
             "count": int(counts[i])}
            for i in range(len(counts))
        ]

    splits = {}
    for name, m in (("train", tr), ("val", va), ("test", te)):
        splits[name] = {
            "n": int(m.sum()),
            "fraud_rate": round(float(y[m].mean()), 6),
            "mean_score": round(float(probs[m].mean()), 6),
            "median_score": round(float(np.median(probs[m])), 6),
            "histogram": hist(probs[m]),
        }

    psi_val = _psi(probs[tr], probs[va])
    psi_test = _psi(probs[tr], probs[te])

    def band(v):
        return "stable" if v < 0.1 else ("moderate" if v < 0.25 else "significant")

    # Fraud rate per month, so drift can be read against real time.
    timeline = []
    meta = loader.get_node_meta()
    if meta is not None and "month" in meta.columns and len(meta) == len(y):
        months = meta.drop_duplicates("month").sort_values("TransactionDT")["month"]
        split_of = np.where(tr, "train", np.where(va, "val", "test"))
        for mth in months:
            sel = (meta["month"] == mth).to_numpy()
            if not sel.any():
                continue
            vals, counts = np.unique(split_of[sel], return_counts=True)
            timeline.append({
                "month": mth,
                "n": int(sel.sum()),
                "fraud_rate": round(float(y[sel].mean() * 100), 4),
                "mean_score": round(float(probs[sel].mean()), 6),
                "split": str(vals[np.argmax(counts)]),
            })

    return {
        "split_mode": getattr(data, "split_mode", "unknown"),
        "splits": splits,
        "psi": {
            "train_vs_val": {"value": round(psi_val, 4), "band": band(psi_val)},
            "train_vs_test": {"value": round(psi_test, 4), "band": band(psi_test)},
        },
        "timeline": timeline,
        "note": (
            "PSI convention: <0.10 stable · 0.10-0.25 moderate · >0.25 significant. "
            "On a chronological split some drift is expected and real — it is the "
            "reason validation scores overstate test scores here."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  3. GLOBAL FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

_perm_cache: Dict[str, Any] = {}


@router.get("/global-importance")
def global_importance(
    top_k: int = Query(20, ge=5, le=60),
    sample: int = Query(4000, ge=500, le=20000),
    current_user=Depends(get_current_user),
):
    """
    Which features drive the model overall, rather than for one transaction.

    GNNExplainer answers "why this node"; it cannot answer "what has this model
    learned". Permutation importance does: shuffle one column across a sample
    of test nodes, re-score, and measure how much average precision falls.

    Computed once and cached — it is a few hundred forward passes.
    """
    if _perm_cache.get("top_k", 0) >= top_k and _perm_cache.get("sample") == sample:
        return _perm_cache["payload"]

    import torch
    from sklearn.metrics import average_precision_score
    from graph_sampler import NeighborSampleLoader

    data = loader.get_data()
    model = loader.get_model()
    if data is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    names = loader.get_feature_names()
    device = loader.get_device()
    csr = loader.get_full_csr()

    rng = np.random.default_rng(42)
    test_idx = np.where(data.test_mask.cpu().numpy())[0]
    take = rng.choice(test_idx, size=min(sample, len(test_idx)), replace=False)
    take = np.sort(take)
    seeds = torch.from_numpy(take).long()
    y_true = data.y.cpu().numpy()[take].astype(int)

    def score_with(x_override=None):
        x = data.x if x_override is None else x_override
        ldr = NeighborSampleLoader(x, data.y, csr, seeds, [60, 30],
                                   batch_size=4096, shuffle=False)
        out = []
        model.eval()
        with torch.no_grad():
            for batch in ldr:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                out.append(torch.sigmoid(logits[:batch.batch_size]).squeeze(1).cpu())
        return torch.cat(out).numpy()

    base = average_precision_score(y_true, score_with())

    # Rank candidates cheaply first: a column that barely varies across the
    # sample cannot be carrying much, so only the most variable ones are worth
    # a full permutation pass.
    sub = data.x[seeds].numpy()
    variability = sub.std(axis=0)
    candidates = np.argsort(variability)[::-1][: min(top_k * 2, len(names))]

    results = []
    x_work = data.x.clone()
    for col in candidates:
        original = x_work[:, col].clone()
        perm = original[seeds][torch.randperm(len(seeds))]
        x_work[seeds, col] = perm
        dropped = base - average_precision_score(y_true, score_with(x_work))
        x_work[:, col] = original
        results.append({
            "feature": names[col] if col < len(names) else f"feature_{col}",
            "importance": round(float(dropped), 6),
            "importance_pct": round(float(dropped / base * 100), 3) if base else 0.0,
        })

    results.sort(key=lambda r: r["importance"], reverse=True)
    payload = {
        "method": "permutation importance on the test split (drop in average precision)",
        "baseline_pr_auc": round(float(base), 6),
        "n_sampled": int(len(take)),
        "n_features_tested": int(len(candidates)),
        "features": results[:top_k],
        "note": (
            "Only the most variable columns are permuted — a near-constant "
            "column cannot carry signal and permuting it wastes a forward pass."
        ),
    }
    _perm_cache.update({"top_k": top_k, "sample": sample, "payload": payload})
    return payload
