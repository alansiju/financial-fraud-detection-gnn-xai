"""
api/dashboard.py - KPI cards + overview stats.

What changed
------------
* `_mock_dashboard()` returned hardcoded numbers (accuracy 0.974, ROC-AUC
  0.896, precision 0.82, recall 0.71) whenever the graph was not loaded, and
  `_compute_test_metrics()` returned the same invented figures from a bare
  `except`. The KPI cards on the landing page could therefore show confident
  metrics that no model had produced. Both now fail loudly.

* `model_name` was the string "GraphSAGE-3L" regardless of what was actually
  loaded; the current model has 2 graph layers. It is now derived from the
  checkpoint config.
"""
from fastapi import APIRouter, Depends, HTTPException

from server.auth.dependencies import get_current_user
from server.ml import loader, predictor
from server.config import APP_VERSION

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("")
def get_dashboard(current_user=Depends(get_current_user)):
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(status_code=503,
                            detail="Model or graph not loaded yet.")

    labels   = data.y.cpu().numpy()
    total    = len(labels)
    fraud_ct = int(labels.sum())
    legit_ct = total - fraud_ct
    fraud_pct = round(fraud_ct / total * 100, 2) if total else 0

    # Model metrics from training log
    log = loader.get_training_log()
    best_auc = 0.0
    best_pr = 0.0
    if log:
        best_auc = max(e.get("val_auc", 0) for e in log)
        best_pr = max(e.get("val_pr_auc", 0) for e in log)

    # Quick test-set metrics
    test_metrics = _compute_test_metrics()

    return {
        "total_transactions":   total,
        "fraud_detected":       fraud_ct,
        "legitimate":           legit_ct,
        "fraud_percentage":     fraud_pct,
        "model_accuracy":       test_metrics["accuracy"],
        "roc_auc":              test_metrics["roc_auc"],
        "precision":            test_metrics["precision"],
        "recall":               test_metrics["recall"],
        "f1_score":             test_metrics["f1"],
        "best_val_auc":         round(best_auc, 6),
        "best_val_pr_auc":      round(best_pr, 6),
        "threshold":            round(loader.get_threshold(), 4),
        "split_mode":           getattr(data, "split_mode", "unknown"),
        "model_name":           _model_name(),
        "model_version":        APP_VERSION,
        "graph_nodes":          data.num_nodes,
        "graph_edges":          data.num_edges,
        "is_model_loaded":      loader.is_loaded(),
    }


def _compute_test_metrics():
    """Lightweight test-set evaluation."""
    try:
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, precision_score,
            recall_score, f1_score
        )
        data      = loader.get_data()
        probs     = predictor.get_all_predictions()
        test_mask = data.test_mask.cpu().numpy()
        y_true    = data.y.cpu().numpy()[test_mask]
        y_prob    = probs[test_mask]
        # Tuned threshold from best_model.pt, not a hardcoded 0.5.
        y_pred    = (y_prob >= loader.get_threshold()).astype(int)

        return {
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }
    except Exception as e:
        # Previously returned invented metrics here, which surfaced on the
        # dashboard indistinguishable from real ones.
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Test-set metrics unavailable: {type(e).__name__}: {e}",
        )


def _model_name() -> str:
    """Describe the model that is actually loaded, not a fixed string."""
    # The model actually serving, not always the GNN: with the ensemble
    # active this used to label the ensemble's numbers "GraphSAGE-2L +skip".
    return loader.get_active_model_name()
