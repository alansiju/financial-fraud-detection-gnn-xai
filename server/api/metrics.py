"""api/metrics.py - Model performance metrics: ROC, PR, confusion matrix"""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from server.auth.dependencies import get_current_user
from server.ml import loader, predictor

router = APIRouter(prefix="/api/metrics", tags=["Model Metrics"])


@router.get("")
def get_metrics(current_user=Depends(get_current_user)):
    data = loader.get_data()
    if data is None or not loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model or graph not loaded yet - metrics unavailable.",
        )

    try:
        from sklearn.metrics import (
            roc_curve, precision_recall_curve,
            roc_auc_score, average_precision_score,
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix,
        )
        probs     = predictor.get_all_predictions()
        test_mask = data.test_mask.cpu().numpy()
        y_true    = data.y.cpu().numpy()[test_mask].astype(int)
        y_prob    = probs[test_mask]
        # The validation-tuned threshold, not a hardcoded 0.5. Reporting this
        # model at 0.5 understated recall badly: its probabilities peaked
        # around 0.65 with a median of 0.19.
        threshold = loader.get_threshold()
        y_pred    = (y_prob >= threshold).astype(int)

        # ROC curve (sampled for response size)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        step  = max(1, len(fpr) // 200)
        roc   = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                 for f, t in zip(fpr[::step], tpr[::step])]

        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        step = max(1, len(prec) // 200)
        pr   = [{"precision": round(float(p), 4), "recall": round(float(r), 4)}
                for p, r in zip(prec[::step], rec[::step])]

        # Confusion matrix
        cm    = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Training curves.
        # Read defensively: train.py now logs val_pr_auc / val_f1 / threshold
        # and no longer logs val_loss (it evaluates ranking metrics, not loss,
        # each epoch). Indexing e["val_loss"] directly raised a KeyError that
        # the old `except: return _mock_metrics()` swallowed -- the endpoint
        # kept returning 200 with invented numbers instead of reporting that
        # the curve data had changed shape.
        log   = loader.get_training_log()
        curves = []
        if log:
            step = max(1, len(log) // 100)
            for e in log[::step]:
                point = {"epoch": e.get("epoch")}
                for key in ("train_loss", "val_loss", "val_auc",
                            "val_pr_auc", "val_f1", "threshold"):
                    if e.get(key) is not None:
                        point[key] = round(float(e[key]), 6)
                curves.append(point)

        # Baselines
        baselines = loader.get_baseline_metrics()

        return {
            "threshold":        round(float(threshold), 4),
            "active_model":     loader.get_active_model(),
            "model_name":       loader.get_active_model_name(),
            "split_mode":       getattr(data, "split_mode", "unknown"),
            "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
            "precision":        round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":           round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":               round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc":          round(float(roc_auc_score(y_true, y_prob)), 4),
            "pr_auc":           round(float(average_precision_score(y_true, y_prob)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_curve":        roc,
            "pr_curve":         pr,
            "training_curves":  curves,
            "baselines":        baselines,
        }
    except Exception as e:
        # This used to `return _mock_metrics()` — invented numbers (accuracy
        # 0.974, precision 0.82, recall 0.71) rendered on the dashboard as if
        # they were real, with nothing to signal that scoring had failed.
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Metrics computation failed: {type(e).__name__}: {e}",
        )
