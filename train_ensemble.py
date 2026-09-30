#!/usr/bin/env python3
"""
train_ensemble.py
=================
Blends the GraphSAGE model with LightGBM and tunes the blend on validation.

Why
---
On the honest temporal split the GNN reaches F1 0.442 / PR-AUC 0.415 while
LightGBM reaches 0.514 / 0.520 on the *same* features. That gap is not a bug:
gradient boosting beats neural nets on 469 heterogeneous tabular columns, and
this graph is thin (average degree 7.7, 13% of nodes isolated).

But the two models are wrong about different transactions. The GNN sees
neighbourhood structure LightGBM cannot; LightGBM models the tabular columns
better than the GNN's MLP path. A weighted blend of their probabilities
usually clears both, and costs one extra model at inference.

Everything is fit on train, tuned on validation, and reported on test. The
blend weight and the decision threshold are both validation-tuned — picking
either on test would be the same mistake this project already had once.

Outputs
-------
    outputs/models/lightgbm.pkl
    outputs/models/ensemble.json        (weight, thresholds, metrics)
    outputs/results/model_metrics.csv   (LightGBM + Ensemble rows merged in)

Usage
-----
    py train_ensemble.py
"""

import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.graphsage import GraphSAGE          # noqa: E402
from train import EVAL_NEIGHBORS, csr_view, predict, tune_threshold  # noqa: E402

GRAPH_PATH = os.path.join(ROOT, "fraud_graph.pt")
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "best_model.pt")
LGBM_PATH = os.path.join(ROOT, "outputs", "models", "lightgbm.pkl")
ENSEMBLE_PATH = os.path.join(ROOT, "outputs", "models", "ensemble.json")
METRICS_CSV = os.path.join(ROOT, "outputs", "results", "model_metrics.csv")


def scores(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "PR-AUC": average_precision_score(y_true, probs),
        "Threshold": float(threshold),
    }


def main() -> None:
    import lightgbm as lgb

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print("  GraphSAGE + LightGBM ensemble")
    print("=" * 74)

    data = torch.load(GRAPH_PATH, weights_only=False)
    split_mode = getattr(data, "split_mode", "unknown")
    print(f"  Split: {split_mode}   {data.num_nodes:,} nodes, "
          f"{data.x.shape[1]} features")

    tr = data.train_mask.numpy()
    va = data.val_mask.numpy()
    te = data.test_mask.numpy()
    X = data.x.numpy()
    y = data.y.numpy()

    # ── 1. LightGBM on the same features and the same split ────────────────
    print("\n[1/4] Training LightGBM ...")
    t0 = time.time()
    # `metric` must be set on the estimator, not only passed as eval_metric:
    # otherwise early stopping watches binary_logloss (the objective's default),
    # and with is_unbalance=True that gets worse immediately because the model
    # is calibrated for a reweighted distribution. The first attempt stopped at
    # iteration 1 and scored PR-AUC 0.265 where a plain 600-tree fit reaches
    # 0.520 — the "early stop" was reading the wrong signal entirely.
    clf = lgb.LGBMClassifier(
        n_estimators=1200, num_leaves=64, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
        metric="average_precision", first_metric_only=True,
    )
    clf.fit(
        X[tr], y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
    )
    if (clf.best_iteration_ or 0) < 20:
        raise RuntimeError(
            f"LightGBM stopped at iteration {clf.best_iteration_}; early "
            f"stopping is watching the wrong metric again."
        )
    print(f"      done in {time.time()-t0:.1f}s "
          f"(best iteration {clf.best_iteration_})")

    p_lgb_val = clf.predict_proba(X[va])[:, 1]
    p_lgb_test = clf.predict_proba(X[te])[:, 1]

    # ── 2. GNN probabilities for the same rows ─────────────────────────────
    print("\n[2/4] Scoring the GNN ...")
    t0 = time.time()
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = GraphSAGE(
        in_channels=ckpt["in_channels"],
        hidden1=cfg.get("hidden1", 256), hidden2=cfg.get("hidden2", 128),
        hidden3=cfg.get("hidden3", 64), mlp_hidden=cfg.get("mlp_hidden", 64),
        dropout1=cfg.get("dropout1", 0.2), dropout2=cfg.get("dropout2", 0.2),
        dropout3=cfg.get("dropout3", 0.2),
        num_layers=cfg.get("num_layers", 2), use_skip=cfg.get("use_skip", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_idx = torch.where(data.val_mask)[0]
    test_idx = torch.where(data.test_mask)[0]
    val_csr = csr_view(data, data.val_edge_mask)
    full_csr = csr_view(data, torch.ones(data.edge_index.shape[1], dtype=torch.bool))

    p_gnn_val = predict(model, data.x, data.y, val_csr, val_idx, device,
                        EVAL_NEIGHBORS, 4096)
    p_gnn_test = predict(model, data.x, data.y, full_csr, test_idx, device,
                         EVAL_NEIGHBORS, 4096)
    print(f"      done in {time.time()-t0:.1f}s")

    y_val, y_test = y[va], y[te]

    # ── 3. Tune the blend weight on VALIDATION ─────────────────────────────
    # w is the GNN's share. Selected on validation PR-AUC; touching test here
    # would leak the very thing the test split exists to measure.
    print("\n[3/4] Tuning the blend weight on validation ...")
    best_w, best_pr = 0.0, -1.0
    curve = []
    for w in np.arange(0.0, 1.001, 0.05):
        blend = w * p_gnn_val + (1 - w) * p_lgb_val
        pr = average_precision_score(y_val, blend)
        curve.append({"w": round(float(w), 2), "val_pr_auc": round(float(pr), 6)})
        if pr > best_pr:
            best_pr, best_w = pr, float(w)
    print(f"      best weight w={best_w:.2f} "
          f"(val PR-AUC {best_pr:.4f})   "
          f"GNN-only {average_precision_score(y_val, p_gnn_val):.4f} · "
          f"LGBM-only {average_precision_score(y_val, p_lgb_val):.4f}")

    p_ens_val = best_w * p_gnn_val + (1 - best_w) * p_lgb_val
    p_ens_test = best_w * p_gnn_test + (1 - best_w) * p_lgb_test

    # Thresholds: each model gets its own, all tuned on validation.
    thr_gnn = ckpt.get("threshold") or tune_threshold(y_val, p_gnn_val)[0]
    thr_lgb, _ = tune_threshold(y_val, p_lgb_val)
    thr_ens, _ = tune_threshold(y_val, p_ens_val)

    # ── 4. Report on TEST ──────────────────────────────────────────────────
    print("\n[4/4] Test-split results")
    rows = {
        "GraphSAGE (GNN)": scores(y_test, p_gnn_test, thr_gnn),
        "LightGBM": scores(y_test, p_lgb_test, thr_lgb),
        "Ensemble (GNN+LGBM)": scores(y_test, p_ens_test, thr_ens),
    }
    hdr = f"  {'Model':<22} {'thr':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC':>7} {'PR':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, m in rows.items():
        print(f"  {name:<22} {m['Threshold']:>5.2f} {m['Accuracy']:>7.4f} "
              f"{m['Precision']:>7.4f} {m['Recall']:>7.4f} {m['F1']:>7.4f} "
              f"{m['ROC-AUC']:>7.4f} {m['PR-AUC']:>7.4f}")

    gnn_f1, ens_f1 = rows["GraphSAGE (GNN)"]["F1"], rows["Ensemble (GNN+LGBM)"]["F1"]
    lgb_f1 = rows["LightGBM"]["F1"]
    best_single = max(gnn_f1, lgb_f1)
    verdict = ("the blend beats both single models"
               if ens_f1 > best_single else
               "the blend does NOT beat the best single model")
    print(f"\n  Verdict: {verdict} "
          f"(ensemble F1 {ens_f1:.4f} vs best single {best_single:.4f})")

    # ── Persist ────────────────────────────────────────────────────────────
    with open(LGBM_PATH, "wb") as f:
        pickle.dump(clf, f, protocol=4)

    payload = {
        "weight_gnn": best_w,
        "weight_lgbm": round(1 - best_w, 4),
        "thresholds": {"gnn": float(thr_gnn), "lightgbm": float(thr_lgb),
                       "ensemble": float(thr_ens)},
        "val_pr_auc": {
            "gnn": float(average_precision_score(y_val, p_gnn_val)),
            "lightgbm": float(average_precision_score(y_val, p_lgb_val)),
            "ensemble": float(best_pr),
        },
        "test_metrics": rows,
        "weight_curve": curve,
        "split_mode": split_mode,
        "n_features": int(data.x.shape[1]),
        "lgbm_best_iteration": int(clf.best_iteration_ or 0),
        "verdict": verdict,
    }
    with open(ENSEMBLE_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    # Merge into the comparison CSV the dashboard reads.
    existing = []
    if os.path.exists(METRICS_CSV):
        existing = pd.read_csv(METRICS_CSV).to_dict("records")
    keep = [r for r in existing if r.get("Model") not in rows]
    merged = keep + [{"Model": n, **m} for n, m in rows.items()]
    pd.DataFrame(merged).to_csv(METRICS_CSV, index=False)

    print(f"\n  LightGBM   -> {LGBM_PATH}")
    print(f"  Ensemble   -> {ENSEMBLE_PATH}")
    print(f"  Metrics    -> {METRICS_CSV}")
    print(f"  Total time : {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
