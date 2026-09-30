#!/usr/bin/env python3
"""
baselines.py
============
Traditional ML baselines on the exact same feature matrix (`data.x`) and the
exact same splits as the GNN.

What changed vs. the previous version
-------------------------------------
* Every baseline was scored at a fixed 0.5 cut-off while evaluate.py gave the
  GNN a threshold tuned on validation. That is not a like-for-like comparison.
  Each model now gets its threshold tuned on the validation split, the same way
  the GNN does.

* XGBoost and LightGBM were silently skipped because they were not installed,
  so the "comparison" was Logistic Regression vs Random Forest. Gradient
  boosting is the real bar on IEEE-CIS and it is now in the table.

* Results are written to model_metrics.csv, the file visualize.py plots, so the
  chart captioned "Traditional ML vs GNN" can actually contain the GNN.

Outputs
-------
    outputs/results/baseline_metrics.csv
    outputs/results/model_metrics.csv   (merged with the GraphSAGE row)
    outputs/plots/baseline_comparison.png
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(ROOT, "fraud_graph.pt")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
PLOT_DIR = os.path.join(ROOT, "outputs", "plots")
BASELINE_CSV = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
METRICS_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def tune_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.02, 0.98, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def score(name, y_val, p_val, y_test, p_test):
    """Tune the cut-off on validation, report on test — same rule as the GNN."""
    thr = tune_threshold(y_val, p_val)
    preds = (p_test >= thr).astype(int)
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, p_test),
        "PR-AUC": average_precision_score(y_test, p_test),
        "Threshold": thr,
    }
    print(f"  {name:<22} thr={thr:.2f}  F1={row['F1']:.4f}  "
          f"ROC-AUC={row['ROC-AUC']:.4f}  PR-AUC={row['PR-AUC']:.4f}")
    return row


def main() -> None:
    print("Loading graph for baseline comparison ...")
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"{GRAPH_PATH} not found. Run build_graph.py.")

    data = torch.load(GRAPH_PATH, weights_only=False)
    split_mode = getattr(data, "split_mode", "unknown")

    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_val = data.x[data.val_mask].numpy()
    y_val = data.y[data.val_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    n_pos = int(y_train.sum())
    spw = (len(y_train) - n_pos) / max(n_pos, 1)

    print(f"  Split      : {split_mode}")
    print(f"  Train      : {X_train.shape}   fraud {100*y_train.mean():.2f}%")
    print(f"  Val        : {X_val.shape}   fraud {100*y_val.mean():.2f}%")
    print(f"  Test       : {X_test.shape}   fraud {100*y_test.mean():.2f}%")
    print(f"  pos weight : {spw:.2f}")
    print("\nNote: thresholds are tuned on validation, reported on test.\n")

    results = []

    print("Training Logistic Regression ...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train, y_train)
    results.append(score("Logistic Regression", y_val,
                         lr.predict_proba(X_val)[:, 1], y_test,
                         lr.predict_proba(X_test)[:, 1]))
    print(f"    done in {time.time()-t0:.1f}s")

    print("Training Random Forest ...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=-1, random_state=42, min_samples_leaf=2)
    rf.fit(X_train, y_train)
    results.append(score("Random Forest", y_val,
                         rf.predict_proba(X_val)[:, 1], y_test,
                         rf.predict_proba(X_test)[:, 1]))
    print(f"    done in {time.time()-t0:.1f}s")

    if HAS_XGB:
        print("Training XGBoost ...")
        t0 = time.time()
        clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="aucpr",
            tree_method="hist", random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        results.append(score("XGBoost", y_val,
                             clf.predict_proba(X_val)[:, 1], y_test,
                             clf.predict_proba(X_test)[:, 1]))
        print(f"    done in {time.time()-t0:.1f}s")
    else:
        print("Skipping XGBoost (not installed).")

    if HAS_LGB:
        print("Training LightGBM ...")
        t0 = time.time()
        clf = lgb.LGBMClassifier(
            n_estimators=600, num_leaves=64, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        results.append(score("LightGBM", y_val,
                             clf.predict_proba(X_val)[:, 1], y_test,
                             clf.predict_proba(X_test)[:, 1]))
        print(f"    done in {time.time()-t0:.1f}s")
    else:
        print("Skipping LightGBM (not installed).")

    df_results = pd.DataFrame(results)
    df_results.to_csv(BASELINE_CSV, index=False)

    # Merge with the GraphSAGE row if evaluate.py has already run.
    merged = list(df_results.to_dict("records"))
    if os.path.exists(METRICS_CSV):
        prev = pd.read_csv(METRICS_CSV).to_dict("records")
        names = {r["Model"] for r in merged}
        merged.extend(r for r in prev if r.get("Model") not in names)
    pd.DataFrame(merged).to_csv(METRICS_CSV, index=False)

    print("\n" + "=" * 72)
    print(f"Baseline Metrics Summary  ({split_mode} split)")
    print("=" * 72)
    print(df_results.to_string(index=False))
    print("=" * 72)
    print(f"Saved -> {BASELINE_CSV}")
    print(f"Saved -> {METRICS_CSV}")

    ax = df_results.set_index("Model")[["ROC-AUC", "PR-AUC", "F1"]].plot(
        kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title(f"Baseline Models Comparison ({split_mode} split)")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "baseline_comparison.png"), dpi=150)
    plt.close()
    print(f"Plot  -> {os.path.join(PLOT_DIR, 'baseline_comparison.png')}")


if __name__ == "__main__":
    main()
