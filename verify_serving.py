#!/usr/bin/env python3
"""
verify_serving.py
=================
Measures the *serving* path — the one the web app actually calls — on real,
held-out transactions.

Why this replaces the old verification scripts
----------------------------------------------
None of the existing checks measured anything usable:

  * analyze_probabilities.py, prove_fix.py, verify_final.py
      score `tx[tx.isFraud == 1].head(100)` and `tx[tx.isFraud == 0].head(1900)`
      — rows from the *start* of the training data, which the model was fitted
      on. Any number they produce is memorisation.

  * evaluate_graph_attachment.py
      scores synthetic_corrected_2500.csv, which is fabricated noise: its V1 is
      0.9957 where the real V1 is exactly 1.0, and all 339 V columns are random
      floats uncorrelated with the label. The recorded AUC of 0.504 on that file
      is the expected result for random data and says nothing about the model.

  * every one of them hardcodes `d:\\projects\\GNN`, a path that does not exist
    on this machine, so they cannot run at all.

This script instead takes rows from the held-out test split, writes them out as
a CSV exactly as a user would upload, and runs server/ml/pipeline.run_full_pipeline
over them.

Usage
-----
    py verify_serving.py               # 4000 test rows, full + reduced columns
    py verify_serving.py --n 1000
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from preprocessing_pipeline import load_raw_data  # noqa: E402

# Columns a lightweight integration would realistically send: transaction-level
# only, no V/C/D engineered columns and no identity join.
REDUCED_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
]


def report(name: str, y_true: np.ndarray, probs: np.ndarray,
           threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    roc = roc_auc_score(y_true, probs) if 0 < y_true.sum() < len(y_true) else float("nan")
    pr = average_precision_score(y_true, probs)
    print(f"\n  {name}")
    print(f"    {'-'*64}")
    print(f"    ROC-AUC        : {roc:.4f}")
    print(f"    PR-AUC         : {pr:.4f}   (random = {y_true.mean():.4f})")
    print(f"    Precision      : {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"    Recall         : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"    F1             : {f1_score(y_true, preds, zero_division=0):.4f}")
    print(f"    Confusion      : TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"    Flagged        : {preds.sum():,}/{len(preds):,} "
          f"({100*preds.mean():.2f}%)   actual fraud "
          f"{y_true.sum():,} ({100*y_true.mean():.2f}%)")
    print(f"    Probability    : min {probs.min():.4f}  median "
          f"{np.median(probs):.4f}  max {probs.max():.4f}")
    return {"name": name, "roc_auc": roc, "pr_auc": pr, "flagged": int(preds.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000,
                    help="number of held-out test rows to score")
    args = ap.parse_args()

    print("=" * 72)
    print("  SERVING PATH VERIFICATION  (held-out test transactions)")
    print("=" * 72)

    from server.ml import loader
    from server.ml.pipeline import run_full_pipeline

    loader.load_all()
    threshold = loader.get_threshold()
    data = loader.get_data()
    print(f"\n  Threshold in use : {threshold:.2f}  (from best_model.pt)")
    print(f"  Split mode       : {getattr(data, 'split_mode', 'unknown')}")

    # ── Pull real rows from the held-out test split ────────────────────────
    print("\n  Loading raw transactions ...")
    df_all = load_raw_data(ROOT)
    test_pos = torch.where(data.test_mask)[0].numpy()

    rng = np.random.default_rng(42)
    pick = rng.choice(test_pos, size=min(args.n, len(test_pos)), replace=False)
    pick = np.sort(pick)
    df_test = df_all.iloc[pick].reset_index(drop=True)

    print(f"  Selected {len(df_test):,} held-out rows "
          f"({int(df_test['isFraud'].sum()):,} fraud = "
          f"{100*df_test['isFraud'].mean():.2f}%)")
    print("\n  NOTE: these transactions also exist as nodes in the training\n"
          "  graph, so an uploaded row can attach to its own historical node.\n"
          "  That inflates results slightly. It is still a far better estimate\n"
          "  than the previous scripts, which scored training rows outright.")

    y_true = df_test["isFraud"].values.astype(int)
    results = []

    # ── Scenario 1: full schema, exactly as the training data looks ────────
    print("\n" + "=" * 72)
    print("  SCENARIO 1 - full IEEE-CIS schema (all columns present)")
    print("=" * 72)
    t0 = time.time()
    res = run_full_pipeline(df_test.copy(), threshold=threshold)
    if not res.get("success"):
        print(f"  [FAIL] {res.get('errors')}")
        return
    probs = np.array([r["fraud_probability"] for r in res["results"]])
    print(f"\n  Pipeline time  : {res['processing_time_ms']:,}ms "
          f"(wall {time.time()-t0:.1f}s)")
    ga = res.get("graph_attachment", {})
    if ga:
        print(f"  Graph mode     : {res['graph_stats']['mode']}")
        print(f"  Cross-edges    : {ga['cross_edges']:,}   "
              f"intra-batch: {ga['intra_edges']:,}")
        print(f"  Rows with training neighbours : "
              f"{ga['nodes_with_training_neighbors']:,}/"
              f"{ga['n_uploaded_nodes']:,} "
              f"({100*ga['nodes_with_training_neighbors']/ga['n_uploaded_nodes']:.1f}%)")
    results.append(report("Full schema", y_true, probs, threshold))

    # ── Scenario 2: the thin payload a simple integration would send ───────
    print("\n" + "=" * 72)
    print("  SCENARIO 2 - reduced schema (transaction columns only)")
    print("=" * 72)
    df_reduced = df_test[[c for c in REDUCED_COLS if c in df_test.columns]].copy()
    print(f"  Sending {df_reduced.shape[1]} of {df_test.shape[1]} columns")
    res2 = run_full_pipeline(df_reduced, threshold=threshold)
    if res2.get("success"):
        probs2 = np.array([r["fraud_probability"] for r in res2["results"]])
        results.append(report("Reduced schema", y_true, probs2, threshold))
    else:
        print(f"  [FAIL] {res2.get('errors')}")

    # ── Scenario 3: the same rows scored as graph nodes (upper bound) ──────
    print("\n" + "=" * 72)
    print("  SCENARIO 3 - same rows scored in-graph (reference upper bound)")
    print("=" * 72)
    node_probs = loader.get_all_node_probs()[pick]
    results.append(report("In-graph nodes", y_true, node_probs, threshold))

    # ── Verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Scenario':<22} {'ROC-AUC':>9} {'PR-AUC':>9} {'Flagged':>9}")
    for r in results:
        print(f"  {r['name']:<22} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f} "
              f"{r['flagged']:>9,}")
    print("\n  A large gap between 'Full schema' and 'In-graph nodes' means the")
    print("  upload path and the training path disagree. A large gap between")
    print("  'Full schema' and 'Reduced schema' quantifies what a thin payload")
    print("  costs: every absent column is imputed to a constant and carries")
    print("  no signal.")
    print("=" * 72)


if __name__ == "__main__":
    main()
