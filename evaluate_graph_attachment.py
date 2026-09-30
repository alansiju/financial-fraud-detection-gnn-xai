"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: scores synthetic_corrected_2500.csv, which is fabricated noise (V1=0.9957 where real V1 is exactly 1.0).

 Use verify_serving.py instead. It scores real rows from the held-out test
 split through the actual serving pipeline.

 Kept for reference only. Its hardcoded absolute paths have been
 repointed at the repo root so it at least runs, but do not trust its output.
=============================================================================
"""

#!/usr/bin/env python3
"""
evaluate_graph_attachment.py
============================
Comprehensive evaluation script for the Graph Attachment (Option A) implementation.

Tests:
1. Node mapping verification (TransactionID ↔ node index)
2. Transaction 2991340 (known isolated node)
3. Find transactions with real training neighbors
4. Full 2,500-row evaluation with metrics
5. Before/after probability comparison

Usage:
    py evaluate_graph_attachment.py
"""

import os
import os, sys, time, pickle, json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — VERIFY NODE MAPPING
# ═══════════════════════════════════════════════════════════════════════════

def verify_node_mapping():
    """Verify TransactionID ↔ node index mapping."""
    print("=" * 60)
    print("STEP 3: VERIFY NODE MAPPING")
    print("=" * 60)
    
    # Load lookup
    lookup_path = os.path.join(ROOT, "outputs", "models", "edge_attr_lookup.pkl")
    with open(lookup_path, "rb") as f:
        lookup = pickle.load(f)
    
    tx_ids = lookup["transaction_ids"]
    n_nodes = lookup["n_nodes"]
    
    print(f"Lookup: {n_nodes:,} nodes")
    print(f"TransactionID range: [{tx_ids.min()}, {tx_ids.max()}]")
    
    # Load fraud_graph.pt to get node count
    graph_path = os.path.join(ROOT, "fraud_graph.pt")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    print(f"Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    
    assert n_nodes == data.num_nodes, \
        f"Mismatch: lookup has {n_nodes}, graph has {data.num_nodes}"
    print(f"[OK] Node counts match: {n_nodes:,}")
    
    # Verify known mapping: TransactionID 2991340 → node 4340
    idx_2991340 = np.where(tx_ids == 2991340)[0]
    assert len(idx_2991340) == 1, f"TransactionID 2991340 not found"
    assert idx_2991340[0] == 4340, f"Expected node 4340, got {idx_2991340[0]}"
    print(f"[OK] TransactionID 2991340 → node index {idx_2991340[0]}")
    
    # Verify more transactions
    test_ids = [tx_ids[0], tx_ids[100], tx_ids[1000], tx_ids[10000]]
    for tid in test_ids:
        idx = np.where(tx_ids == tid)[0][0]
        print(f"  TransactionID {tid} → node index {idx}")
    
    # Verify node 4340 has 0 neighbors in original graph
    ei = data.edge_index
    neighbors_4340 = torch.sum((ei[0] == 4340) | (ei[1] == 4340)).item()
    print(f"[CHECK] Node 4340 neighbors in training graph: {neighbors_4340}")
    
    print("[OK] Node mapping verified.\n")
    return data, tx_ids


# ═══════════════════════════════════════════════════════════════════════════
#  STEPS 8-10 — TEST WITH EVALUATION DATA
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline_test(data, tx_ids):
    """Run the full pipeline on the 2,500-row evaluation file."""
    print("=" * 60)
    print("STEPS 8-10: EVALUATION")
    print("=" * 60)
    
    eval_path = os.path.join(ROOT, "synthetic_corrected_2500.csv")
    if not os.path.exists(eval_path):
        print(f"[!] Evaluation file not found: {eval_path}")
        return
    
    df = pd.read_csv(eval_path, low_memory=False)
    print(f"Evaluation data: {len(df):,} rows, {df.shape[1]} cols")
    print(f"isFraud distribution: {df['isFraud'].value_counts().to_dict()}")
    
    # Load all necessary components
    from server.ml import loader
    loader.load_all()
    
    from server.ml.pipeline import (
        preprocess_features, build_transaction_graph, build_cross_edges,
        build_intra_batch_edges, run_inference, run_attached_inference,
        run_full_pipeline, EDGE_ATTRIBUTES
    )
    
    inverted_index = loader.get_edge_attr_lookup()
    n_training = data.num_nodes
    
    # ── Find transactions with training neighbors ──────────────────────────
    print("\n── Finding transactions with training neighbors ──")
    df_reset = df.reset_index(drop=True)
    
    neighbor_map = {}
    for attr, weight in EDGE_ATTRIBUTES.items():
        if attr not in df_reset.columns or attr not in inverted_index:
            continue
        attr_lookup = inverted_index[attr]
        for i in range(len(df_reset)):
            val = df_reset.at[i, attr]
            if pd.isna(val):
                continue
            lookup_key = val
            if isinstance(val, float) and val == int(val):
                lookup_key = int(val)
            matches = attr_lookup.get(lookup_key, [])
            if matches and len(matches) <= 30:
                if i not in neighbor_map:
                    neighbor_map[i] = set()
                neighbor_map[i].update(matches)
    
    with_neighbors = [(i, len(n)) for i, n in neighbor_map.items() if len(n) > 0]
    with_neighbors.sort(key=lambda x: -x[1])
    
    print(f"Transactions with training neighbors: {len(with_neighbors)}/{len(df)}")
    print(f"Transactions without: {len(df) - len(with_neighbors)}")
    
    # ── STEP 8: Test TransactionID 2991340 ────────────────────────────────
    print("\n── STEP 8: Test TransactionID 2991340 ──")
    if 2991340 in df["TransactionID"].values:
        idx_in_eval = df.index[df["TransactionID"] == 2991340].tolist()
        if idx_in_eval:
            row_idx = idx_in_eval[0]
            label = df.at[row_idx, "isFraud"]
            n_neighbors = len(neighbor_map.get(row_idx, set()))
            print(f"  Row index in eval: {row_idx}")
            print(f"  isFraud: {label}")
            print(f"  Training neighbors: {n_neighbors}")
    else:
        print("  TransactionID 2991340 NOT in evaluation set")
    
    # ── Run OLD mini-graph inference ──────────────────────────────────────
    print("\n── Running OLD mini-graph inference ──")
    t_old = time.time()
    features = preprocess_features(df_reset)
    ei_old, ew_old = build_transaction_graph(df_reset, len(df_reset))
    probs_old = run_inference(features, ei_old)
    old_time = time.time() - t_old
    print(f"  Old inference time: {old_time:.1f}s")
    print(f"  Old edges: {ei_old.shape[1] // 2}")
    print(f"  Old probability stats: mean={probs_old.mean():.4f}, "
          f"std={probs_old.std():.4f}, min={probs_old.min():.4f}, max={probs_old.max():.4f}")
    
    # ── Run NEW attached-graph inference ──────────────────────────────────
    print("\n── Running NEW attached-graph inference ──")
    t_new = time.time()
    probs_new, attach_info = run_attached_inference(features, df_reset)
    new_time = time.time() - t_new
    print(f"  New inference time: {new_time:.1f}s")
    print(f"  Cross edges: {attach_info['cross_edges']:,}")
    print(f"  Intra edges: {attach_info['intra_edges']:,}")
    print(f"  Training edges: {attach_info['training_edges']:,}")
    print(f"  Total edges: {attach_info['total_edges']:,}")
    print(f"  Nodes w/ neighbors: {attach_info['nodes_with_training_neighbors']}")
    print(f"  New probability stats: mean={probs_new.mean():.4f}, "
          f"std={probs_new.std():.4f}, min={probs_new.min():.4f}, max={probs_new.max():.4f}")
    
    # ── STEP 9: Compare specific transactions with neighbors ──────────────
    print("\n── STEP 9: Transactions with real neighbors ──")
    print(f"{'Row':<6} {'TxnID':<12} {'Neighbors':<10} {'OldProb':<10} {'NewProb':<10} {'Diff':<10} {'Label':<6}")
    print("-" * 64)
    
    count = 0
    for row_i, n_count in with_neighbors[:10]:  # Show top 10
        txn_id = df_reset.at[row_i, "TransactionID"] if "TransactionID" in df_reset.columns else f"ROW_{row_i}"
        label = int(df_reset.at[row_i, "isFraud"]) if "isFraud" in df_reset.columns else "?"
        old_p = probs_old[row_i]
        new_p = probs_new[row_i]
        diff = new_p - old_p
        print(f"{row_i:<6} {str(txn_id):<12} {n_count:<10} {old_p:<10.4f} {new_p:<10.4f} {diff:+.4f}    {label}")
        count += 1
    
    # Also show some transactions WITHOUT neighbors for comparison
    print(f"\n── Transactions WITHOUT training neighbors (control) ──")
    no_neighbor_rows = [i for i in range(min(20, len(df))) if i not in neighbor_map]
    for row_i in no_neighbor_rows[:5]:
        txn_id = df_reset.at[row_i, "TransactionID"] if "TransactionID" in df_reset.columns else f"ROW_{row_i}"
        label = int(df_reset.at[row_i, "isFraud"]) if "isFraud" in df_reset.columns else "?"
        old_p = probs_old[row_i]
        new_p = probs_new[row_i]
        diff = new_p - old_p
        print(f"{row_i:<6} {str(txn_id):<12} {0:<10} {old_p:<10.4f} {new_p:<10.4f} {diff:+.4f}    {label}")
    
    # ── STEP 10: Full evaluation metrics ──────────────────────────────────
    print("\n── STEP 10: Full Evaluation ──")
    true_labels = df["isFraud"].values.astype(int)
    threshold = 0.5
    
    for name, probs in [("OLD (mini-graph)", probs_old), ("NEW (attached)", probs_new)]:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        
        acc  = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec  = recall_score(true_labels, preds, zero_division=0)
        f1   = f1_score(true_labels, preds, zero_division=0)
        auc  = roc_auc_score(true_labels, probs)
        pr_auc = average_precision_score(true_labels, probs)
        
        print(f"\n  {name}:")
        print(f"    TP={tp:>5}  FP={fp:>5}")
        print(f"    FN={fn:>5}  TN={tn:>5}")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")
        print(f"    PR-AUC:    {pr_auc:.4f}")
    
    # Probability distribution comparison
    print(f"\n── Probability Distribution Comparison ──")
    print(f"{'Metric':<20} {'Old':<12} {'New':<12} {'Diff':<12}")
    print("-" * 56)
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        old_q = np.quantile(probs_old, q)
        new_q = np.quantile(probs_new, q)
        print(f"  P{int(q*100):>2}            {old_q:<12.6f} {new_q:<12.6f} {new_q-old_q:+.6f}")
    
    # Focus on fraud transactions
    fraud_mask = true_labels == 1
    if fraud_mask.sum() > 0:
        print(f"\n── Fraud Transactions Only (n={fraud_mask.sum()}) ──")
        print(f"  Old mean prob: {probs_old[fraud_mask].mean():.4f}")
        print(f"  New mean prob: {probs_new[fraud_mask].mean():.4f}")
        print(f"  Old median prob: {np.median(probs_old[fraud_mask]):.4f}")
        print(f"  New median prob: {np.median(probs_new[fraud_mask]):.4f}")
    
    legit_mask = true_labels == 0
    if legit_mask.sum() > 0:
        print(f"\n── Legitimate Transactions Only (n={legit_mask.sum()}) ──")
        print(f"  Old mean prob: {probs_old[legit_mask].mean():.4f}")
        print(f"  New mean prob: {probs_new[legit_mask].mean():.4f}")
    
    # Timing summary
    print(f"\n── Timing ──")
    print(f"  Old (mini-graph): {old_time:.1f}s")
    print(f"  New (attached):   {new_time:.1f}s")
    print(f"  Overhead:         {new_time - old_time:.1f}s")
    
    # Save results
    results = {
        "old_probs": probs_old.tolist(),
        "new_probs": probs_new.tolist(),
        "true_labels": true_labels.tolist(),
        "old_time_s": old_time,
        "new_time_s": new_time,
        "attachment_info": {
            k: v for k, v in attach_info.items()
            if k != "neighbor_counts"  # don't serialize large dict
        },
    }
    results_path = os.path.join(ROOT, "backups", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    
    data, tx_ids = verify_node_mapping()
    run_pipeline_test(data, tx_ids)
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL EVALUATION TIME: {time.time() - t_start:.1f}s")
    print(f"{'=' * 60}")
