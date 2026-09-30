"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: scores tx[isFraud==1].head(100) and tx[isFraud==0].head(1900) -- rows the model trained on.

 Use verify_serving.py instead. It scores real rows from the held-out test
 split through the actual serving pipeline.

 Kept for reference only. Its hardcoded absolute paths have been
 repointed at the repo root so it at least runs, but do not trust its output.
=============================================================================
"""

#!/usr/bin/env python3
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from server.ml import loader
from server.ml.pipeline import run_full_pipeline

print("=" * 75)
print("  MODEL OUTPUT INVESTIGATION — PROBABILITY ANALYSIS")
print("=" * 75)

print("\n[SETUP] Loading ML stack...")
loader.load_all()

# ── Build 2,000-row dataset ──
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), low_memory=False)
idt = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_identity.csv"), low_memory=False)

fraud_df = tx[tx["isFraud"] == 1].head(100)
legit_df = tx[tx["isFraud"] == 0].head(1900)
df_full = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# ── SCENARIO 1: Missing columns (15 of 432) ──
keep_cols = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"
]
df_missing = df_full[[c for c in keep_cols if c in df_full.columns]]

# ── SCENARIO 2: Full features (432 of 432) ──
df_full_features = df_full.merge(idt, on="TransactionID", how="left")
preprocessor = loader.get_preprocessor()
all_expected = set(preprocessor.all_cols) | {"TransactionID", "isFraud"}
available = [c for c in df_full_features.columns if c in all_expected]
df_full_features = df_full_features[available]

true_labels = df_full["isFraud"].values

for scenario_name, df_input in [("MISSING COLUMNS (15/432)", df_missing),
                                  ("FULL FEATURES (432/432)", df_full_features)]:
    print("\n" + "=" * 75)
    print(f"  SCENARIO: {scenario_name}")
    print("=" * 75)

    res = run_full_pipeline(df_input, threshold=0.51)
    probs = np.array([r["fraud_probability"] for r in res["results"]])

    fraud_probs = probs[true_labels == 1]
    legit_probs = probs[true_labels == 0]

    # ── Probability statistics ──
    print(f"\n{'':>35} {'Fraud (n=100)':>15} {'Legit (n=1900)':>15} {'All (n=2000)':>15}")
    print(f"  {'-'*80}")
    print(f"  {'Minimum probability':<33} {fraud_probs.min():>15.6f} {legit_probs.min():>15.6f} {probs.min():>15.6f}")
    print(f"  {'Maximum probability':<33} {fraud_probs.max():>15.6f} {legit_probs.max():>15.6f} {probs.max():>15.6f}")
    print(f"  {'Mean probability':<33} {fraud_probs.mean():>15.6f} {legit_probs.mean():>15.6f} {probs.mean():>15.6f}")
    print(f"  {'Median probability':<33} {np.median(fraud_probs):>15.6f} {np.median(legit_probs):>15.6f} {np.median(probs):>15.6f}")
    print(f"  {'Std deviation':<33} {fraud_probs.std():>15.6f} {legit_probs.std():>15.6f} {probs.std():>15.6f}")

    # ── Threshold analysis ──
    thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
    print(f"\n  {'Threshold':>10} | {'Fraud>Thr':>10} {'(of 100)':>10} | {'Legit>Thr':>10} {'(of 1900)':>10} | {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*95}")
    for thr in thresholds:
        fraud_above = (fraud_probs >= thr).sum()
        legit_above = (legit_probs >= thr).sum()
        total_above = fraud_above + legit_above
        precision = fraud_above / total_above if total_above > 0 else 0
        recall = fraud_above / 100
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {thr:>10.2f} | {fraud_above:>10} {f'({fraud_above}%)':>10} | {legit_above:>10} {f'({legit_above/19:.1f}%)':>10} | {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    # ── Histogram (text-based) ──
    bins = np.arange(0, 1.05, 0.05)
    fraud_hist, _ = np.histogram(fraud_probs, bins=bins)
    legit_hist, _ = np.histogram(legit_probs, bins=bins)

    print(f"\n  Probability Distribution Histogram:")
    print(f"  {'Bin Range':<15} | {'Fraud':>8} {'Bar':>22} | {'Legit':>8} {'Bar':>22}")
    print(f"  {'-'*82}")
    for i in range(len(bins) - 1):
        f_bar = '#' * min(fraud_hist[i], 20)
        l_bar = '#' * min(legit_hist[i] // 10, 20)  # scale down legit
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f})   | {fraud_hist[i]:>8} {f_bar:<22} | {legit_hist[i]:>8} {l_bar:<22}")

    # ── Separation analysis ──
    print(f"\n  SEPARATION ANALYSIS:")
    mean_diff = fraud_probs.mean() - legit_probs.mean()
    pooled_std = np.sqrt((fraud_probs.std()**2 + legit_probs.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # ROC-AUC
    from sklearn.metrics import roc_auc_score, roc_curve
    roc_auc = roc_auc_score(true_labels, probs)
    
    # Optimal threshold (Youden's J)
    fpr, tpr, roc_thresholds = roc_curve(true_labels, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    print(f"    Mean difference (fraud - legit):  {mean_diff:.6f}")
    print(f"    Cohen's d (effect size):          {cohens_d:.4f}", end="")
    if abs(cohens_d) < 0.2:
        print("  (negligible)")
    elif abs(cohens_d) < 0.5:
        print("  (small)")
    elif abs(cohens_d) < 0.8:
        print("  (medium)")
    else:
        print("  (large)")
    print(f"    ROC-AUC:                          {roc_auc:.4f}")
    print(f"    Optimal threshold (Youden's J):   {optimal_threshold:.4f}")
    
    # Metrics at optimal threshold
    preds_opt = (probs >= optimal_threshold).astype(int)
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    tn, fp, fn, tp = confusion_matrix(true_labels, preds_opt).ravel()
    print(f"\n    At optimal threshold {optimal_threshold:.4f}:")
    print(f"      TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"      Precision: {precision_score(true_labels, preds_opt, zero_division=0):.4f}")
    print(f"      Recall:    {recall_score(true_labels, preds_opt, zero_division=0):.4f}")
    print(f"      F1:        {f1_score(true_labels, preds_opt, zero_division=0):.4f}")

# ── Root cause analysis ──
print("\n" + "=" * 75)
print("  ROOT CAUSE ANALYSIS")
print("=" * 75)

print("""
The analysis above reveals whether the model can separate fraud from legit
in each scenario. Key factors affecting the missing-columns scenario:

1. FEATURE AVAILABILITY (15 of 432 features = 3.5%)
   - The model learned fraud patterns using 432 features
   - With only 15 available, 417 features are neutralized to zero
   - The model effectively receives almost no discriminative signal

2. GRAPH STRUCTURE DIFFERENCE  
   - Training graph: 590,540 nodes with 481,343 edges (rich connectivity)
   - Inference graph: 2,000 nodes with sparse edges (limited neighbors)
   - GraphSAGE aggregates neighbor features — fewer/no neighbors = weaker signal

3. DATA ORIGIN
   - The test rows come from the REAL IEEE-CIS training data (not synthetic)
   - These are actual fraud vs legitimate patterns the model was trained on
   - So data distribution is NOT the issue — missing features ARE
""")
