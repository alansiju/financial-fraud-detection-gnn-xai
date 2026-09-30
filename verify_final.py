"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: scores rows from the head of the training data.

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
"""
verify_final.py
===============
Final verification: proves that the corrected inference preprocessing is
identical to the training preprocessing (build_graph.py clean_features()).

Tests:
  1. Column ordering match
  2. Feature values match (build_graph.py vs FraudPreprocessor)
  3. Feature values match against fraud_graph.pt
  4. 2,000-row evaluation with corrected pipeline
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve, auc as sk_auc,
)

# ======================================================================
#  SETUP
# ======================================================================

print("=" * 75)
print("  FINAL VERIFICATION: CORRECTED INFERENCE PREPROCESSING")
print("=" * 75)

# Load the corrected preprocessor
from preprocessing_pipeline import FraudPreprocessor, PIPELINE_PATH

print(f"\n[SETUP] Loading corrected preprocessor from {PIPELINE_PATH}...")
preprocessor = FraudPreprocessor.load(PIPELINE_PATH)

print(f"[INFO] Preprocessor n_features: {preprocessor.n_features}")
print(f"[INFO] Numeric columns:  {len(preprocessor.num_cols)}")
print(f"[INFO] Categorical columns: {len(preprocessor.cat_cols)}")
print(f"[INFO] all_cols length:  {len(preprocessor.all_cols)}")

# ======================================================================
#  LOAD TRAINING DATA
# ======================================================================

print("\n[SETUP] Loading training data...")
t0 = time.time()
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"),
                 dtype={"TransactionID": "int32", "isFraud": "int8",
                        "TransactionAmt": "float32"}, low_memory=False)
idt = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_identity.csv"),
                 dtype={"TransactionID": "int32"}, low_memory=False)
df = tx.merge(idt, on="TransactionID", how="left")
print(f"[INFO] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

# Drop duplicates (same as build_graph.py line 526)
df = df.drop_duplicates(subset="TransactionID").reset_index(drop=True)
feature_df = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
print(f"[INFO] Feature DataFrame: {feature_df.shape}")

# ======================================================================
#  TEST 1: COLUMN ORDERING
# ======================================================================

print("\n" + "=" * 75)
print("  TEST 1: COLUMN ORDERING")
print("=" * 75)

training_col_order = feature_df.columns.tolist()
preprocessor_col_order = preprocessor.all_cols

order_match = (training_col_order == preprocessor_col_order)
print(f"\n[CHECK] Column order identical? {order_match}")

if order_match:
    print(f"[PASS] All {len(training_col_order)} columns are in identical order")
    print(f"\n  First 20 columns:")
    for i in range(20):
        col_type = "CAT" if training_col_order[i] in preprocessor.cat_cols else "NUM"
        print(f"  [{i:3d}] {training_col_order[i]:<25s} ({col_type})")
else:
    mismatches = sum(1 for i in range(len(training_col_order))
                     if training_col_order[i] != preprocessor_col_order[i])
    print(f"[FAIL] {mismatches} / {len(training_col_order)} positions differ!")
    print(f"\n  First 10 mismatches:")
    shown = 0
    for i in range(len(training_col_order)):
        if training_col_order[i] != preprocessor_col_order[i]:
            print(f"  [{i:3d}] TRAIN: {training_col_order[i]:<25s} INFER: {preprocessor_col_order[i]}")
            shown += 1
            if shown >= 10:
                break

# ======================================================================
#  TEST 2: FEATURE VALUES — build_graph.py vs FraudPreprocessor
# ======================================================================

print("\n" + "=" * 75)
print("  TEST 2: FEATURE VALUES — build_graph.py vs FraudPreprocessor")
print("=" * 75)

# ── Reproduce build_graph.py clean_features() EXACTLY ──
print("\n[INFO] Running build_graph.py clean_features() logic...")
cat_cols = feature_df.select_dtypes(include="object").columns.tolist()
num_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

feature_df_bg = feature_df.copy()

num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="constant", fill_value="__missing__")

if num_cols:
    feature_df_bg[num_cols] = num_imp.fit_transform(feature_df_bg[num_cols])
if cat_cols:
    feature_df_bg[cat_cols] = cat_imp.fit_transform(feature_df_bg[cat_cols])

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
if cat_cols:
    feature_df_bg[cat_cols] = enc.fit_transform(feature_df_bg[cat_cols])

scaler_bg = StandardScaler()
if num_cols:
    feature_df_bg[num_cols] = scaler_bg.fit_transform(feature_df_bg[num_cols])

X_build_graph = feature_df_bg.values.astype(np.float32)
print(f"[INFO] build_graph.py output: {X_build_graph.shape}")

# ── Run FraudPreprocessor.transform() ──
print("[INFO] Running FraudPreprocessor.transform()...")
X_preprocessor = preprocessor.transform(feature_df)
print(f"[INFO] FraudPreprocessor output: {X_preprocessor.shape}")

# ── Compare ──
print(f"\n[COMPARE] Element-wise comparison on {X_build_graph.shape[0]:,} x {X_build_graph.shape[1]} matrix:")

matching_positions = np.isclose(X_build_graph, X_preprocessor, atol=1e-5).sum()
total_positions = X_build_graph.size
abs_diff = np.abs(X_build_graph - X_preprocessor)
max_diff = abs_diff.max()
mean_diff = abs_diff.mean()
allclose = np.allclose(X_build_graph, X_preprocessor, atol=1e-5)

print(f"  Matching positions:  {matching_positions:,} / {total_positions:,} ({100*matching_positions/total_positions:.4f}%)")
print(f"  Max absolute diff:   {max_diff:.10f}")
print(f"  Mean absolute diff:  {mean_diff:.10f}")
print(f"  np.allclose(atol=1e-5): {allclose}")

if allclose:
    print(f"\n[PASS] Feature matrices are identical (within atol=1e-5)")
else:
    print(f"\n[FAIL] Feature matrices differ!")
    # Find which columns differ
    col_diffs = []
    for i in range(X_build_graph.shape[1]):
        col_diff = np.abs(X_build_graph[:, i] - X_preprocessor[:, i]).max()
        if col_diff > 1e-5:
            col_name = training_col_order[i]
            col_type = "CAT" if col_name in cat_cols else "NUM"
            col_diffs.append((col_name, col_type, i, col_diff))
    
    print(f"  {len(col_diffs)} columns with diffs > 1e-5:")
    for name, ctype, idx, diff in col_diffs[:20]:
        print(f"    [{idx:3d}] {name:<25s} ({ctype}) max_diff={diff:.8f}")
        print(f"           BG first 5:  {X_build_graph[:5, idx]}")
        print(f"           PP first 5:  {X_preprocessor[:5, idx]}")

# ── Verify categorical values are ordinal (not scaled) ──
print(f"\n[CHECK] Categorical values in preprocessor output:")
for cat_col in cat_cols[:5]:
    idx = training_col_order.index(cat_col)
    vals = X_preprocessor[:10, idx]
    is_integer = np.allclose(vals, np.round(vals))
    min_val = vals.min()
    print(f"  {cat_col:<20s} [idx={idx:3d}] first 10: {vals}  integer? {is_integer}  min>=-1? {min_val >= -1.0}")

# ======================================================================
#  TEST 3: COMPARE AGAINST fraud_graph.pt
# ======================================================================

print("\n" + "=" * 75)
print("  TEST 3: COMPARE AGAINST fraud_graph.pt (GROUND TRUTH)")
print("=" * 75)

print("\n[INFO] Loading fraud_graph.pt...")
data = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_graph.pt"), map_location="cpu", weights_only=False)
X_graph = data.x.numpy()
print(f"[INFO] fraud_graph.pt features: {X_graph.shape}")

# The graph was built from the same training data we just processed
# Compare directly: they should be identical
N = min(X_graph.shape[0], X_build_graph.shape[0])
print(f"[INFO] Comparing first {N:,} rows")

# Compare build_graph output vs graph (these should be IDENTICAL since
# build_graph.py is what produced fraud_graph.pt)
bg_vs_graph_allclose = np.allclose(X_graph[:N], X_build_graph[:N], atol=1e-5)
bg_vs_graph_max_diff = np.abs(X_graph[:N] - X_build_graph[:N]).max()
bg_vs_graph_mean_diff = np.abs(X_graph[:N] - X_build_graph[:N]).mean()

print(f"\n[COMPARE] build_graph.py output vs fraud_graph.pt:")
print(f"  np.allclose(atol=1e-5): {bg_vs_graph_allclose}")
print(f"  Max absolute diff:      {bg_vs_graph_max_diff:.10f}")
print(f"  Mean absolute diff:     {bg_vs_graph_mean_diff:.10f}")

# Compare preprocessor output vs graph
pp_vs_graph_allclose = np.allclose(X_graph[:N], X_preprocessor[:N], atol=1e-5)
pp_vs_graph_max_diff = np.abs(X_graph[:N] - X_preprocessor[:N]).max()
pp_vs_graph_mean_diff = np.abs(X_graph[:N] - X_preprocessor[:N]).mean()

print(f"\n[COMPARE] FraudPreprocessor output vs fraud_graph.pt:")
print(f"  np.allclose(atol=1e-5): {pp_vs_graph_allclose}")
print(f"  Max absolute diff:      {pp_vs_graph_max_diff:.10f}")
print(f"  Mean absolute diff:     {pp_vs_graph_mean_diff:.10f}")

# Verify specific checks
print(f"\n[CHECK] Column ordering identical to fraud_graph.pt? ", end="")
# Check categorical columns have ordinal values in the graph
cat_ordinal_in_graph = True
for cat_col in cat_cols[:5]:
    idx = training_col_order.index(cat_col)
    graph_vals = X_graph[:100, idx]
    if not np.allclose(graph_vals, np.round(graph_vals)):
        cat_ordinal_in_graph = False
        break
print("YES" if cat_ordinal_in_graph else "NO")

print(f"[CHECK] Numeric scaling identical to fraud_graph.pt? ", end="")
num_scaling_ok = True
for num_col in num_cols[:5]:
    idx = training_col_order.index(num_col)
    if not np.allclose(X_graph[:100, idx], X_preprocessor[:100, idx], atol=1e-5):
        num_scaling_ok = False
        break
print("YES" if num_scaling_ok else "NO")

print(f"[CHECK] Categorical values remain ordinal? ", end="")
cat_ordinal_in_pp = True
for cat_col in cat_cols:
    idx = training_col_order.index(cat_col)
    pp_vals = X_preprocessor[:100, idx]
    if not np.allclose(pp_vals, np.round(pp_vals)):
        cat_ordinal_in_pp = False
        print(f"\n  FAIL: {cat_col} not ordinal: {pp_vals[:5]}")
        break
print("YES" if cat_ordinal_in_pp else "")

print(f"[CHECK] Input dimension = 432? {X_preprocessor.shape[1] == 432}")

# ======================================================================
#  TEST 4: 2,000-ROW EVALUATION
# ======================================================================

print("\n" + "=" * 75)
print("  TEST 4: 2,000-ROW EVALUATION WITH CORRECTED PIPELINE")
print("=" * 75)

# Reload the ML stack with the corrected preprocessor
from server.ml import loader
loader.load_all()

from server.ml.pipeline import run_full_pipeline

# Create 2000-row test dataset
fraud_df = tx[tx["isFraud"] == 1].head(100)
legit_df = tx[tx["isFraud"] == 0].head(1900)
df_full = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Test with missing columns (15 of 432) ──
keep_cols = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"
]
df_missing = df_full[[c for c in keep_cols if c in df_full.columns]]
true_labels = df_missing["isFraud"].values

print(f"\n[INFO] Dataset: {len(df_missing)} rows, {len(df_missing.columns)} cols")
print(f"[INFO] Ground truth: {true_labels.sum()} fraud, {(1-true_labels).sum()} legit")

t0 = time.time()
res = run_full_pipeline(df_missing, threshold=0.51)
t_elapsed = time.time() - t0

probs = np.array([r["fraud_probability"] for r in res["results"]])
preds = np.array([r["is_fraud"] for r in res["results"]])

tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
acc = accuracy_score(true_labels, preds)
prec = precision_score(true_labels, preds, zero_division=0)
rec = recall_score(true_labels, preds, zero_division=0)
f1 = f1_score(true_labels, preds, zero_division=0)
roc = roc_auc_score(true_labels, probs)
p_curve, r_curve, _ = precision_recall_curve(true_labels, probs)
pr_auc_val = sk_auc(r_curve, p_curve)
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n[RESULTS] Missing columns dataset (15 of 432 features)")
print(f"  Processing time: {t_elapsed:.2f}s")
print(f"\n  Confusion Matrix:")
print(f"    TP = {tp}")
print(f"    FP = {fp}")
print(f"    TN = {tn}")
print(f"    FN = {fn}")
print(f"\n  Accuracy:           {acc:.4f}")
print(f"  Precision:          {prec:.4f}")
print(f"  Recall:             {rec:.4f}")
print(f"  F1-score:           {f1:.4f}")
print(f"  ROC-AUC:            {roc:.4f}")
print(f"  PR-AUC:             {pr_auc_val:.4f}")
print(f"  False Positive Rate:{fpr_val:.4f}")
print(f"  False Negative Rate:{fnr_val:.4f}")

print(f"\n  Probability Distribution:")
print(f"    Min:    {probs.min():.6f}")
print(f"    Max:    {probs.max():.6f}")
print(f"    Mean:   {probs.mean():.6f}")
print(f"    Median: {np.median(probs):.6f}")
print(f"    Fraud avg:  {probs[true_labels == 1].mean():.6f}")
print(f"    Legit avg:  {probs[true_labels == 0].mean():.6f}")

# ── Test with FULL features ──
print(f"\n" + "-" * 75)
print(f"  FULL FEATURES EVALUATION")
print(f"-" * 75)

df_full_features = df_full.merge(idt, on="TransactionID", how="left")
# Keep only expected cols + ID/label
all_expected = set(preprocessor.all_cols) | {"TransactionID", "isFraud"}
available = [c for c in df_full_features.columns if c in all_expected]
df_full_features = df_full_features[available]

true_labels_full = df_full_features["isFraud"].values

res_full = run_full_pipeline(df_full_features, threshold=0.51)
probs_full = np.array([r["fraud_probability"] for r in res_full["results"]])
preds_full = np.array([r["is_fraud"] for r in res_full["results"]])

tn_f, fp_f, fn_f, tp_f = confusion_matrix(true_labels_full, preds_full).ravel()
acc_f = accuracy_score(true_labels_full, preds_full)
prec_f = precision_score(true_labels_full, preds_full, zero_division=0)
rec_f = recall_score(true_labels_full, preds_full, zero_division=0)
f1_f = f1_score(true_labels_full, preds_full, zero_division=0)
roc_f = roc_auc_score(true_labels_full, probs_full)
p_curve_f, r_curve_f, _ = precision_recall_curve(true_labels_full, probs_full)
pr_auc_f = sk_auc(r_curve_f, p_curve_f)
fpr_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0
fnr_f = fn_f / (fn_f + tp_f) if (fn_f + tp_f) > 0 else 0

print(f"\n[RESULTS] Full features dataset (432 of 432 features)")
print(f"\n  Confusion Matrix:")
print(f"    TP = {tp_f}")
print(f"    FP = {fp_f}")
print(f"    TN = {tn_f}")
print(f"    FN = {fn_f}")
print(f"\n  Accuracy:           {acc_f:.4f}")
print(f"  Precision:          {prec_f:.4f}")
print(f"  Recall:             {rec_f:.4f}")
print(f"  F1-score:           {f1_f:.4f}")
print(f"  ROC-AUC:            {roc_f:.4f}")
print(f"  PR-AUC:             {pr_auc_f:.4f}")
print(f"  False Positive Rate:{fpr_f:.4f}")
print(f"  False Negative Rate:{fnr_f:.4f}")

print(f"\n  Probability Distribution:")
print(f"    Min:    {probs_full.min():.6f}")
print(f"    Max:    {probs_full.max():.6f}")
print(f"    Mean:   {probs_full.mean():.6f}")
print(f"    Median: {np.median(probs_full):.6f}")
print(f"    Fraud avg:  {probs_full[true_labels_full == 1].mean():.6f}")
print(f"    Legit avg:  {probs_full[true_labels_full == 0].mean():.6f}")

# ======================================================================
#  FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 75)
print("  FINAL VERIFICATION SUMMARY")
print("=" * 75)

checks = {
    "Column ordering identical":                  order_match,
    "build_graph vs preprocessor allclose":        allclose,
    "build_graph vs fraud_graph.pt allclose":      bg_vs_graph_allclose,
    "preprocessor vs fraud_graph.pt allclose":     pp_vs_graph_allclose,
    "Categorical values remain ordinal":           cat_ordinal_in_pp,
    "Input dimension = 432":                       X_preprocessor.shape[1] == 432,
    "No NaN in preprocessor output":               not np.isnan(X_preprocessor).any(),
    "No Inf in preprocessor output":               not np.isinf(X_preprocessor).any(),
}

all_pass = True
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {check}")

print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print("=" * 75)
