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

import sys
import os
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.ml import loader
import server.ml.pipeline
from server.ml.pipeline import run_full_pipeline, preprocess_features
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, auc

print("\n--- 1 & 2. VERIFYING PREPROCESSING & FEATURE ORDERING ---")
loader.load_all()
preprocessor = loader.get_preprocessor()
model = loader.get_model()

print(f"Total features expected by preprocessor: {len(preprocessor.all_cols)}")
print(f"First 20 features from preprocessor:\n{preprocessor.all_cols[:20]}")

print("\n--- 4. VERIFYING MODEL COMPATIBILITY ---")
# Check the input dimension of the first linear layer in the SAGE model
input_dim = model.conv1.lin_l.weight.shape[1] if hasattr(model.conv1, 'lin_l') else "Check manually"
print(f"Model expects input feature dimension: {input_dim}")
print(f"Preprocessor outputs feature dimension: {len(preprocessor.all_cols)}")
assert input_dim == len(preprocessor.all_cols), "SHAPE MISMATCH DETECTED!"
print("Shape match confirmed!")

print("\n--- GENERATING DATASETS ---")
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), low_memory=False)
fraud_df = tx[tx["isFraud"] == 1].head(100)
legit_df = tx[tx["isFraud"] == 0].head(1900)
df_full = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

keep_cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", 
             "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", 
             "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
df_missing = df_full[[c for c in keep_cols if c in df_full.columns]]
print(f"Dataset with missing columns shape: {df_missing.shape}")
print(f"Dataset with full columns shape: {df_full.shape}")

print("\n--- 3. VERIFYING MISSING COLUMN HANDLING ---")
# Use the new preprocess_features on the missing dataframe
X_missing = preprocess_features(df_missing)

feature_df = df_missing.drop(columns=["TransactionID", "isFraud"], errors="ignore")
expected_cols = set(preprocessor.all_cols)
uploaded_cols = set(feature_df.columns)
missing_cols = expected_cols - uploaded_cols

print(f"Total missing columns identified: {len(missing_cols)}")
print(f"First 10 missing columns: {list(missing_cols)[:10]}")

# Verify they are exactly 0.0
all_zero = True
for i, col in enumerate(preprocessor.all_cols):
    if col in missing_cols:
        if not np.all(X_missing[:, i] == 0.0):
            all_zero = False
            print(f"Error: {col} is NOT 0.0!")
            break
print(f"All {len(missing_cols)} missing columns successfully forced to 0.0? {all_zero}")

# Verify present columns are untouched
X_raw = preprocessor.transform(feature_df)
present_untouched = True
for i, col in enumerate(preprocessor.all_cols):
    if col in uploaded_cols:
        if not np.allclose(X_missing[:, i], X_raw[:, i], equal_nan=True):
            present_untouched = False
            print(f"Error: Present column {col} was modified!")
            break
print(f"All {len(uploaded_cols)} present columns untouched? {present_untouched}")

print("\n--- 5. RUNNING REAL EVALUATION (MISSING COLUMNS DATASET) ---")
true_labels = df_missing["isFraud"].values
res_missing = run_full_pipeline(df_missing, threshold=0.51)
probs_missing = np.array([r["fraud_probability"] for r in res_missing["results"]])
preds_missing = np.array([r["is_fraud"] for r in res_missing["results"]])

tn, fp, fn, tp = confusion_matrix(true_labels, preds_missing).ravel()
acc = accuracy_score(true_labels, preds_missing)
prec = precision_score(true_labels, preds_missing, zero_division=0)
rec = recall_score(true_labels, preds_missing, zero_division=0)
f1 = f1_score(true_labels, preds_missing, zero_division=0)
roc = roc_auc_score(true_labels, probs_missing)
precision_curve, recall_curve, _ = precision_recall_curve(true_labels, probs_missing)
pr_auc = auc(recall_curve, precision_curve)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
bal_acc = balanced_accuracy_score(true_labels, preds_missing)
mcc = matthews_corrcoef(true_labels, preds_missing)

print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"MCC: {mcc:.4f}")

print("\n--- 7. CHECKING FOR REGRESSIONS (FULL COLUMNS DATASET) ---")
res_full = run_full_pipeline(df_full, threshold=0.51)
probs_full = np.array([r["fraud_probability"] for r in res_full["results"]])
preds_full = np.array([r["is_fraud"] for r in res_full["results"]])

tn_full, fp_full, fn_full, tp_full = confusion_matrix(true_labels, preds_full).ravel()
acc_full = accuracy_score(true_labels, preds_full)
prec_full = precision_score(true_labels, preds_full, zero_division=0)
rec_full = recall_score(true_labels, preds_full, zero_division=0)
f1_full = f1_score(true_labels, preds_full, zero_division=0)

print("Full features evaluation:")
print(f"Confusion Matrix: TP={tp_full}, FP={fp_full}, TN={tn_full}, FN={fn_full}")
print(f"Accuracy: {acc_full:.4f}")
print(f"Precision: {prec_full:.4f}")
print(f"Recall: {rec_full:.4f}")
print(f"F1-score: {f1_full:.4f}")

# Simulate BEFORE fix to print comparison table
print("\n--- 6. COMPARISON (BEFORE vs AFTER) ---")
original_preprocess = server.ml.pipeline.preprocess_features
def buggy_preprocess(df: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(df.drop(columns=["TransactionID", "isFraud"], errors="ignore"))

server.ml.pipeline.preprocess_features = buggy_preprocess
res_before = run_full_pipeline(df_missing, threshold=0.51)
preds_before = np.array([r["is_fraud"] for r in res_before["results"]])
server.ml.pipeline.preprocess_features = original_preprocess

tn_b, fp_b, fn_b, tp_b = confusion_matrix(true_labels, preds_before).ravel()
acc_b = accuracy_score(true_labels, preds_before)
prec_b = precision_score(true_labels, preds_before, zero_division=0)
rec_b = recall_score(true_labels, preds_before, zero_division=0)
f1_b = f1_score(true_labels, preds_before, zero_division=0)

print(f"{'Metric':<20} | {'Before Fix':<15} | {'After Fix':<15}")
print("-" * 55)
print(f"{'Predicted Frauds':<20} | {sum(preds_before):<15} | {sum(preds_missing):<15}")
print(f"{'False Positives':<20} | {fp_b:<15} | {fp:<15}")
print(f"{'False Negatives':<20} | {fn_b:<15} | {fn:<15}")
print(f"{'Accuracy':<20} | {acc_b:<15.4f} | {acc:<15.4f}")
print(f"{'Precision':<20} | {prec_b:<15.4f} | {prec:<15.4f}")
print(f"{'Recall':<20} | {rec_b:<15.4f} | {rec:<15.4f}")
print(f"{'F1-score':<20} | {f1_b:<15.4f} | {f1:<15.4f}")
