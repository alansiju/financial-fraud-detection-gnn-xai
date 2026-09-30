"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: scores training rows.

 Use verify_serving.py instead. It scores real rows from the held-out test
 split through the actual serving pipeline.

 Kept for reference only. Its hardcoded absolute paths have been
 repointed at the repo root so it at least runs, but do not trust its output.
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from server.ml import loader
from server.ml.pipeline import run_full_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, roc_curve, precision_recall_curve, auc

print("Loading ML stack...")
loader.load_all()

print("Generating synthetic 2000-row test dataset (5% fraud) from train data...")
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), low_memory=False)
fraud_df = tx[tx["isFraud"] == 1].head(100)
legit_df = tx[tx["isFraud"] == 0].head(1900)
df_full = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Simulate missing columns by only keeping 15 columns (mimicking the 417 missing columns issue)
keep_cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
df_subset = df_full[[c for c in keep_cols if c in df_full.columns]]
print(f"Subset shape: {df_subset.shape}")

res = run_full_pipeline(df_subset, threshold=0.51)

print("\n--- PIPELINE RESULTS ---")
print(f"Total Rows: {res['total_rows']}")
print(f"Fraud Detected: {res['fraud_count']} ({res['fraud_rate']}%)")
print(f"Missing Columns: {res['validation']['missing_columns']}")

probs = np.array([r["fraud_probability"] for r in res["results"]])
preds = np.array([r["is_fraud"] for r in res["results"]])
true_labels = df_subset["isFraud"].values

print("\n--- DISTRIBUTION ---")
print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Avg: {probs.mean():.4f}, Median: {np.median(probs):.4f}")
print("Fraud prob avg:", probs[true_labels == 1].mean())
print("Legit prob avg:", probs[true_labels == 0].mean())

print("\n--- GROUND TRUTH METRICS ---")
tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
print(f"Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Accuracy: {accuracy_score(true_labels, preds):.4f}")
print(f"Precision: {precision_score(true_labels, preds, zero_division=0):.4f}")
print(f"Recall: {recall_score(true_labels, preds, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(true_labels, preds, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(true_labels, probs):.4f}")

# Calculate PR-AUC
precision, recall, _ = precision_recall_curve(true_labels, probs)
print(f"PR-AUC: {auc(recall, precision):.4f}")

print(f"False Positive Rate: {fp / (fp + tn):.4f}")
print(f"False Negative Rate: {fn / (fn + tp):.4f}")
print(f"Matthews Correlation: {matthews_corrcoef(true_labels, preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(true_labels, preds):.4f}")

print("\n--- FEATURE IMPORTANCE ANALYSIS ---")
top_features_list = [r["top_features"] for r in res["results"]]
from collections import Counter
all_features = [f for sublist in top_features_list for f in sublist]
top_common = Counter(all_features).most_common(5)
print("Most common top features driving predictions:")
for f, count in top_common:
    print(f"  {f}: {count} times")
