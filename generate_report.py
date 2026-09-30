import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.ml import loader
import server.ml.pipeline
from server.ml.pipeline import run_full_pipeline, preprocess_features
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, auc

print("Loading ML stack...")
loader.load_all()

# Generate dataset
print("Generating synthetic 2000-row test dataset (5% fraud) from train data...")
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), low_memory=False)
fraud_df = tx[tx["isFraud"] == 1].head(100)
legit_df = tx[tx["isFraud"] == 0].head(1900)
df_full = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

keep_cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", 
             "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", 
             "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
df_subset = df_full[[c for c in keep_cols if c in df_full.columns]]
print(f"Dataset shape: {df_subset.shape}")
true_labels = df_subset["isFraud"].values

def evaluate(probs, preds):
    tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)
    roc = roc_auc_score(true_labels, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, probs)
    pr_auc = auc(recall_curve, precision_curve)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    bal_acc = balanced_accuracy_score(true_labels, preds)
    mcc = matthews_corrcoef(true_labels, preds)
    
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "ROC-AUC": roc, "PR-AUC": pr_auc, "FPR": fpr, "FNR": fnr,
        "Balanced Accuracy": bal_acc, "MCC": mcc
    }

# --- BEFORE FIX ---
print("\nRunning inference WITHOUT fix (simulating the bug)...")
original_preprocess = server.ml.pipeline.preprocess_features

def buggy_preprocess(df: pd.DataFrame) -> np.ndarray:
    preprocessor = loader.get_preprocessor()
    feature_df = df.drop(columns=["TransactionID", "isFraud"], errors="ignore")
    return preprocessor.transform(feature_df)

server.ml.pipeline.preprocess_features = buggy_preprocess
res_before = run_full_pipeline(df_subset, threshold=0.51)
probs_before = np.array([r["fraud_probability"] for r in res_before["results"]])
preds_before = np.array([r["is_fraud"] for r in res_before["results"]])
eval_before = evaluate(probs_before, preds_before)

# --- AFTER FIX ---
print("\nRunning inference WITH fix...")
server.ml.pipeline.preprocess_features = original_preprocess
res_after = run_full_pipeline(df_subset, threshold=0.51)
probs_after = np.array([r["fraud_probability"] for r in res_after["results"]])
preds_after = np.array([r["is_fraud"] for r in res_after["results"]])
eval_after = evaluate(probs_after, preds_after)

# --- REPORT ---
print("\n" + "="*50)
print("EVALUATION COMPARISON REPORT")
print("="*50)
print(f"{'Metric':<20} | {'Before Fix':<15} | {'After Fix':<15}")
print("-" * 55)
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR", "Balanced Accuracy", "MCC"]
for m in metrics:
    print(f"{m:<20} | {eval_before[m]:<15.4f} | {eval_after[m]:<15.4f}")
print("-" * 55)
print(f"Predicted Frauds     | {res_before['fraud_count']:<15} | {res_after['fraud_count']:<15}")
print(f"Predicted Legit      | {res_before['legit_count']:<15} | {res_after['legit_count']:<15}")
print(f"Confusion (TP,FP,TN,FN)| {eval_before['TP']},{eval_before['FP']},{eval_before['TN']},{eval_before['FN']} | {eval_after['TP']},{eval_after['FP']},{eval_after['TN']},{eval_after['FN']}")
print("="*50)

# Write metrics to markdown file for easy copying
with open(r"C:\Users\Alan siju\.gemini\antigravity-ide\brain\1e445e9c-3d5d-43ff-970e-204858d1e16c\evaluation_results.md", "w") as f:
    f.write("# Evaluation Comparison Report\n\n")
    f.write("| Metric | Before Fix | After Fix |\n")
    f.write("|---|---|---|\n")
    for m in metrics:
        f.write(f"| {m} | {eval_before[m]:.4f} | {eval_after[m]:.4f} |\n")
    f.write(f"| Predicted Frauds | {res_before['fraud_count']} | {res_after['fraud_count']} |\n")
    f.write(f"| Predicted Legit | {res_before['legit_count']} | {res_after['legit_count']} |\n")
    f.write(f"| Confusion Matrix (TP, FP, TN, FN) | {eval_before['TP']}, {eval_before['FP']}, {eval_before['TN']}, {eval_before['FN']} | {eval_after['TP']}, {eval_after['FP']}, {eval_after['TN']}, {eval_after['FN']} |\n")

print("Report saved.")
