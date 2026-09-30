"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: generates a synthetic CSV whose non-amount columns are random, so no model can score better than chance on it.

 Use verify_serving.py instead. It scores real rows from the held-out test
 split through the actual serving pipeline.

 Kept for reference only. Its hardcoded absolute paths have been
 repointed at the repo root so it at least runs, but do not trust its output.
=============================================================================
"""

#!/usr/bin/env python3
"""
diagnose_inference.py
=====================
Comprehensive diagnostic covering all 11 investigation points for the
"0 fraud predictions" bug on the 2,500-row synthetic IEEE-CIS CSV.

Run from the project root:
    python diagnose_inference.py

Does NOT modify any model, weights, threshold, or pipeline.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Fix encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Section header helper ──────────────────────────────────────────────────────
def section(title, width=80):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

# ════════════════════════════════════════════════════════════════════════════════
#  SETUP — load ML stack
# ════════════════════════════════════════════════════════════════════════════════
section("SETUP -- Loading ML stack")

from server.ml import loader
loader.load_all()

model        = loader.get_model()
preprocessor = loader.get_preprocessor()
train_data   = loader.get_data()   # PyG Data object (fraud_graph.pt)

if model is None:
    print("[FATAL] Model not loaded. Aborting.")
    sys.exit(1)
if preprocessor is None:
    print("[FATAL] Preprocessor not loaded. Aborting.")
    sys.exit(1)

print(f"\n  Model parameters : {model.count_parameters():,}")
print(f"  Preprocessor cols: {len(preprocessor.all_cols)}")
print(f"  Training graph   : {train_data.num_nodes:,} nodes, {train_data.num_edges:,} edges")
print(f"  Training features: {train_data.x.shape}")


# ════════════════════════════════════════════════════════════════════════════════
#  GENERATE the 2,500-row synthetic dataset  (mirrors what the UI uploads)
# ════════════════════════════════════════════════════════════════════════════════
section("GENERATING 2,500-row synthetic IEEE-CIS-style dataset")

rng = np.random.default_rng(42)
N       = 2500
N_FRAUD = 250

rows = []
for i in range(N):
    is_fraud = 1 if i < N_FRAUD else 0
    card1 = int(rng.choice([13926, 2755, 4663, 7336, 12695, 6573, 9999, 8001]))
    card2 = float(rng.choice([111.0, 222.0, 321.0, 404.0, 490.0, 555.0, 360.0]))
    card3 = float(rng.choice([150.0, 185.0, 117.0]))
    card4 = str(rng.choice(["visa", "mastercard", "discover", "american express"]))
    card5 = float(rng.choice([102.0, 142.0, 166.0, 226.0, 117.0]))
    card6 = str(rng.choice(["debit", "credit"]))
    addr1 = float(rng.choice([315.0, 204.0, 330.0, 123.0, 272.0, np.nan]))
    addr2 = float(rng.choice([87.0, 96.0, np.nan]))
    amt   = round(float(rng.lognormal(mean=5.0 if is_fraud else 4.0,
                                      sigma=1.2 if is_fraud else 1.0)), 2)
    dt    = int(rng.integers(86400, 15811200))
    prod  = str(rng.choice(["W", "H", "C", "S", "R"]))
    P_email = str(rng.choice(["gmail.com", "yahoo.com", "hotmail.com",
                               "anonymous.com", "outlook.com"]))
    R_email_choices = ["gmail.com", "yahoo.com", None, None]
    R_email = rng.choice(R_email_choices)  # type: ignore
    dev_type_choices = ["desktop", "mobile", None]
    dev_type = rng.choice(dev_type_choices)  # type: ignore
    dev_info_choices = ["Windows", "iOS Device", "MacOS", "Samsung", "rv:57.0", None]
    dev_info = rng.choice(dev_info_choices)  # type: ignore
    id30_choices = ["Windows 10", "iOS 11.4", "Mac OS X 10_13", None]
    id_30 = rng.choice(id30_choices)  # type: ignore
    id31_choices = ["chrome 62.0", "safari 11.0", "firefox 57.0", None]
    id_31 = rng.choice(id31_choices)  # type: ignore
    id33_choices = ["1920x1080", "1366x768", "1440x900", None]
    id_33 = rng.choice(id33_choices)  # type: ignore

    row = {
        "TransactionID":  3000000 + i,
        "isFraud":        is_fraud,
        "TransactionDT":  dt,
        "TransactionAmt": amt,
        "ProductCD":      prod,
        "card1": card1, "card2": card2, "card3": card3,
        "card4": card4, "card5": card5, "card6": card6,
        "addr1": addr1, "addr2": addr2,
        "dist1": float(rng.uniform(0, 500)) if rng.random() > 0.4 else np.nan,
        "dist2": float(rng.uniform(0, 500)) if rng.random() > 0.7 else np.nan,
        "P_emaildomain": P_email,
        "R_emaildomain": R_email,
        "DeviceType": dev_type,
        "DeviceInfo":  dev_info,
        "id_30": id_30,
        "id_31": id_31,
        "id_33": id_33,
    }
    for c in range(1, 15):
        row[f"C{c}"] = float(rng.integers(0, 20))
    for d in range(1, 16):
        row[f"D{d}"] = float(rng.uniform(0, 200)) if rng.random() > 0.3 else np.nan
    for m in range(1, 10):
        row[f"M{m}"] = rng.choice(["T", "F", None])  # type: ignore
    for v in range(1, 340):
        if rng.random() > 0.3:
            row[f"V{v}"] = float(rng.uniform(0, 5 if is_fraud else 2))
        else:
            row[f"V{v}"] = np.nan
    rows.append(row)

synth_df    = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
true_labels = synth_df["isFraud"].values.astype(int)
n_total     = len(synth_df)
n_fraud     = int(true_labels.sum())
n_legit     = n_total - n_fraud

print(f"  Dataset shape : {synth_df.shape}")
print(f"  Fraud rows    : {n_fraud} ({n_fraud/n_total*100:.1f}%)")
print(f"  Legit rows    : {n_legit} ({n_legit/n_total*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 10: Duplicate Column Schema
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 10: CSV Schema -- Duplicate Column Detection")

cols     = synth_df.columns.tolist()
suffixed = [c for c in cols if c.endswith(".1") or c.endswith(".2")]
KNOWN_DUP_PAIRS = [("id_30", "id_30.1"), ("id_31", "id_31.1"), ("id_33", "id_33.1")]
found_dups = [(a, b) for a, b in KNOWN_DUP_PAIRS if a in cols and b in cols]

print(f"\n  Total columns            : {len(cols)}")
print(f"  .1/.2-suffixed columns   : {suffixed if suffixed else 'NONE'}")
print(f"  Known problematic pairs  : {found_dups if found_dups else 'NONE'}")
if found_dups or suffixed:
    print("  [WARNING] Duplicate columns detected -- can cause shape mismatches!")
else:
    print("  [OK] No duplicate columns.")


# ════════════════════════════════════════════════════════════════════════════════
#  PREPROCESS
# ════════════════════════════════════════════════════════════════════════════════
section("PREPROCESSING -- Building Feature Matrix X")

from server.ml.pipeline import preprocess_features, build_transaction_graph

df_reset = synth_df.reset_index(drop=True)
print("\n  Running preprocess_features()...")
X = preprocess_features(df_reset)
print(f"  Feature matrix X shape: {X.shape}")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 8: Feature Matrix Statistics
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 8: Feature Matrix Sent to GraphSAGE")

nan_count  = int(np.isnan(X).sum())
inf_count  = int(np.isinf(X).sum())
zero_count = int((X == 0.0).sum())
total_vals = X.size

print(f"\n  Shape              : {X.shape}  (rows x features)")
print(f"  dtype              : {X.dtype}")
print(f"  Min                : {X.min():.6f}")
print(f"  Max                : {X.max():.6f}")
print(f"  Mean               : {X.mean():.6f}")
print(f"  Std deviation      : {X.std():.6f}")
print(f"  % zero values      : {zero_count/total_vals*100:.2f}%  ({zero_count:,} / {total_vals:,})")
print(f"  % NaN values       : {nan_count/total_vals*100:.4f}%  ({nan_count:,})")
print(f"  % Inf values       : {inf_count/total_vals*100:.4f}%  ({inf_count:,})")

col_zero_pct = (X == 0.0).mean(axis=0) * 100
print(f"\n  Columns 100% zero  : {(col_zero_pct == 100).sum()} / {X.shape[1]}")
print(f"  Columns  >90% zero : {(col_zero_pct > 90).sum()} / {X.shape[1]}")
print(f"  Columns  >50% zero : {(col_zero_pct > 50).sum()} / {X.shape[1]}")

row_norms     = np.linalg.norm(X, axis=1)
zero_norm_rows = int((row_norms == 0.0).sum())
print(f"\n  Per-row L2 norms:")
print(f"    Min={row_norms.min():.4f}  Max={row_norms.max():.4f}  "
      f"Mean={row_norms.mean():.4f}  Median={np.median(row_norms):.4f}  "
      f"Std={row_norms.std():.4f}")
print(f"    Rows with zero norm: {zero_norm_rows}")


# ════════════════════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ════════════════════════════════════════════════════════════════════════════════
section("GRAPH CONSTRUCTION")

edge_index, edge_weight = build_transaction_graph(df_reset, len(df_reset))
n_edges = edge_index.shape[1] // 2 if edge_index.shape[1] > 0 else 0

print(f"\n  Nodes             : {len(df_reset):,}")
print(f"  Undirected edges  : {n_edges:,}")
print(f"  Bidirectional EI  : {edge_index.shape}")
if edge_weight.numel() > 0:
    print(f"  Edge weight range : [{edge_weight.min():.1f}, {edge_weight.max():.1f}]")
else:
    print("  [WARNING] No edges built! Graph is fully disconnected.")

train_density = train_data.num_edges / max(train_data.num_nodes, 1)
our_density   = (n_edges * 2) / max(len(df_reset), 1)
print(f"\n  Training graph density : {train_density:.4f} edges/node")
print(f"  Inference graph density: {our_density:.4f} edges/node")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 6: Raw Logits BEFORE Sigmoid
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 6: Raw Model Logits BEFORE Sigmoid")

device    = loader.get_device()
x_tensor  = torch.tensor(X, dtype=torch.float32).to(device)
ei_tensor = edge_index.to(device)

model.eval()
with torch.no_grad():
    raw_logits = model(x_tensor, ei_tensor)   # [N, 1]

logits_np = raw_logits.squeeze(1).cpu().numpy()

print(f"\n  Logit shape  : {logits_np.shape}")
print(f"  Min logit    : {logits_np.min():.6f}")
print(f"  Max logit    : {logits_np.max():.6f}")
print(f"  Mean logit   : {logits_np.mean():.6f}")
print(f"  Median logit : {np.median(logits_np):.6f}")
print(f"  Std logit    : {logits_np.std():.6f}")

# Text histogram
lo = float(np.floor(logits_np.min()) - 1)
hi = float(np.ceil(logits_np.max())  + 1)
n_bins = min(25, max(5, int(hi - lo)))
logit_hist, logit_edges = np.histogram(logits_np, bins=n_bins, range=(lo, hi))
print(f"\n  Logit distribution histogram:")
for i in range(len(logit_hist)):
    bar = "#" * min(int(logit_hist[i] / max(n_total / 40, 1)), 40)
    print(f"    [{logit_edges[i]:8.2f}, {logit_edges[i+1]:8.2f})  {logit_hist[i]:5d}  {bar}")

# Float32 underflow boundaries
print(f"\n  Float32 underflow thresholds:")
print(f"    sigmoid(-87)  = {torch.sigmoid(torch.tensor(-87.0)).item():.2e}  (near underflow)")
print(f"    sigmoid(-104) = {torch.sigmoid(torch.tensor(-104.0)).item():.2e}  (exact 0.0 float32)")

extreme_neg   = int((logits_np < -87).sum())
extreme_neg104 = int((logits_np < -104).sum())
positive_logits = int((logits_np > 0).sum())
print(f"    Logits < -87  : {extreme_neg}")
print(f"    Logits < -104 : {extreme_neg104}")
print(f"    Logits > 0    : {positive_logits}  (would give prob > 0.5)")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 7: Numerical Underflow
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 7: Sigmoid Probabilities -- Numerical Underflow")

probs = torch.sigmoid(raw_logits).squeeze(1).cpu().numpy()

exact_zeros = int((probs == 0.0).sum())
exact_ones  = int((probs == 1.0).sum())
near_zero   = int((probs < 1e-6).sum())

print(f"\n  Probs == 0.0 (exact)   : {exact_zeros}  ({exact_zeros/n_total*100:.2f}%)")
print(f"  Probs == 1.0 (exact)   : {exact_ones}  ({exact_ones/n_total*100:.2f}%)")
print(f"  Probs <  1e-6 (~zero)  : {near_zero}  ({near_zero/n_total*100:.2f}%)")

if exact_zeros > 0:
    zero_logits = logits_np[probs == 0.0]
    print(f"\n  Logits that gave exact 0.0 probs:")
    print(f"    Min={zero_logits.min():.4f}  Max={zero_logits.max():.4f}  Mean={zero_logits.mean():.4f}")
    print("  [DIAGNOSIS] Numerical underflow confirmed for these transactions.")
else:
    print("\n  [OK] No numerical underflow to exactly 0.0 detected.")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 1: Probability Statistics
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 1: Fraud Probability Statistics -- All 2,500 Transactions")

fraud_probs = probs[true_labels == 1]
legit_probs = probs[true_labels == 0]

hdr = f"  {'Metric':<30} {'Fraud (n=%d)'%n_fraud:>15} {'Legit (n=%d)'%n_legit:>15} {'All (n=%d)'%n_total:>15}"
print(f"\n{hdr}")
print(f"  {'─'*75}")

def row(label, fv, lv, av):
    print(f"  {label:<30} {fv:>15.6f} {lv:>15.6f} {av:>15.6f}")

row("Minimum",         fraud_probs.min(),             legit_probs.min(),             probs.min())
row("Maximum",         fraud_probs.max(),             legit_probs.max(),             probs.max())
row("Mean",            fraud_probs.mean(),            legit_probs.mean(),            probs.mean())
row("Median",          np.median(fraud_probs),        np.median(legit_probs),        np.median(probs))
row("Std deviation",   fraud_probs.std(),             legit_probs.std(),             probs.std())
row("25th percentile", np.percentile(fraud_probs,25), np.percentile(legit_probs,25), np.percentile(probs,25))
row("75th percentile", np.percentile(fraud_probs,75), np.percentile(legit_probs,75), np.percentile(probs,75))
row("95th percentile", np.percentile(fraud_probs,95), np.percentile(legit_probs,95), np.percentile(probs,95))
row("99th percentile", np.percentile(fraud_probs,99), np.percentile(legit_probs,99), np.percentile(probs,99))


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 2: Top-50 Transactions
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 2: Top 50 Transactions by Fraud Probability")

top50_idx = np.argsort(probs)[::-1][:50]
tx_ids    = synth_df["TransactionID"].values

print(f"\n  {'Rank':>5} {'TransactionID':>15} {'Prob':>12} {'PredClass':>10} {'TrueLabel':>10}")
print(f"  {'─'*57}")
for rank, idx in enumerate(top50_idx, 1):
    p    = probs[idx]
    pred = 1 if p >= 0.5 else 0
    true = true_labels[idx]
    flag = "  *** FRAUD" if true == 1 else ""
    print(f"  {rank:>5} {tx_ids[idx]:>15} {p:>12.6f} {pred:>10} {true:>10}{flag}")

top50_true = true_labels[top50_idx]
print(f"\n  Actual fraud in top-50  : {top50_true.sum()} / 50")
print(f"  Fraud cases in top-50   : {top50_true.sum()} / {n_fraud} total fraud")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 3: Threshold Sweep
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 3: Transactions at Each Threshold")

thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.10, 0.05]
print(f"\n  {'Threshold':>10} | {'Total>=thr':>10} | {'TP':>7} {'FP':>7} | {'Recall':>8} {'Prec':>8} {'F1':>8}")
print(f"  {'─'*72}")
for thr in thresholds:
    above = probs >= thr
    tp    = int((above & (true_labels == 1)).sum())
    fp    = int((above & (true_labels == 0)).sum())
    tot   = tp + fp
    rec   = tp / n_fraud if n_fraud > 0 else 0.0
    prec  = tp / tot     if tot > 0     else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"  {thr:>10.2f} | {tot:>10} | {tp:>7} {fp:>7} | {rec:>8.4f} {prec:>8.4f} {f1:>8.4f}")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 4: Distribution by Class
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 4: Probability Distribution -- Fraud vs Legitimate")

bins = np.arange(0, 1.05, 0.05)
fh, _ = np.histogram(fraud_probs, bins=bins)
lh, _ = np.histogram(legit_probs, bins=bins)

print(f"\n  {'Bin':>14} | {'Fraud':>7} {'Bar':>20} | {'Legit':>7} {'Bar':>20}")
print(f"  {'─'*72}")
scale = max(n_legit // 125, 1)
for i in range(len(bins) - 1):
    fb = "#" * min(fh[i], 20)
    lb = "#" * min(lh[i] // scale, 20)
    print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f})  | {fh[i]:>7} {fb:<20} | {lh[i]:>7} {lb:<20}")

print(f"\n  Fraud >= 0.50: {(fraud_probs >= 0.50).sum()} / {n_fraud}")
print(f"  Legit >= 0.50: {(legit_probs >= 0.50).sum()} / {n_legit}")
print(f"  Fraud >= 0.10: {(fraud_probs >= 0.10).sum()} / {n_fraud}")
print(f"  Fraud >= 0.05: {(fraud_probs >= 0.05).sum()} / {n_fraud}")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 5: Classification Metrics
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 5: Classification Metrics")

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve,
)

preds_50 = (probs >= 0.50).astype(int)

try:
    roc_auc = roc_auc_score(true_labels, probs)
except Exception as e:
    roc_auc = float("nan"); print(f"  [WARN] ROC-AUC: {e}")

try:
    pr_auc = average_precision_score(true_labels, probs)
except Exception as e:
    pr_auc = float("nan"); print(f"  [WARN] PR-AUC: {e}")

cm = confusion_matrix(true_labels, preds_50)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, n_fraud, 0)

print(f"\n  -- At threshold 0.50 --")
print(f"  ROC-AUC   : {roc_auc:.6f}")
print(f"  PR-AUC    : {pr_auc:.6f}")
print(f"  Accuracy  : {accuracy_score(true_labels, preds_50):.6f}")
print(f"  Precision : {precision_score(true_labels, preds_50, zero_division=0):.6f}")
print(f"  Recall    : {recall_score(true_labels, preds_50, zero_division=0):.6f}")
print(f"  F1        : {f1_score(true_labels, preds_50, zero_division=0):.6f}")
print(f"\n  Confusion Matrix (thr=0.50):")
print(f"                Pred=0   Pred=1")
print(f"    True=0  {tn:>8}  {fp:>7}")
print(f"    True=1  {fn:>8}  {tp:>7}")

try:
    fpr, tpr, roc_thr = roc_curve(true_labels, probs)
    j       = tpr - fpr
    opt_idx = int(np.argmax(j))
    opt_thr = float(roc_thr[opt_idx])
    preds_opt = (probs >= opt_thr).astype(int)
    cm2   = confusion_matrix(true_labels, preds_opt)
    tn2, fp2, fn2, tp2 = cm2.ravel() if cm2.size == 4 else (0, 0, n_fraud, 0)
    print(f"\n  Optimal threshold (Youden's J): {opt_thr:.6f}")
    print(f"  -- At threshold {opt_thr:.4f} --")
    print(f"  Precision : {precision_score(true_labels, preds_opt, zero_division=0):.6f}")
    print(f"  Recall    : {recall_score(true_labels, preds_opt, zero_division=0):.6f}")
    print(f"  F1        : {f1_score(true_labels, preds_opt, zero_division=0):.6f}")
    print(f"  Confusion Matrix (thr={opt_thr:.4f}):")
    print(f"                Pred=0   Pred=1")
    print(f"    True=0  {tn2:>8}  {fp2:>7}")
    print(f"    True=1  {fn2:>8}  {tp2:>7}")
except Exception as e:
    print(f"  [WARN] Optimal threshold calc failed: {e}")


# ════════════════════════════════════════════════════════════════════════════════
#  CHECK 9: Feature Distribution -- Synthetic vs Training
# ════════════════════════════════════════════════════════════════════════════════
section("CHECK 9: Feature Distribution -- Synthetic vs Training (fraud_graph.pt)")

train_X  = train_data.x.cpu().numpy()
synth_X  = X

print(f"\n  Training graph X : {train_X.shape}")
print(f"  Synthetic     X  : {synth_X.shape}")

print(f"\n  {'Stat':<20} {'Train':>12} {'Synthetic':>12} {'Diff':>12}")
print(f"  {'─'*58}")
for name, tv, sv in [
    ("Min",     train_X.min(),          synth_X.min()),
    ("Max",     train_X.max(),          synth_X.max()),
    ("Mean",    train_X.mean(),         synth_X.mean()),
    ("Median",  float(np.median(train_X)), float(np.median(synth_X))),
    ("Std",     train_X.std(),          synth_X.std()),
    ("% zero",  float((train_X==0).mean()*100), float((synth_X==0).mean()*100)),
]:
    print(f"  {name:<20} {tv:>12.4f} {sv:>12.4f} {sv-tv:>+12.4f}")

# Per-column analysis
n_compare = min(train_X.shape[1], synth_X.shape[1])
feat_names = preprocessor.all_cols[:n_compare]
feat_mean_diff = np.abs(
    train_X[:, :n_compare].mean(axis=0) - synth_X[:, :n_compare].mean(axis=0)
)
top10_idx = np.argsort(feat_mean_diff)[::-1][:10]
print(f"\n  Top-10 features with largest mean shift (scaled space):")
print(f"  {'Feature':<25} {'Train_mean':>12} {'Synth_mean':>12} {'|Diff|':>10}")
print(f"  {'─'*62}")
for idx in top10_idx:
    fn = feat_names[idx] if idx < len(feat_names) else f"Feature_{idx}"
    print(f"  {fn:<25} {train_X[:,idx].mean():>12.4f} {synth_X[:,idx].mean():>12.4f} "
          f"{feat_mean_diff[idx]:>10.4f}")

# Named feature comparison
key_cols = ["TransactionAmt", "TransactionDT", "card1", "card2", "C1", "D1", "V1"]
print(f"\n  Key named features:")
print(f"  {'Feature':<20} {'Train_mean':>12} {'Synth_mean':>12} {'Train_std':>12} {'Synth_std':>12}")
print(f"  {'─'*70}")
for col in key_cols:
    if col in feat_names:
        i = feat_names.index(col)
        if i < train_X.shape[1] and i < synth_X.shape[1]:
            print(f"  {col:<20} {train_X[:,i].mean():>12.4f} {synth_X[:,i].mean():>12.4f} "
                  f"{train_X[:,i].std():>12.4f} {synth_X[:,i].std():>12.4f}")
    else:
        print(f"  {col:<20}  -- not found in preprocessor.all_cols")


# ════════════════════════════════════════════════════════════════════════════════
#  ROOT CAUSE DIAGNOSIS
# ════════════════════════════════════════════════════════════════════════════════
section("ROOT CAUSE DIAGNOSIS")

max_prob_val    = float(probs.max())
mean_diff_val   = float(fraud_probs.mean() - legit_probs.mean())
n_above_50      = int((probs >= 0.50).sum())
n_above_10      = int((probs >= 0.10).sum())
pct_exact_zero  = exact_zeros / n_total * 100

print(f"""
  KEY NUMBERS
  -----------
  Max probability overall   : {max_prob_val:.6f}
  Mean prob (fraud rows)    : {fraud_probs.mean():.6f}
  Mean prob (legit rows)    : {legit_probs.mean():.6f}
  Mean diff (fraud - legit) : {mean_diff_val:+.6f}
  Count >= 0.50             : {n_above_50}
  Count >= 0.10             : {n_above_10}
  Exact zero probabilities  : {exact_zeros}  ({pct_exact_zero:.2f}%)
  Positive logits (>0)      : {positive_logits}
  Logits < -104 (float32=0) : {extreme_neg104}
  ROC-AUC                   : {roc_auc:.6f}
""")

print("  PRIMARY CAUSE VERDICT:")
if exact_zeros > n_total * 0.5:
    print("""
  [B] NUMERICAL UNDERFLOW -- More than 50% of probabilities are exactly 0.0.
      Logits are so deeply negative that float32 sigmoid underflows to 0.0.
      Likely driven by [C]+[D]: synthetic features that, after StandardScaler,
      produce very large negative logits through the learned weight matrices.
""")
elif max_prob_val < 0.05:
    print("""
  [C]+[D] FEATURE DISTRIBUTION MISMATCH + SYNTHETIC DATA QUALITY
  All probabilities are < 5%. The model sees feature vectors far outside the
  training distribution. After BatchNorm (using training running stats), the
  activations collapse, driving all logits deeply negative.
  Secondary factor: sparse graph (few edges) reduces GraphSAGE aggregation quality.
""")
elif max_prob_val < 0.50 and abs(mean_diff_val) < 0.01:
    print("""
  [C]+[E] FEATURE MISMATCH + GRAPH STRUCTURE MISMATCH
  Probabilities are compressed below 0.50 and fraud/legit are indistinguishable.
  The sparse inference graph (vs dense training graph) + OOD synthetic features
  together prevent the model from producing discriminative scores.
""")
elif max_prob_val < 0.50 and mean_diff_val > 0.005:
    print("""
  [F] THRESHOLDING -- The model shows SOME separation (fraud mean > legit mean)
  but all probabilities fall below the 0.50 threshold. The fix is threshold
  calibration, not model retraining. See the Youden-optimal threshold above.
""")
else:
    print("""
  Mixed or unclear. Review the individual section outputs above for the
  dominant driver. Most likely [C] feature distribution mismatch.
""")

print("""
  NEXT STEPS (no model changes):
  ─────────────────────────────
  1. Verify the actual uploaded CSV columns against preprocessor.all_cols.
     Count how many of the 432 expected features are present vs filled with 0.

  2. Check whether the synthetic generator produces values in realistic ranges
     (e.g., V-features 0-2 in real data vs 0-5 here, TransactionAmt scale, etc.)

  3. Compare BatchNorm running_mean/running_var of the training checkpoint
     against the actual activation distributions for the synthetic batch.

  4. If [F] is confirmed: lower threshold to the Youden-optimal value above,
     or apply Platt scaling on a calibration split of real data.
""")

print("=" * 80)
print("  DIAGNOSTIC COMPLETE")
print("=" * 80)
