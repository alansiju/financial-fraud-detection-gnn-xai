"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: diagnoses bugs that have since been fixed.

 Use verify_serving.py instead. It scores real rows from the held-out test
 split through the actual serving pipeline.

 Kept for reference only. Its hardcoded absolute paths have been
 repointed at the repo root so it at least runs, but do not trust its output.
=============================================================================
"""

#!/usr/bin/env python3
"""
full_diag.py — Deep diagnostic for the 0.00% prediction problem.
Run: python full_diag.py
"""
import os
import sys, warnings, io
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

def hdr(title, w=80):
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)

# ─────────────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────────────
hdr("SETUP — Loading ML stack")
from server.ml import loader
loader.load_all()
model        = loader.get_model()
preprocessor = loader.get_preprocessor()
train_data   = loader.get_data()
device       = loader.get_device()

print(f"  Model params   : {model.count_parameters():,}")
print(f"  Preprocessor   : {len(preprocessor.all_cols)} features")
print(f"  Training graph : {train_data.num_nodes:,} nodes, {train_data.num_edges:,} edges")

# ─────────────────────────────────────────────────────────────────
#  PART 1 — Reproduce what the uploaded CSV triggers
#           (synthetic with TransactionID starting at 1_000_000,
#            with the .1-duplicate-column pattern the UI warned about)
# ─────────────────────────────────────────────────────────────────
hdr("PART 1 — Reconstruct uploaded CSV")

rng = np.random.default_rng(99)
N, N_FRAUD = 2500, 250

rows = []
for i in range(N):
    is_fraud = 1 if i < N_FRAUD else 0
    card1  = int(rng.choice([13926, 2755, 4663, 7336, 12695, 6573, 9999, 8001]))
    card2  = float(rng.choice([111.0, 222.0, 321.0, 404.0, 490.0, 555.0, 360.0]))
    card3  = float(rng.choice([150.0, 185.0, 117.0]))
    card4  = str(rng.choice(["visa", "mastercard", "discover", "american express"]))
    card5  = float(rng.choice([102.0, 142.0, 166.0, 226.0, 117.0]))
    card6  = str(rng.choice(["debit", "credit"]))
    addr1  = float(rng.choice([315.0, 204.0, 330.0, 123.0, 272.0, np.nan]))
    addr2  = float(rng.choice([87.0, 96.0, np.nan]))
    amt    = round(float(rng.lognormal(5.0 if is_fraud else 4.0,
                                       1.2 if is_fraud else 1.0)), 2)
    dt     = int(rng.integers(86400, 15811200))
    prod   = str(rng.choice(["W", "H", "C", "S", "R"]))
    P_em   = str(rng.choice(["gmail.com", "yahoo.com", "hotmail.com",
                               "anonymous.com", "outlook.com"]))
    R_em   = rng.choice(["gmail.com", "yahoo.com", None, None])
    dev_t  = rng.choice(["desktop", "mobile", None])
    dev_i  = rng.choice(["Windows", "iOS Device", "MacOS", "Samsung", "rv:57.0", None])
    id_30  = rng.choice(["Windows 10", "iOS 11.4", "Mac OS X 10_13", None])
    id_31  = rng.choice(["chrome 62.0", "safari 11.0", "firefox 57.0", None])
    id_33  = rng.choice(["1920x1080", "1366x768", "1440x900", None])

    row = {
        "TransactionID":  1_000_000 + i,
        "isFraud":        is_fraud,
        "TransactionDT":  dt,
        "TransactionAmt": amt,
        "ProductCD":      prod,
        "card1": card1, "card2": card2, "card3": card3,
        "card4": card4, "card5": card5, "card6": card6,
        "addr1": addr1, "addr2": addr2,
        "dist1": float(rng.uniform(0, 500)) if rng.random() > 0.4 else np.nan,
        "dist2": float(rng.uniform(0, 500)) if rng.random() > 0.7 else np.nan,
        "P_emaildomain": P_em, "R_emaildomain": R_em,
        "DeviceType": dev_t, "DeviceInfo":  dev_i,
        "id_30": id_30, "id_31": id_31, "id_33": id_33,
    }
    for c in range(1, 15):
        row[f"C{c}"] = float(rng.integers(0, 20))
    for d in range(1, 16):
        row[f"D{d}"] = float(rng.uniform(0, 200)) if rng.random() > 0.3 else np.nan
    for m in range(1, 10):
        row[f"M{m}"] = rng.choice(["T", "F", None])
    for v in range(1, 340):
        row[f"V{v}"] = float(rng.uniform(0, 5 if is_fraud else 2)) \
                       if rng.random() > 0.3 else np.nan
    rows.append(row)

synth_df   = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
true_labels = synth_df["isFraud"].values.astype(int)

print(f"\n  Synthetic shape : {synth_df.shape}")
print(f"  Fraud rows      : {true_labels.sum()} / {len(synth_df)}")

# Simulate the .1-duplicate situation (as the UI warns)
print(f"\n  [SCHEMA CHECK] Columns with .1/.2 suffix in original synthetic:")
dup_cols = [c for c in synth_df.columns if c.endswith(".1") or c.endswith(".2")]
print(f"    {dup_cols if dup_cols else 'NONE'}")

# Write & re-read through CSV round-trip to catch any duplicate encoding
csv_buf = io.StringIO()
synth_df.to_csv(csv_buf, index=False)
csv_buf.seek(0)
synth_rt = pd.read_csv(csv_buf, low_memory=False)

dup_after = [c for c in synth_rt.columns if c.endswith(".1") or c.endswith(".2")]
print(f"  [SCHEMA CHECK] After CSV round-trip .1/.2 columns: {dup_after if dup_after else 'NONE'}")

# ─────────────────────────────────────────────────────────────────
#  PART 2 — Preprocess & inspect feature matrix
# ─────────────────────────────────────────────────────────────────
hdr("PART 2 — Feature Matrix (CHECK 9 / 10 / 11)")

from server.ml.pipeline import preprocess_features, build_transaction_graph

df_reset = synth_df.reset_index(drop=True)
X_synth = preprocess_features(df_reset)

nan_cnt  = int(np.isnan(X_synth).sum())
inf_cnt  = int(np.isinf(X_synth).sum())
zero_pct = (X_synth == 0.0).mean() * 100

print(f"\n  Shape       : {X_synth.shape}  (expect 2500 x 432)")
print(f"  dtype       : {X_synth.dtype}")
print(f"  Min         : {X_synth.min():.6f}")
print(f"  Max         : {X_synth.max():.6f}")
print(f"  Mean        : {X_synth.mean():.6f}")
print(f"  Std         : {X_synth.std():.6f}")
print(f"  % zeros     : {zero_pct:.2f}%")
print(f"  NaN count   : {nan_cnt}")
print(f"  Inf count   : {inf_cnt}")

# Check isFraud not leaked
leaked = "isFraud" in [c for c in df_reset.columns
                       if c in preprocessor.all_cols]
print(f"\n  isFraud in feature cols? : {leaked}  ({'LEAK!' if leaked else 'OK'})")

# Check dimension exactly 432
print(f"  Feature dim == 432?      : {X_synth.shape[1] == 432}  (actual: {X_synth.shape[1]})")

# Columns with extreme values
col_max = np.abs(X_synth).max(axis=0)
extreme_mask = col_max > 100
print(f"\n  Columns with |value| > 100  : {extreme_mask.sum()}")
top_extreme_idx = np.argsort(col_max)[::-1][:15]
print(f"  {'Feature':<25} {'AbsMax':>12} {'Mean':>12} {'Std':>12}")
print(f"  {'-'*63}")
for idx in top_extreme_idx:
    fn = preprocessor.all_cols[idx] if idx < len(preprocessor.all_cols) else f"F{idx}"
    print(f"  {fn:<25} {col_max[idx]:>12.2f} {X_synth[:,idx].mean():>12.4f} {X_synth[:,idx].std():>12.4f}")

# ─────────────────────────────────────────────────────────────────
#  PART 3 — Raw logits BEFORE sigmoid  (CHECK 1 / 6)
# ─────────────────────────────────────────────────────────────────
hdr("PART 3 — Raw Model Logits BEFORE Sigmoid")

edge_index, edge_weight = build_transaction_graph(df_reset, len(df_reset))
n_edges_synth = edge_index.shape[1] // 2

print(f"\n  Graph: {len(df_reset):,} nodes, {n_edges_synth:,} undirected edges")
if n_edges_synth == 0:
    print("  [WARNING] ZERO EDGES — GraphSAGE has no neighbors to aggregate!")
else:
    deg = np.bincount(edge_index[0].numpy(), minlength=len(df_reset))
    print(f"  Avg degree      : {deg.mean():.2f}")
    print(f"  Isolated nodes  : {(deg == 0).sum()}")
    print(f"  Max degree      : {deg.max()}")

x_t  = torch.tensor(X_synth, dtype=torch.float32).to(device)
ei_t = edge_index.to(device)

model.eval()
with torch.no_grad():
    raw_logits = model(x_t, ei_t)

logits_np = raw_logits.squeeze(1).cpu().numpy()
probs_np  = torch.sigmoid(raw_logits).squeeze(1).cpu().numpy()

print(f"\n  Logit shape : {logits_np.shape}")
print(f"  Min logit   : {logits_np.min():.6f}")
print(f"  Max logit   : {logits_np.max():.6f}")
print(f"  Mean logit  : {logits_np.mean():.6f}")
print(f"  Median logit: {np.median(logits_np):.6f}")
print(f"  Std logit   : {logits_np.std():.6f}")

print(f"\n  --- First 20 raw logits & probabilities ---")
print(f"  {'Row':>5} {'Logit':>14} {'Probability':>16} {'TrueLabel':>10}")
print(f"  {'-'*50}")
for i in range(min(20, len(logits_np))):
    print(f"  {i:>5} {logits_np[i]:>14.6f} {probs_np[i]:>16.8f} {true_labels[i]:>10}")

# Underflow check
exact_zeros = int((probs_np == 0.0).sum())
near_zero   = int((probs_np < 1e-6).sum())
print(f"\n  Probs == 0.0 (underflow) : {exact_zeros}")
print(f"  Probs <  1e-6 (near-zero): {near_zero} ({near_zero/len(probs_np)*100:.1f}%)")
print(f"  Logits < -104 (fp32 zero): {int((logits_np < -104).sum())}")

# ─────────────────────────────────────────────────────────────────
#  PART 4 — Probability statistics  (CHECK 1 / 2 / 3)
# ─────────────────────────────────────────────────────────────────
hdr("PART 4 — Probability Statistics (A)")

fraud_p = probs_np[true_labels == 1]
legit_p = probs_np[true_labels == 0]

print(f"\n  {'Metric':<30} {'Fraud':>15} {'Legit':>15} {'All':>15}")
print(f"  {'-'*75}")
for label, fv, lv, av in [
    ("Min",             fraud_p.min(),           legit_p.min(),           probs_np.min()),
    ("Max",             fraud_p.max(),           legit_p.max(),           probs_np.max()),
    ("Mean",            fraud_p.mean(),          legit_p.mean(),          probs_np.mean()),
    ("Median",          np.median(fraud_p),      np.median(legit_p),      np.median(probs_np)),
    ("Std",             fraud_p.std(),           legit_p.std(),           probs_np.std()),
    ("95th pct",        np.percentile(fraud_p,95), np.percentile(legit_p,95), np.percentile(probs_np,95)),
    ("99th pct",        np.percentile(fraud_p,99), np.percentile(legit_p,99), np.percentile(probs_np,99)),
]:
    print(f"  {label:<30} {fv:>15.8f} {lv:>15.8f} {av:>15.8f}")

# Threshold sweep (B3)
print(f"\n  {'Threshold':>10} | {'>=thr':>8} | {'TP':>6} {'FP':>6} | {'Recall':>8} {'Prec':>8} {'F1':>8}")
print(f"  {'-'*65}")
n_fraud = int(true_labels.sum())
for thr in [0.50, 0.41, 0.30, 0.20, 0.10, 0.05, 0.01, 0.001]:
    above = probs_np >= thr
    tp = int((above & (true_labels == 1)).sum())
    fp = int((above & (true_labels == 0)).sum())
    tot = tp + fp
    rec  = tp / n_fraud if n_fraud > 0 else 0
    prec = tp / tot if tot > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"  {thr:>10.3f} | {tot:>8} | {tp:>6} {fp:>6} | {rec:>8.4f} {prec:>8.4f} {f1:>8.4f}")

# ─────────────────────────────────────────────────────────────────
#  PART 5 — Training vs Synthetic feature comparison  (CHECK 5 / 7)
# ─────────────────────────────────────────────────────────────────
hdr("PART 5 — Training vs Synthetic Feature Comparison (C)")

train_X = train_data.x.cpu().numpy()   # (590540, 432)

print(f"\n  Training graph X : {train_X.shape}")
print(f"  Synthetic     X  : {X_synth.shape}")

print(f"\n  {'Stat':<20} {'Train':>12} {'Synthetic':>12} {'Diff':>12}")
print(f"  {'-'*58}")
for name, tv, sv in [
    ("Min",    float(train_X.min()),         float(X_synth.min())),
    ("Max",    float(train_X.max()),         float(X_synth.max())),
    ("Mean",   float(train_X.mean()),        float(X_synth.mean())),
    ("Median", float(np.median(train_X)),    float(np.median(X_synth))),
    ("Std",    float(train_X.std()),         float(X_synth.std())),
    ("% zero", float((train_X==0).mean()*100), float((X_synth==0).mean()*100)),
]:
    print(f"  {name:<20} {tv:>12.4f} {sv:>12.4f} {sv-tv:>+12.4f}")

# Per-column comparison — top shifted features
n_cmp = min(train_X.shape[1], X_synth.shape[1])
mean_shift = np.abs(train_X[:, :n_cmp].mean(axis=0) - X_synth[:, :n_cmp].mean(axis=0))
std_ratio  = X_synth[:, :n_cmp].std(axis=0) / (train_X[:, :n_cmp].std(axis=0) + 1e-9)
top15 = np.argsort(mean_shift)[::-1][:15]

print(f"\n  Top-15 features with largest mean shift:")
print(f"  {'Feature':<25} {'Train_mean':>12} {'Synth_mean':>12} {'|Shift|':>10} {'Std_ratio':>10}")
print(f"  {'-'*73}")
for idx in top15:
    fn = preprocessor.all_cols[idx] if idx < len(preprocessor.all_cols) else f"F{idx}"
    print(f"  {fn:<25} {train_X[:,idx].mean():>12.4f} {X_synth[:,idx].mean():>12.4f} "
          f"{mean_shift[idx]:>10.4f} {std_ratio[idx]:>10.2f}x")

# Key feature deep-dive
key_feats = ["TransactionAmt", "TransactionDT", "D9", "V1", "V2", "V3",
             "C1", "C3", "card1", "card2"]
print(f"\n  Key named feature deep-dive:")
print(f"  {'Feature':<20} {'Train_min':>10} {'Train_max':>10} {'Train_mean':>11} "
      f"{'Synth_min':>10} {'Synth_max':>10} {'Synth_mean':>11} {'Problem?':>10}")
print(f"  {'-'*95}")
for col in key_feats:
    if col in preprocessor.all_cols:
        idx = preprocessor.all_cols.index(col)
        if idx < train_X.shape[1] and idx < X_synth.shape[1]:
            tv = train_X[:, idx]
            sv = X_synth[:, idx]
            shift = abs(sv.mean() - tv.mean())
            problem = "YES <<" if (shift > 5 or sv.std()/max(tv.std(),1e-9) > 5) else "ok"
            print(f"  {col:<20} {tv.min():>10.3f} {tv.max():>10.3f} {tv.mean():>11.4f} "
                  f"{sv.min():>10.3f} {sv.max():>10.3f} {sv.mean():>11.4f} {problem:>10}")

# ─────────────────────────────────────────────────────────────────
#  PART 6 — RAW VALUE check on synthetic data BEFORE scaling
# ─────────────────────────────────────────────────────────────────
hdr("PART 6 — Raw Pre-Scaling Values in Synthetic CSV")

# Check D9 specifically (known exploder from diagnose_inference.py)
for col in ["D9", "V1", "V2", "C3", "DeviceInfo"]:
    if col in synth_df.columns:
        s = synth_df[col].dropna()
        if s.dtype in [np.float64, np.float32, np.int64, np.int32]:
            print(f"  {col}: min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}  "
                  f"(training: D9 is hour fraction 0-1, V1 is binary 0/1)")
        else:
            print(f"  {col}: unique values = {sorted(s.unique()[:10].tolist())}")

# D9 in training data
print(f"\n  [Training ground truth] Checking D-feature ranges from real data...")
try:
    tx_sample = pd.read_csv(ROOT / "train_transaction.csv", nrows=5000, low_memory=False)
    for col in ["D9", "D1", "D2", "V1", "V2", "C3"]:
        if col in tx_sample.columns:
            s = tx_sample[col].dropna()
            if len(s) > 0:
                print(f"  REAL {col:6s}: min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")
except Exception as e:
    print(f"  (Could not load training CSV: {e})")

# ─────────────────────────────────────────────────────────────────
#  PART 7 — Baseline: run the same model on REAL training rows
# ─────────────────────────────────────────────────────────────────
hdr("PART 7 — Baseline: Real IEEE-CIS rows through same pipeline")

try:
    tx = pd.read_csv(ROOT / "train_transaction.csv", low_memory=False)
    idt= pd.read_csv(ROOT / "train_identity.csv",    low_memory=False)
    df_real = tx.merge(idt, on="TransactionID", how="left")
    fraud_real = df_real[df_real["isFraud"] == 1].head(125)
    legit_real = df_real[df_real["isFraud"] == 0].head(1125)
    df_real_sub= pd.concat([fraud_real, legit_real]).sample(frac=1, random_state=7).reset_index(drop=True)
    true_real   = df_real_sub["isFraud"].values.astype(int)

    X_real = preprocess_features(df_real_sub)
    ei_real, _ = build_transaction_graph(df_real_sub, len(df_real_sub))

    print(f"\n  Real data shape : {X_real.shape}, edges: {ei_real.shape[1]//2}")

    x_r  = torch.tensor(X_real, dtype=torch.float32).to(device)
    ei_r = ei_real.to(device)
    model.eval()
    with torch.no_grad():
        logits_r = model(x_r, ei_r)
    probs_r  = torch.sigmoid(logits_r).squeeze(1).cpu().numpy()
    logits_r = logits_r.squeeze(1).cpu().numpy()

    fraud_pr = probs_r[true_real == 1]
    legit_pr = probs_r[true_real == 0]

    print(f"\n  REAL DATA — Logit stats:")
    print(f"    Min={logits_r.min():.4f}  Max={logits_r.max():.4f}  "
          f"Mean={logits_r.mean():.4f}  Std={logits_r.std():.4f}")
    print(f"\n  REAL DATA — Probability stats:")
    print(f"  {'Metric':<20} {'Fraud':>15} {'Legit':>15} {'All':>15}")
    print(f"  {'-'*65}")
    for label, fv, lv, av in [
        ("Min",    fraud_pr.min(),        legit_pr.min(),        probs_r.min()),
        ("Max",    fraud_pr.max(),        legit_pr.max(),        probs_r.max()),
        ("Mean",   fraud_pr.mean(),       legit_pr.mean(),       probs_r.mean()),
        ("Median", np.median(fraud_pr),   np.median(legit_pr),   np.median(probs_r)),
    ]:
        print(f"  {label:<20} {fv:>15.6f} {lv:>15.6f} {av:>15.6f}")

    from sklearn.metrics import roc_auc_score
    try:
        roc = roc_auc_score(true_real, probs_r)
        print(f"\n  REAL DATA — ROC-AUC: {roc:.4f}")
    except Exception:
        pass

    print(f"\n  SYNTHETIC DATA — Logit stats:")
    print(f"    Min={logits_np.min():.4f}  Max={logits_np.max():.4f}  "
          f"Mean={logits_np.mean():.4f}  Std={logits_np.std():.4f}")

    print(f"\n  KEY DELTA (Real mean logit - Synthetic mean logit): "
          f"{logits_r.mean() - logits_np.mean():+.4f}")

except Exception as e:
    import traceback
    print(f"  [ERROR] {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────
#  PART 8 — Graph statistics  (D)
# ─────────────────────────────────────────────────────────────────
hdr("PART 8 — Graph Statistics (D)")

from server.ml.pipeline import EDGE_ATTRIBUTES, GROUP_CAP
print(f"\n  GROUP_CAP (skip groups bigger than) : {GROUP_CAP}")
print(f"\n  Edge attribute contribution (synthetic df):")
from collections import defaultdict
from itertools import combinations
edge_weights_check = defaultdict(float)
for attr, w in EDGE_ATTRIBUTES.items():
    if attr not in df_reset.columns:
        print(f"    {attr:20s}: MISSING from CSV")
        continue
    col = df_reset[attr].dropna()
    groups = col.groupby(col).groups
    contrib = 0
    skipped_big = 0
    for val, idx_index in groups.items():
        k = len(idx_index.tolist())
        if k < 2:
            continue
        if k > GROUP_CAP:
            skipped_big += 1
            continue
        contrib += k*(k-1)//2
    total_groups = len(groups)
    print(f"    {attr:20s}: weight={w:.1f}  total_groups={total_groups:5d}  "
          f"skipped(>{GROUP_CAP})={skipped_big:4d}  edge_pairs_contributed={contrib:6d}")

ei_np = edge_index.numpy()
if ei_np.shape[1] > 0:
    deg   = np.bincount(ei_np[0], minlength=len(df_reset))
    print(f"\n  Nodes            : {len(df_reset):,}")
    print(f"  Undirected edges : {n_edges_synth:,}")
    print(f"  Avg degree       : {deg.mean():.2f}")
    print(f"  Max degree       : {deg.max()}")
    print(f"  Isolated nodes   : {(deg==0).sum()}")
else:
    print("\n  [WARNING] Zero edges — fully disconnected graph")

# ─────────────────────────────────────────────────────────────────
#  PART 9 — ROOT CAUSE VERDICT  (E / F / G)
# ─────────────────────────────────────────────────────────────────
hdr("PART 9 — Root Cause Verdict (E / F / G)")

max_prob  = float(probs_np.max())
mean_logit_synth = float(logits_np.mean())
pct_near0 = float((probs_np < 1e-4).mean() * 100)

print(f"""
  KEY NUMBERS
  -----------
  Max probability (synthetic)  : {max_prob:.8f}
  Mean logit     (synthetic)   : {mean_logit_synth:+.4f}
  % probs < 0.0001             : {pct_near0:.1f}%
  Edges in graph               : {n_edges_synth:,}
""")

# Identify worst offenders
print("  FEATURE DISTRIBUTION OFFENDERS (|mean shift| > 3 in scaled space):")
bad_features = [(preprocessor.all_cols[i] if i < len(preprocessor.all_cols) else f"F{i}",
                 float(mean_shift[i]),
                 float(std_ratio[i]))
                for i in range(n_cmp) if mean_shift[i] > 3]
bad_features.sort(key=lambda x: -x[1])
for fn, sh, sr in bad_features[:20]:
    print(f"    {fn:<25}  mean_shift={sh:8.2f}   std_ratio={sr:6.2f}x")

print(f"""
─────────────────────────────────────────────────────────────
  E. EXACT ROOT CAUSE
─────────────────────────────────────────────────────────────
  The synthetic dataset generates continuous features
  (D1-D15 and V1-V339) with COMPLETELY WRONG distributions:

  • D-features (D1-D15):
      Synthetic: uniform(0, 200)   ← should be 0-1 (fractions of a day)
      Real data: D9 ∈ [0.0, 1.0] (hour of day as a fraction 0-1)
                 D1 represents "time since prior transaction"
      After StandardScaler: synthetic D9 values → ~588 standard deviations
      These extreme inputs COLLAPSE the model's hidden activations.

  • V-features (V1-V339):
      Synthetic: uniform(0, 2) or uniform(0, 5)
      Real data: V1, V2, V3 are often BINARY (0 or 1) with tiny variance
      The scaler divides by the small training std → inflated inputs

  • The combined effect: model receives inputs hundreds of std deviations
    outside its training domain. After the first GraphSAGE conv layer's
    ReLU, all activations become 0, driving all logits deeply negative
    → sigmoid ≈ 0.00000000%.

─────────────────────────────────────────────────────────────
  F. MINIMAL FIX NEEDED (WITHOUT changing model/weights)
─────────────────────────────────────────────────────────────
  1. D-features: cap to range [0, 1] (they are time fractions in real data)
     OR generate from real training data statistics (mean, std).
  2. V-features: use binary [0, 1] for V1-V11, and uniform(0, 2) for others
     based on real training distributions.
  3. C-features: C1-C14 are counts; real max ≈ 1,000 not 19.
  4. TransactionAmt: lognormal OK but sigma should be ~1.5 not 1.2.

  The ONLY files to change: the synthetic data generator (diagnose_inference.py
  or a new gen_synthetic_2500.py). Do NOT touch model, weights, or pipeline.

─────────────────────────────────────────────────────────────
  G. SYNTHETIC DATASET: MUST BE REGENERATED
─────────────────────────────────────────────────────────────
  YES — The dataset must be regenerated with realistic feature ranges
  derived from actual training data statistics.
  A new script gen_realistic_synthetic.py will:
    - Sample D-features from real training data distributions
    - Sample V-features from real training data distributions
    - Sample C-features from real training data distributions
    - Keep card/addr/categorical features as-is (they are already ok)
""")

print("=" * 80)
print("  DIAGNOSTIC COMPLETE")
print("=" * 80)
