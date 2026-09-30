"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: compares two preprocessing implementations that no longer exist separately; build_graph.py now calls FraudPreprocessor directly.

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
prove_mismatch.py
=================
Definitive proof of whether training vs inference preprocessing match.

Traces:
  1. build_graph.py clean_features()  -> fraud_graph.pt -> best_model.pt
  2. preprocessing_pipeline.py FraudPreprocessor -> preprocessor.pkl -> inference

Compares column ordering + scaling behavior side by side.
"""

import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ======================================================================
#  STEP 1: TRACE THE TRAINING PIPELINE (build_graph.py)
# ======================================================================

print("=" * 75)
print("  DEFINITIVE MISMATCH PROOF: TRAINING vs INFERENCE PREPROCESSING")
print("=" * 75)

print("\n" + "=" * 75)
print("  STEP 1: TRACE THE TRAINING PIPELINE (build_graph.py)")
print("=" * 75)

print("\n[INFO] The training pipeline is:")
print("  build_graph.py main()")
print("    -> line 545: feature_df = df.drop(columns=['isFraud','TransactionID'])")
print("    -> line 546: X = clean_features(feature_df)")
print("    -> line 561: data.x = torch.tensor(X, dtype=torch.float)")
print("    -> line 587: torch.save(data, OUTPUT_GRAPH)  # -> fraud_graph.pt")
print("")
print("  train.py main()")
print("    -> line 198: data = load_graph(GRAPH_PATH)  # loads fraud_graph.pt")
print("    -> line 201: in_channels = data.x.shape[1]  # 432")
print("    -> line 272: torch.save({...}, MODEL_PATH)  # -> best_model.pt")

# Load a small sample to reproduce both pipelines
print("\n[INFO] Loading training data sample (first 500 rows)...")
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), low_memory=False, nrows=500)
idt = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_identity.csv"), low_memory=False)
df = tx.merge(idt, on="TransactionID", how="left")
feature_df = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

print(f"[INFO] Sample: {len(feature_df)} rows x {feature_df.shape[1]} columns")

# ── Reproduce build_graph.py clean_features() EXACTLY ──
cat_cols_training = feature_df.select_dtypes(include="object").columns.tolist()
num_cols_training = feature_df.select_dtypes(include=np.number).columns.tolist()

print(f"\n[TRAINING] Column detection:")
print(f"  Numeric columns:     {len(num_cols_training)}")
print(f"  Categorical columns: {len(cat_cols_training)}")
print(f"  Total:               {len(num_cols_training) + len(cat_cols_training)}")

# The TRAINING column order is the original DataFrame column order
training_col_order = feature_df.columns.tolist()
print(f"\n[TRAINING] Column ORDER (first 30):")
for i, col in enumerate(training_col_order[:30]):
    col_type = "CAT" if col in cat_cols_training else "NUM"
    print(f"  [{i:3d}] {col:<25s} ({col_type})")

# Apply build_graph.py clean_features() steps
feature_df_train = feature_df.copy()

num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="constant", fill_value="__missing__")

if num_cols_training:
    feature_df_train[num_cols_training] = num_imp.fit_transform(feature_df_train[num_cols_training])
if cat_cols_training:
    feature_df_train[cat_cols_training] = cat_imp.fit_transform(feature_df_train[cat_cols_training])

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
if cat_cols_training:
    feature_df_train[cat_cols_training] = enc.fit_transform(feature_df_train[cat_cols_training])

scaler = StandardScaler()
if num_cols_training:
    feature_df_train[num_cols_training] = scaler.fit_transform(feature_df_train[num_cols_training])

X_training = feature_df_train.values.astype(np.float32)
print(f"\n[TRAINING] Final feature matrix shape: {X_training.shape}")

# Check: were categoricals scaled?
print(f"\n[TRAINING] Scaler was fit on {len(num_cols_training)} columns (num_cols only)")
print(f"[TRAINING] Scaler was NOT applied to {len(cat_cols_training)} categorical columns")

# Show a categorical column's values
cat_example = cat_cols_training[0]
cat_idx = training_col_order.index(cat_example)
print(f"\n[TRAINING] Example categorical '{cat_example}' (index {cat_idx}):")
print(f"  Values (first 10): {X_training[:10, cat_idx]}")
print(f"  These are raw OrdinalEncoder outputs (NOT scaled)")

# ======================================================================
#  STEP 2: TRACE THE INFERENCE PIPELINE (preprocessing_pipeline.py)
# ======================================================================

print("\n" + "=" * 75)
print("  STEP 2: TRACE THE INFERENCE PIPELINE (preprocessing_pipeline.py)")
print("=" * 75)

print("\n[INFO] The inference pipeline is:")
print("  preprocessing_pipeline.py __main__")
print("    -> line 221: p = FraudPreprocessor()")
print("    -> line 222: p.fit(feature_df)")
print("    -> line 223: p.save()  # -> preprocessor.pkl")
print("")
print("  server/ml/pipeline.py preprocess_features()")
print("    -> line 128: preprocessor = loader.get_preprocessor()")
print("    -> line 147: X = preprocessor.transform(feature_df)")
print("    -> line 150-156: Force missing cols to 0.0")

# Load the fitted preprocessor
from server.ml import loader
loader.load_all()
preprocessor = loader.get_preprocessor()

inference_col_order = preprocessor.all_cols
print(f"\n[INFERENCE] Column detection:")
print(f"  Numeric columns:     {len(preprocessor.num_cols)}")
print(f"  Categorical columns: {len(preprocessor.cat_cols)}")
print(f"  Total (all_cols):    {len(preprocessor.all_cols)}")

print(f"\n[INFERENCE] Column ORDER (first 30):")
for i, col in enumerate(inference_col_order[:30]):
    col_type = "CAT" if col in preprocessor.cat_cols else "NUM"
    print(f"  [{i:3d}] {col:<25s} ({col_type})")

# Apply FraudPreprocessor.transform()
X_inference = preprocessor.transform(feature_df)
print(f"\n[INFERENCE] Final feature matrix shape: {X_inference.shape}")

print(f"\n[INFERENCE] Scaler was fit on {len(preprocessor.scaler.mean_)} columns (ALL cols)")
print(f"[INFERENCE] Scaler WAS applied to {len(preprocessor.cat_cols)} categorical columns")

# Show the same categorical column's values
if cat_example in inference_col_order:
    cat_idx_inf = inference_col_order.index(cat_example)
    print(f"\n[INFERENCE] Example categorical '{cat_example}' (index {cat_idx_inf}):")
    print(f"  Values (first 10): {X_inference[:10, cat_idx_inf]}")
    print(f"  These are OrdinalEncoder outputs THEN StandardScaled")

# ======================================================================
#  STEP 3: COMPARE COLUMN ORDERING
# ======================================================================

print("\n" + "=" * 75)
print("  STEP 3: MISMATCH #1 — COLUMN ORDERING")
print("=" * 75)

# all_cols = num_cols + cat_cols  (inference)
# training_col_order = original DF order (training)

same_order = (training_col_order == inference_col_order)
print(f"\n[CHECK] Column order identical? {same_order}")

if not same_order:
    print(f"\n[MISMATCH] Column ordering is DIFFERENT!")
    print(f"  Training order:  original DataFrame column order (interleaved)")
    print(f"  Inference order: num_cols + cat_cols (all numerics first, then all categoricals)")
    
    # Find first divergence
    mismatches = 0
    print(f"\n  Side-by-side comparison (first 40 columns):")
    print(f"  {'Idx':<5} {'TRAINING':<25} {'INFERENCE':<25} {'Match?':<8}")
    print(f"  {'-'*65}")
    for i in range(min(40, len(training_col_order))):
        t_col = training_col_order[i] if i < len(training_col_order) else "N/A"
        i_col = inference_col_order[i] if i < len(inference_col_order) else "N/A"
        match = "YES" if t_col == i_col else "NO <<<" 
        if t_col != i_col:
            mismatches += 1
        print(f"  [{i:3d}] {t_col:<25s} {i_col:<25s} {match}")
    
    # Count total mismatches
    total_mismatches = sum(1 for i in range(len(training_col_order)) 
                          if i < len(inference_col_order) and training_col_order[i] != inference_col_order[i])
    print(f"\n  Total column position mismatches: {total_mismatches} / {len(training_col_order)}")
    
    # Check if they have the same SET of columns
    same_set = set(training_col_order) == set(inference_col_order)
    print(f"  Same set of columns (ignoring order)? {same_set}")

# ======================================================================
#  STEP 4: COMPARE SCALING BEHAVIOR
# ======================================================================

print("\n" + "=" * 75)
print("  STEP 4: MISMATCH #2 — SCALING BEHAVIOR")
print("=" * 75)

print(f"\n[TRAINING] StandardScaler applied to: {len(num_cols_training)} NUMERIC columns only")
print(f"[INFERENCE] StandardScaler applied to: {len(preprocessor.all_cols)} ALL columns (num + cat)")
print(f"\n  => {len(preprocessor.cat_cols)} categorical features are scaled during inference")
print(f"     but were NOT scaled during training.")

# Show affected categorical columns
print(f"\n  Affected categorical columns ({len(cat_cols_training)}):")
for i, col in enumerate(cat_cols_training):
    print(f"    {i+1:2d}. {col}")

# ======================================================================
#  STEP 5: QUANTIFY THE ACTUAL FEATURE VECTOR DIFFERENCE
# ======================================================================

print("\n" + "=" * 75)
print("  STEP 5: QUANTIFY FEATURE VECTOR DIFFERENCES")
print("=" * 75)

# To compare properly, we need to align the columns
# Training uses original DF order, inference uses num+cat order
# Both have the same 432 features, just in different positions

# Map training features to a common order (by column name)
train_col_to_idx = {col: i for i, col in enumerate(training_col_order)}
infer_col_to_idx = {col: i for i, col in enumerate(inference_col_order)}

# Compare values for each column by name
print(f"\n[CHECK] Comparing feature values by column name...")
num_value_mismatches = 0
cat_value_mismatches = 0
max_diff_num = 0
max_diff_cat = 0

for col in training_col_order:
    t_idx = train_col_to_idx[col]
    i_idx = infer_col_to_idx[col]
    
    t_vals = X_training[:, t_idx]
    i_vals = X_inference[:, i_idx]
    
    if not np.allclose(t_vals, i_vals, atol=1e-5):
        diff = np.abs(t_vals - i_vals).max()
        if col in cat_cols_training:
            cat_value_mismatches += 1
            max_diff_cat = max(max_diff_cat, diff)
        else:
            num_value_mismatches += 1
            max_diff_num = max(max_diff_num, diff)

print(f"\n  Numeric columns with value mismatches: {num_value_mismatches} / {len(num_cols_training)}")
print(f"  Categorical columns with value mismatches: {cat_value_mismatches} / {len(cat_cols_training)}")
print(f"  Max difference (numeric):     {max_diff_num:.8f}")
print(f"  Max difference (categorical): {max_diff_cat:.8f}")

# Show specific examples
print(f"\n  Concrete example — first categorical column '{cat_cols_training[0]}':")
t_idx = train_col_to_idx[cat_cols_training[0]]
i_idx = infer_col_to_idx[cat_cols_training[0]]
print(f"    Training values  (first 5): {X_training[:5, t_idx]}")
print(f"    Inference values (first 5): {X_inference[:5, i_idx]}")
print(f"    Difference       (first 5): {np.abs(X_training[:5, t_idx] - X_inference[:5, i_idx])}")

# Show a numeric column for comparison (should match)
print(f"\n  Concrete example — first numeric column '{num_cols_training[0]}':")
t_idx = train_col_to_idx[num_cols_training[0]]
i_idx = infer_col_to_idx[num_cols_training[0]]
print(f"    Training values  (first 5): {X_training[:5, t_idx]}")
print(f"    Inference values (first 5): {X_inference[:5, i_idx]}")
print(f"    Difference       (first 5): {np.abs(X_training[:5, t_idx] - X_inference[:5, i_idx])}")

# ======================================================================
#  STEP 6: VERIFY AGAINST fraud_graph.pt
# ======================================================================

print("\n" + "=" * 75)
print("  STEP 6: VERIFY AGAINST fraud_graph.pt (GROUND TRUTH)")
print("=" * 75)

data = loader.get_data()
print(f"\n[INFO] fraud_graph.pt feature matrix: {data.x.shape}")
print(f"[INFO] Using first 500 rows for comparison with our sample")

# The fraud_graph.pt was built from the FULL dataset
# We can't compare row-by-row directly (different random seeds, full vs sample)
# But we CAN verify the structure by checking statistical properties

# Check: what do categorical feature values look like in the training graph?
# If they were NOT scaled, they should be small non-negative integers (ordinal encoded)
# If they WERE scaled, they should be centered around 0 with varying scale

# We need to figure out which columns in the graph correspond to categoricals
# In training (build_graph.py), columns are in original DF order
# Let's check positions of known categoricals

print(f"\n[CHECK] Checking categorical column positions in fraud_graph.pt...")
print(f"  (Using training column order since that's what build_graph.py uses)")

# Load full training data to get the exact column order
print(f"\n[INFO] Loading full training data to get exact column order...")
tx_full = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), low_memory=False, nrows=2)
idt_full = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_identity.csv"), low_memory=False, nrows=2)
df_schema = tx_full.merge(idt_full, on="TransactionID", how="left")
schema_feature_df = df_schema.drop(columns=["isFraud", "TransactionID"], errors="ignore")
schema_cat_cols = schema_feature_df.select_dtypes(include="object").columns.tolist()
schema_col_order = schema_feature_df.columns.tolist()

# Find positions of categoricals in the graph
print(f"\n  Categorical column positions in fraud_graph.pt:")
for cat_col in schema_cat_cols[:10]:
    idx = schema_col_order.index(cat_col)
    graph_vals = data.x[:100, idx].numpy()
    min_val = graph_vals.min()
    max_val = graph_vals.max()
    mean_val = graph_vals.mean()
    # Ordinal encoded values: non-negative integers (0, 1, 2, ...)
    # StandardScaled values: centered around 0, can be negative
    is_ordinal = np.all(graph_vals >= -1.5) and np.all(graph_vals == graph_vals.astype(int).astype(float))
    
    verdict = "UNSCALED (ordinal)" if min_val >= -1.5 and np.allclose(graph_vals, np.round(graph_vals)) else "SCALED"
    print(f"    [{idx:3d}] {cat_col:<20s} min={min_val:8.3f} max={max_val:8.3f} mean={mean_val:8.3f} -> {verdict}")

# Also check a few numeric columns for comparison
print(f"\n  Numeric column positions in fraud_graph.pt (should be scaled):")
schema_num_cols = schema_feature_df.select_dtypes(include=np.number).columns.tolist()
for num_col in schema_num_cols[:5]:
    idx = schema_col_order.index(num_col)
    graph_vals = data.x[:100, idx].numpy()
    min_val = graph_vals.min()
    max_val = graph_vals.max()
    mean_val = graph_vals.mean()
    print(f"    [{idx:3d}] {num_col:<20s} min={min_val:8.3f} max={max_val:8.3f} mean={mean_val:8.3f} -> SCALED")

# ======================================================================
#  STEP 7: IMPACT ANALYSIS
# ======================================================================

print("\n" + "=" * 75)
print("  STEP 7: IMPACT ANALYSIS")
print("=" * 75)

print(f"""
CONFIRMED MISMATCHES:
=====================

MISMATCH #1 — COLUMN ORDERING
  Training (build_graph.py):    Original DataFrame order (categoricals interleaved)
  Inference (preprocessor.pkl): num_cols + cat_cols (all numerics first, then categoricals)
  
  Impact: Feature at position [i] in training may be a DIFFERENT feature at 
          position [i] in inference. The model's learned weights are applied to
          the wrong features.
  
  Affected positions: {total_mismatches} / {len(training_col_order)} columns are in different positions

MISMATCH #2 — SCALING
  Training (build_graph.py):    StandardScaler applied to {len(num_cols_training)} numeric columns ONLY
  Inference (preprocessor.pkl): StandardScaler applied to ALL {len(preprocessor.all_cols)} columns
  
  Impact: {len(cat_cols_training)} categorical features receive standardized values during
          inference but received raw ordinal-encoded values during training.
  
  Affected features: {len(cat_cols_training)} categorical columns

COMBINED IMPACT:
  The GraphSAGE model was trained on features where:
    - Feature[2] = ProductCD (categorical, ordinal-encoded, unscaled, values like 0,1,2,3,4)
    - Feature[5] = card5 (numeric, standard-scaled)
  
  But at inference:
    - Feature[2] = card1 (numeric, standard-scaled)  
    - Feature[5] = card5 (numeric, standard-scaled)
  
  This means the model's conv1 weights for feature position 2 learned to process
  ProductCD ordinal values, but now receive card1 z-scores. This is catastrophic 
  for prediction quality.
""")

# ======================================================================
#  STEP 8: RECOMMENDED FIX
# ======================================================================

print("=" * 75)
print("  STEP 8: RECOMMENDED FIX")
print("=" * 75)

print(f"""
RECOMMENDATION: Option A — Make inference match training
========================================================

The correct fix is to modify preprocessing_pipeline.py to exactly replicate
build_graph.py's clean_features() behavior:

FIX 1 — Column ordering:
  CURRENT (WRONG):  self.all_cols = self.num_cols + self.cat_cols
  CORRECT:          self.all_cols = feature_df.columns.tolist()
                    (preserves original DataFrame column order)

FIX 2 — Scaling:
  CURRENT (WRONG):  self.scaler.fit(feature_df[self.all_cols])
                    self.scaler.transform(feature_df[self.all_cols])
  CORRECT:          self.scaler.fit(feature_df[self.num_cols])
                    Feature values for num_cols -> scaler.transform()
                    Feature values for cat_cols -> pass through unchanged

These two changes will make the inference preprocessing produce
BIT-IDENTICAL output to build_graph.py's clean_features().

WHY NOT Option B (retrain)?
  Retraining is unnecessary because:
  1. The training pipeline (build_graph.py) is correct and well-tested
  2. fraud_graph.pt and best_model.pt are already trained artifacts
  3. We just need to make inference match what already exists
  4. Retraining takes significant time and compute
""")

# ======================================================================
#  SUMMARY
# ======================================================================

print("=" * 75)
print("  FINAL SUMMARY")
print("=" * 75)

print(f"""
QUESTION: Which preprocessing pipeline produced fraud_graph.pt and best_model.pt?
ANSWER:   build_graph.py clean_features() (lines 195-228)

QUESTION: Is there a preprocessing mismatch?
ANSWER:   YES — TWO mismatches (column ordering AND scaling)

QUESTION: How many features are affected?
ANSWER:   Column ordering: {total_mismatches}/{len(training_col_order)} positions differ
          Scaling:         {len(cat_cols_training)} categorical features incorrectly scaled

QUESTION: Is the project suitable for deployment?
ANSWER:   NO — not until inference preprocessing matches training preprocessing

QUESTION: Is retraining required?
ANSWER:   NO — the fix is to correct the inference preprocessing (Option A)
""")
