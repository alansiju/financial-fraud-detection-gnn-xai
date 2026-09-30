"""
=============================================================================
 SUPERSEDED -- this script does not produce a valid measurement.

 Reason: scores the first 2000 training rows.

 Use verify_serving.py instead. It scores real rows from the held-out test
 split through the actual serving pipeline.

 Kept for reference only. Its hardcoded absolute paths have been
 repointed at the repo root so it at least runs, but do not trust its output.
=============================================================================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from server.ml.pipeline import run_full_pipeline
from server.ml import loader

print("Loading ML stack...")
loader.load_all()

print("Loading 2000 rows...")
tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), nrows=2000, low_memory=False)
idt = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_identity.csv"), low_memory=False)
df = tx.merge(idt, on="TransactionID", how="left")
print(f"DataFrame shape: {df.shape}")

print("Running pipeline...")
try:
    res = run_full_pipeline(df)
    print("Success! Processed rows:", res["total_rows"])
except Exception as e:
    import traceback
    traceback.print_exc()
