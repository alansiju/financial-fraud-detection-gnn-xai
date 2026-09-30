#!/usr/bin/env python3
"""
save_node_meta.py
=================
Saves the raw, human-readable columns for every graph node so the dashboard can
report real numbers.

Why this exists
---------------
`server/api/analytics.py` had no access to the original columns — `data.x` is
imputed, ordinal-encoded and standardised — so it invented its charts:

    fraud_by_month  : 0.03 + 0.02 * sin(i * pi / 6)
    fraud_by_device : rng.uniform(0.02, 0.08)
    fraud_by_card   : rng.uniform(0.02, 0.06)
    fraud_by_email  : rng.uniform(0.02, 0.05)
    amounts         : abs(x[:, 0]) * 500 + 10   # a standardised feature, not a price

Those are fabricated values rendered as analytics. This artifact replaces them
with the actual card types, devices, email domains, amounts and timestamps.

Output: outputs/models/node_meta.pkl   (row i == graph node i)

Run after build_graph.py:
    py save_node_meta.py
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from preprocessing_pipeline import load_raw_data  # noqa: E402

OUT_PATH = os.path.join(ROOT, "outputs", "models", "node_meta.pkl")

# Kept small on purpose: this is loaded into the API process and held in RAM.
META_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card4", "card6", "DeviceType", "DeviceInfo",
    "P_emaildomain", "R_emaildomain", "addr1", "isFraud",
]


def main() -> None:
    print("Reading training data ...", flush=True)
    t0 = time.time()
    df = load_raw_data(ROOT)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    cols = [c for c in META_COLS if c in df.columns]
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        print(f"[!] Not present in data: {missing}", flush=True)

    meta = df[cols].copy()

    # TransactionDT is a second offset from a fixed (undisclosed) start date.
    # Kaggle consensus puts it at 2017-12-01; it only needs to be consistent for
    # the month/hour breakdowns to be meaningful.
    ref = pd.Timestamp("2017-12-01")
    meta["timestamp"] = ref + pd.to_timedelta(meta["TransactionDT"], unit="s")
    meta["month"] = meta["timestamp"].dt.strftime("%b %Y")
    meta["hour"] = meta["timestamp"].dt.hour
    meta["dayofweek"] = meta["timestamp"].dt.day_name()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(meta, f, protocol=4)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nSaved -> {OUT_PATH}  ({size_mb:.1f} MB, {len(meta):,} rows)")
    print(f"Columns: {list(meta.columns)}")
    print(f"\nReal distributions now available, e.g. fraud rate by card type:")
    if "card4" in meta.columns:
        g = meta.groupby("card4")["isFraud"].agg(["size", "mean"])
        for name, row in g.iterrows():
            print(f"  {str(name):<20} n={int(row['size']):>7,}  "
                  f"fraud={100*row['mean']:.2f}%")


if __name__ == "__main__":
    main()
