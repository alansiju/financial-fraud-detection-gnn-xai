#!/usr/bin/env python3
"""
save_edge_attr_lookup.py
========================
Builds the inverted index the server uses to attach freshly uploaded
transactions to the training graph.

Output: outputs/models/edge_attr_lookup.pkl

    {
      "inverted_index": {attr: {value: [most-recent training node indices]}},
      "weights":        {attr: edge weight},
      "cols":           [...],
      "n_nodes":        590540,
      "transaction_ids": np.int32 array aligned with node indices,
      "k_temporal":     3,
      "max_group":      5000,
    }

What changed
------------
* ROOT was hardcoded to `D:\\projects\\GNN`; it now follows this file.

* It used to store *every* node index for each value, then the server dropped
  any value with more than 30 matches. That mirrored the old build rule, which
  is exactly the rule that produced zero edges on a real batch. The index now
  keeps the K most recent transactions per value — the same rule build_graph.py
  uses — so an uploaded row attaches to recent history instead of to nothing.

* Edge weights are stored alongside, so the server does not keep its own copy
  of the weight table.
"""

import os
import pickle
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from build_graph import K_TEMPORAL, MAX_GROUP_SIZE  # noqa: E402
from preprocessing_pipeline import EDGE_ATTRIBUTES, load_raw_data  # noqa: E402

OUT_PATH = os.path.join(ROOT, "outputs", "models", "edge_attr_lookup.pkl")


def main() -> None:
    print("Reading training data ...", flush=True)
    t0 = time.time()
    df = load_raw_data(ROOT)
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    dt = df["TransactionDT"].values.astype(np.int64)

    cols_present = [c for c in EDGE_ATTRIBUTES if c in df.columns]
    cols_missing = [c for c in EDGE_ATTRIBUTES if c not in df.columns]
    print(f"Edge attr cols found  : {cols_present}", flush=True)
    if cols_missing:
        print(f"Edge attr cols MISSING: {cols_missing}", flush=True)

    print(f"\nBuilding inverted index "
          f"(keep {K_TEMPORAL} most recent per value, "
          f"skip values with > {MAX_GROUP_SIZE:,} rows) ...", flush=True)
    t1 = time.time()

    inverted_index: dict = {}
    total_entries = total_skipped = 0

    for attr in cols_present:
        col = df[attr]
        valid = col.notna().to_numpy()
        pos = np.flatnonzero(valid)
        if pos.size == 0:
            inverted_index[attr] = {}
            continue

        vals = col.to_numpy()[pos]
        # Sort by time so the tail of each group is the most recent.
        order = np.argsort(dt[pos], kind="stable")
        pos = pos[order]
        vals = vals[order]

        attr_index: dict = {}
        skipped = 0
        # Group-by on the sorted values via a dict of running lists, keeping
        # only the last K. Bounded memory, single pass.
        counts: dict = {}
        for node_idx, v in zip(pos.tolist(), vals.tolist()):
            key = _normalise(v)
            counts[key] = counts.get(key, 0) + 1
            lst = attr_index.setdefault(key, [])
            lst.append(node_idx)
            if len(lst) > K_TEMPORAL:
                lst.pop(0)

        for key, n in counts.items():
            if n > MAX_GROUP_SIZE:
                attr_index.pop(key, None)
                skipped += 1

        inverted_index[attr] = attr_index
        total_entries += len(attr_index)
        total_skipped += skipped
        print(f"  [{attr:<16}] {len(attr_index):>8,} values kept | "
              f"{skipped:>4,} too generic", flush=True)

    print(f"\nTotal value groups: {total_entries:,}  "
          f"(skipped {total_skipped:,})", flush=True)
    print(f"Build time: {time.time()-t1:.1f}s", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "inverted_index": inverted_index,
            "weights": dict(EDGE_ATTRIBUTES),
            "cols": cols_present,
            "n_nodes": len(df),
            "transaction_ids": df["TransactionID"].values.astype(np.int32),
            "k_temporal": K_TEMPORAL,
            "max_group": MAX_GROUP_SIZE,
        }, f, protocol=4)

    print(f"\nSaved -> {OUT_PATH} "
          f"({os.path.getsize(OUT_PATH)/1e6:.1f} MB)", flush=True)

    for attr in ("card1", "addr1", "DeviceInfo"):
        if inverted_index.get(attr):
            k = next(iter(inverted_index[attr]))
            print(f"  sample {attr}[{k!r}] -> {inverted_index[attr][k]}", flush=True)


def _normalise(v):
    """
    Canonical dict key. Must match server/ml/pipeline.py exactly, or every
    numeric lookup silently misses and uploads attach to nothing.
    """
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if f != f:          # NaN
            return None
        if f.is_integer():
            return int(f)
        return f
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


if __name__ == "__main__":
    main()
