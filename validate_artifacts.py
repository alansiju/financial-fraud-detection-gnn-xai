#!/usr/bin/env python3
"""
validate_artifacts.py
=====================
Offline consistency check across every saved artifact. No server required.

The original failure mode of this project was artifacts that individually
loaded fine but disagreed with each other: the graph was built by one
preprocessing implementation and inference used a second, the edge lookup was
keyed by a rule the graph no longer used, and nothing checked that the model's
input width still matched the feature matrix. Each mismatch degraded
predictions silently.

Run after any rebuild:
    py validate_artifacts.py
"""

import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

GRAPH = os.path.join(ROOT, "fraud_graph.pt")
MODEL = os.path.join(ROOT, "outputs", "models", "best_model.pt")
PREPROC = os.path.join(ROOT, "outputs", "models", "preprocessor.pkl")
LOOKUP = os.path.join(ROOT, "outputs", "models", "edge_attr_lookup.pkl")
META = os.path.join(ROOT, "outputs", "models", "node_meta.pkl")

fails = []


def check(name, ok, detail="", info=""):
    """
    `info` is context worth showing either way (counts, versions).
    `detail` is the reason it failed, and is only printed when it did -- a
    failure-phrased string next to [PASS] reads as though the check failed.
    """
    line = "  [PASS] " if ok else "  [FAIL] "
    line += name
    if info:
        line += f"  -- {info}"
    if not ok and detail:
        line += f"  -- {detail}"
    print(line)
    if not ok:
        fails.append(f"{name}: {detail or info}")
    return ok


def main() -> int:
    print("=" * 72)
    print("  ARTIFACT CONSISTENCY")
    print("=" * 72)

    print("\n[presence]")
    for label, path in [("fraud_graph.pt", GRAPH), ("best_model.pt", MODEL),
                        ("preprocessor.pkl", PREPROC),
                        ("edge_attr_lookup.pkl", LOOKUP),
                        ("node_meta.pkl", META)]:
        if not check(f"{label} exists", os.path.exists(path), path):
            return 1

    print("\n[loading]")
    data = torch.load(GRAPH, weights_only=False, map_location="cpu")
    ckpt = torch.load(MODEL, weights_only=False, map_location="cpu")
    from preprocessing_pipeline import FraudPreprocessor  # noqa: F401
    pre = pickle.load(open(PREPROC, "rb"))
    lookup = pickle.load(open(LOOKUP, "rb"))
    meta = pickle.load(open(META, "rb"))
    print(f"  graph  : {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} "
          f"directed edges, {data.x.shape[1]} features")
    print(f"  model  : in_channels={ckpt.get('in_channels')}, "
          f"epoch={ckpt.get('epoch')}, threshold={ckpt.get('threshold')}")

    n = data.num_nodes

    # ── Feature width agrees everywhere ────────────────────────────────────
    print("\n[feature width]")
    check("model in_channels == graph feature count",
          ckpt.get("in_channels") == data.x.shape[1],
          info=f"{ckpt.get('in_channels')} vs {data.x.shape[1]}")
    check("preprocessor n_features == graph feature count",
          pre.n_features == data.x.shape[1],
          info=f"{pre.n_features} vs {data.x.shape[1]}")
    check("feature_names length matches",
          len(getattr(data, "feature_names", [])) == data.x.shape[1],
          info=f"{len(getattr(data, 'feature_names', []))} vs {data.x.shape[1]}")
    check("preprocessor column order matches the graph's",
          list(pre.all_cols) == list(getattr(data, "feature_names", [])),
          "all_cols and data.feature_names differ")

    # ── Model actually loads into the architecture the config describes ────
    print("\n[model]")
    from models.graphsage import GraphSAGE
    cfg = ckpt.get("config", {})
    model = GraphSAGE(
        in_channels=ckpt["in_channels"],
        hidden1=cfg.get("hidden1", 256), hidden2=cfg.get("hidden2", 128),
        hidden3=cfg.get("hidden3", 64), mlp_hidden=cfg.get("mlp_hidden", 64),
        dropout1=cfg.get("dropout1", 0.2), dropout2=cfg.get("dropout2", 0.2),
        dropout3=cfg.get("dropout3", 0.2),
        num_layers=cfg.get("num_layers", 2), use_skip=cfg.get("use_skip", True),
    )
    try:
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=True), None
        check("state_dict loads strictly into the configured architecture", True)
    except RuntimeError as e:
        check("state_dict loads strictly into the configured architecture",
              False, str(e)[:160])

    check("checkpoint carries a tuned threshold",
          isinstance(ckpt.get("threshold"), float),
          info=f"threshold={ckpt.get('threshold')}")
    check("tuned threshold is not the old hardcoded 0.5",
          abs(float(ckpt.get("threshold", 0.5)) - 0.5) > 1e-9,
          "threshold is exactly 0.5, which is what broke prediction before")

    # ── Node-aligned artifacts ─────────────────────────────────────────────
    print("\n[node alignment]")
    check("edge lookup n_nodes == graph nodes", lookup.get("n_nodes") == n,
          info=f"{lookup.get('n_nodes')} vs {n}")
    check("transaction_ids length == graph nodes",
          len(lookup.get("transaction_ids", [])) == n,
          info=f"{len(lookup.get('transaction_ids', []))} vs {n}")
    check("node_meta rows == graph nodes", len(meta) == n, info=f"{len(meta)} vs {n}")
    check("node_meta TransactionIDs align with the lookup's",
          np.array_equal(meta["TransactionID"].values.astype(np.int64),
                         np.asarray(lookup["transaction_ids"]).astype(np.int64)),
          "row order differs between node_meta and edge_attr_lookup")
    check("node_meta labels match the graph's y",
          np.array_equal(meta["isFraud"].values.astype(np.int64),
                         data.y.numpy().astype(np.int64)),
          "isFraud disagrees with data.y")

    # ── Splits ─────────────────────────────────────────────────────────────
    print("\n[splits]")
    tr, va, te = (int(data.train_mask.sum()), int(data.val_mask.sum()),
                  int(data.test_mask.sum()))
    check("splits partition every node exactly once", tr + va + te == n,
          info=f"{tr:,} + {va:,} + {te:,} = {tr+va+te:,} vs {n:,}")
    overlap = int((data.train_mask & data.val_mask).sum()
                  + (data.train_mask & data.test_mask).sum()
                  + (data.val_mask & data.test_mask).sum())
    check("splits do not overlap", overlap == 0, info=f"{overlap} nodes in two splits")

    split_mode = getattr(data, "split_mode", "unknown")
    print(f"  split mode: {split_mode}")
    if split_mode == "temporal":
        dt = meta["TransactionDT"].values
        tr_max = dt[data.train_mask.numpy()].max()
        va_min = dt[data.val_mask.numpy()].min()
        va_max = dt[data.val_mask.numpy()].max()
        te_min = dt[data.test_mask.numpy()].min()
        check("train ends before validation begins", tr_max < va_min,
              info=f"train max {tr_max:,} vs val min {va_min:,}")
        check("validation ends before test begins", va_max < te_min,
              info=f"val max {va_max:,} vs test min {te_min:,}")

    # ── Edge masks must not let training see the future ────────────────────
    print("\n[edge masks]")
    if hasattr(data, "train_edge_mask"):
        src, dst = data.edge_index[0], data.edge_index[1]
        tm = data.train_mask
        bad = int((data.train_edge_mask & ~(tm[src] & tm[dst])).sum())
        check("train_edge_mask contains only train-train edges", bad == 0,
              info=f"{bad:,} offending edges")
        seen = data.train_mask | data.val_mask
        bad_v = int((data.val_edge_mask & ~(seen[src] & seen[dst])).sum())
        check("val_edge_mask excludes test nodes", bad_v == 0,
              info=f"{bad_v:,} offending edges")
    else:
        check("graph carries edge masks", False, "rebuild with build_graph.py")

    # ── Graph structure ────────────────────────────────────────────────────
    print("\n[graph structure]")
    check("edge_attr length matches edge_index",
          data.edge_attr.shape[0] == data.edge_index.shape[1],
          info=f"{data.edge_attr.shape[0]} vs {data.edge_index.shape[1]}")
    check("edges are stored in both directions",
          data.edge_index.shape[1] % 2 == 0,
          info=f"{data.edge_index.shape[1]:,} directed edges")
    check("features are finite", bool(torch.isfinite(data.x).all()),
          "data.x contains NaN or inf")
    deg = torch.zeros(n)
    deg.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
    iso = int((deg == 0).sum())
    check("most nodes have at least one neighbour", iso < n * 0.5,
          info=f"{iso:,} isolated ({100*iso/n:.1f}%)")
    print(f"  average degree: {deg.mean():.2f}, isolated: {iso:,} "
          f"({100*iso/n:.1f}%)")

    # ── Serving reproduces training features exactly ───────────────────────
    print("\n[train/serve feature parity]")
    from preprocessing_pipeline import load_raw_data
    if os.path.exists(os.path.join(ROOT, "train_transaction.csv")):
        df = load_raw_data(ROOT)
        idx = np.sort(np.random.default_rng(0).choice(len(df), 500, replace=False))
        X_serve = pre.transform(df.iloc[idx])
        diff = float(np.abs(data.x.numpy()[idx] - X_serve).max())
        check("serving preprocessor reproduces data.x exactly", diff < 1e-5,
              info=f"max abs diff {diff:.3e}")

        # The check above feeds the in-memory frame, which already has the
        # training dtypes. A real upload arrives as CSV text and is re-parsed
        # (TransactionAmt as float64, not float32), which used to shift
        # TransactionAmt_decimal and move LightGBM scores by up to 7e-3 —
        # invisible to the in-memory check. Round-trip through CSV explicitly.
        import io
        import pandas as pd
        rt = pd.read_csv(io.StringIO(df.iloc[idx].to_csv(index=False)), low_memory=False)
        diff_csv = float(np.abs(data.x.numpy()[idx] - pre.transform(rt)).max())
        check("CSV-uploaded rows reproduce data.x exactly", diff_csv < 1e-5,
              info=f"max abs diff {diff_csv:.3e}")
    else:
        print("  [SKIP] raw CSVs absent")

    # ── Lookup sanity ──────────────────────────────────────────────────────
    print("\n[edge lookup]")
    inv = lookup.get("inverted_index", {})
    k = lookup.get("k_temporal")
    check("lookup stores the temporal rule used by the graph", k is not None,
          info=f"k_temporal={k}")
    if inv and k:
        too_long = sum(1 for attr in inv.values()
                       for v in attr.values() if len(v) > k)
        check(f"no lookup entry exceeds k={k}", too_long == 0,
              info=f"{too_long} entries longer than {k}")
        max_idx = max((max(v) for attr in inv.values() for v in attr.values()
                       if v), default=-1)
        check("lookup node indices are in range", max_idx < n,
              info=f"max index {max_idx:,} vs {n:,} nodes")

    print("\n" + "=" * 72)
    print(f"  {'ALL CONSISTENT' if not fails else str(len(fails)) + ' INCONSISTENCIES'}")
    print("=" * 72)
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
