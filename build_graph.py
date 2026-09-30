#!/usr/bin/env python3
"""
build_graph.py
==============
Turns the IEEE-CIS Fraud Detection dataset into a weighted, undirected
PyTorch Geometric `Data` object for GraphSAGE training.

Pipeline
--------
1. Extract  : unzip ieee-fraud-detection.zip (skipped if the CSVs are present)
2. Merge    : train_transaction.csv  LEFT JOIN  train_identity.csv
3. Split    : chronological 70/15/15 by TransactionDT  (see WHY below)
4. Clean    : FraudPreprocessor fitted on TRAIN ROWS ONLY
5. Edges    : weighted multi-relation graph, temporal k-nearest within groups
6. Masks    : node masks + edge masks that stop training from seeing the future
7. Save     : fraud_graph.pt + outputs/models/preprocessor.pkl
8. Stats    : nodes, edges, degree, components, density, isolated-node count

What changed vs. the previous version
-------------------------------------
* DATA_DIR was hardcoded to `d:\\projects\\GNN`. It is now the directory this
  file lives in, so the build runs wherever the repo is checked out.

* The split was a stratified *shuffle*. IEEE-CIS is a time series, so a random
  split lets the model interpolate between transactions that bracket a test
  row in time — performance that cannot exist in production. The split is now
  chronological by default (`SPLIT_MODE`).

* Preprocessing was `fit_transform`-ed on all 590,540 rows, so imputation
  medians, ordinal categories and scaler statistics were learned partly from
  the test set. It is now fit on training rows only.

* Feature cleaning was a second, independent implementation of what the server
  does at inference time. It now calls the same FraudPreprocessor the server
  loads, so the two cannot drift.

* GROUP_CAP silently deleted every attribute group with more than 30 members —
  which is precisely what a fraud ring or a card-testing burst looks like. On a
  2,500-row batch this produced literally zero intra-batch edges. Groups are
  now linked by temporal adjacency (each transaction to its K most recent
  predecessors sharing that attribute), which is O(k) instead of O(k^2) and
  keeps the large groups.

* Edge weights are now normalised and actually consumed by the model.

Usage
-----
    py build_graph.py                 # chronological split (recommended)
    py build_graph.py --split random  # old behaviour, for comparison
"""

import argparse
import os
import sys
import time
import warnings
import zipfile

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sc_connected_components
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from preprocessing_pipeline import (
    EDGE_ATTRIBUTES,
    FraudPreprocessor,
    PIPELINE_PATH,
    load_raw_data,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE = os.path.join(DATA_DIR, "ieee-fraud-detection.zip")
OUTPUT_GRAPH = os.path.join(DATA_DIR, "fraud_graph.pt")
OUTPUT_IMG = os.path.join(DATA_DIR, "subgraph.png")

# ── Edge construction ──────────────────────────────────────────────────────
K_TEMPORAL = 3
"""
Each transaction is linked to its K most recent predecessors that share an
attribute value. Small K keeps the graph sparse and the edges meaningful:
consecutive charges on one card are the signal; two charges a year apart that
happen to share a billing zip are noise.
"""

MAX_GROUP_SIZE = 5000
"""
Attribute values shared by more than this many transactions carry no identifying
information (e.g. P_emaildomain == 'gmail.com' covers ~230k rows) and are
skipped. Unlike the old GROUP_CAP=30 this only discards genuinely generic
values, not fraud rings.
"""

MIN_EDGE_WEIGHT = 1.0

# ── Split ──────────────────────────────────────────────────────────────────
SPLIT_MODE = "temporal"        # "temporal" | "random"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SUBGRAPH_NODES = 500
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_header(split_mode: str) -> None:
    bar = "=" * 62
    print(f"\n{bar}")
    print("  IEEE-CIS Fraud Detection -> PyTorch Geometric Graph")
    print(bar)
    print(f"  PyTorch    : {torch.__version__}")
    if device.type == "cuda":
        print(f"  Device     : CUDA - {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("  Device     : CPU")
    print(f"  Split mode : {split_mode}")
    print(f"  Edge rule  : each tx -> {K_TEMPORAL} most recent peers per attribute")
    print(f"  Group skip : attribute values shared by > {MAX_GROUP_SIZE:,} rows")
    print(f"{bar}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — EXTRACT
# ═══════════════════════════════════════════════════════════════════════════

NEEDED_FILES = ["train_transaction.csv", "train_identity.csv"]


def extract_data() -> None:
    missing = [f for f in NEEDED_FILES
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if not missing:
        print("[1/8] Data files already extracted - skipping.")
        return
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(
            f"Missing {missing} and no archive at {ZIP_FILE}"
        )
    print(f"[1/8] Extracting {os.path.basename(ZIP_FILE)} ...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        for member in z.namelist():
            base = os.path.basename(member)
            if base in NEEDED_FILES:
                with z.open(member) as src, \
                     open(os.path.join(DATA_DIR, base), "wb") as tgt:
                    tgt.write(src.read())
                print(f"    Extracted -> {base}")
    print(f"    Done ({time.time()-t0:.1f}s)\n")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def make_masks(
    y: np.ndarray,
    dt: np.ndarray,
    mode: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """
    Returns boolean train/val/test masks over the N nodes.

    temporal : oldest `train_ratio` of transactions train, next `val_ratio`
               validate, most recent block tests. This is the only split that
               answers the question the product actually asks — "will this
               model catch tomorrow's fraud".
    random   : the previous stratified shuffle, kept so the two can be compared.
    """
    N = len(y)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    if mode == "temporal":
        order = np.argsort(dt, kind="stable")
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        train_mask[order[:n_train]] = True
        val_mask[order[n_train:n_train + n_val]] = True
        test_mask[order[n_train + n_val:]] = True

        cut1, cut2 = dt[order[n_train]], dt[order[n_train + n_val]]
        print(f"    Chronological cut points: TransactionDT {cut1:,} / {cut2:,}")
        print(f"    Fraud rate  train={y[train_mask.numpy()].mean()*100:.2f}%  "
              f"val={y[val_mask.numpy()].mean()*100:.2f}%  "
              f"test={y[test_mask.numpy()].mean()*100:.2f}%")
        print("    (fraud rate drifting across the splits is real concept drift, "
              "not a bug)")
    elif mode == "random":
        idx = np.arange(N)
        idx_train, idx_rem, _, y_rem = train_test_split(
            idx, y, test_size=1.0 - train_ratio, stratify=y, random_state=seed)
        val_share = val_ratio / (1.0 - train_ratio)
        idx_val, idx_test = train_test_split(
            idx_rem, test_size=1.0 - val_share, stratify=y_rem, random_state=seed)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True
    else:
        raise ValueError(f"unknown split mode: {mode}")

    return train_mask, val_mask, test_mask


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5 — EDGE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_edges(
    df_edge: pd.DataFrame,
    dt: np.ndarray,
    edge_attrs: dict,
    k_temporal: int = K_TEMPORAL,
    max_group: int = MAX_GROUP_SIZE,
):
    """
    Weighted undirected edge set.

    For each attribute, transactions are grouped by shared value, ordered in
    time, and each one is linked to its `k_temporal` most recent predecessors
    in that group. Weights are additive across attributes: two transactions on
    the same card AND the same device score 3.0 + 3.0 = 6.0.

    Fully vectorised — the previous nested `combinations()` loop was the
    slowest part of the build.

    Returns
    -------
    edge_index  : LongTensor  [2, 2E]
    edge_weight : FloatTensor [2E]  (raw additive weights)
    """
    print(f"[5/8] Building edges (k={k_temporal}, max_group={max_group:,}) ...")
    t0 = time.time()
    N = len(df_edge)

    src_parts, dst_parts, w_parts = [], [], []

    for attr, weight in edge_attrs.items():
        if attr not in df_edge.columns:
            print(f"    [!] '{attr}' not in DataFrame - skipped")
            continue

        col = df_edge[attr]
        valid_pos = np.flatnonzero(col.notna().to_numpy())
        if valid_pos.size == 0:
            print(f"    [!] '{attr}' is all-NaN - skipped")
            continue

        codes, _ = pd.factorize(col.iloc[valid_pos], sort=False)

        # Drop values that are too generic to identify anybody.
        counts = np.bincount(codes)
        keep = counts[codes] <= max_group
        n_skipped_groups = int((counts > max_group).sum())
        codes = codes[keep]
        pos = valid_pos[keep]
        if codes.size == 0:
            print(f"    [{attr:<16}] every group exceeded max_group - skipped")
            continue

        # Order by (group, time) so that within a group, neighbours in the
        # sorted array are neighbours in time.
        order = np.lexsort((dt[pos], codes))
        s_codes = codes[order]
        s_pos = pos[order]

        attr_edges = 0
        for lag in range(1, k_temporal + 1):
            if s_codes.size <= lag:
                break
            same = s_codes[lag:] == s_codes[:-lag]
            if not same.any():
                continue
            a = s_pos[lag:][same]
            b = s_pos[:-lag][same]
            src_parts.append(a)
            dst_parts.append(b)
            w_parts.append(np.full(a.size, weight, dtype=np.float32))
            attr_edges += a.size

        print(f"    [{attr:<16}] w={weight:.0f} | "
              f"linked={pos.size:>8,} rows | "
              f"generic groups skipped={n_skipped_groups:>5,} | "
              f"pairs={attr_edges:>9,}")

    if not src_parts:
        print("    [!] No edges were created.")
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    wts = np.concatenate(w_parts)

    # Collapse duplicates, summing weights across attributes.
    lo = np.minimum(src, dst).astype(np.int64)
    hi = np.maximum(src, dst).astype(np.int64)
    keys = lo * np.int64(N) + hi
    uniq_keys, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(inverse, weights=wts.astype(np.float64)).astype(np.float32)

    u_lo = (uniq_keys // N).astype(np.int64)
    u_hi = (uniq_keys % N).astype(np.int64)

    keep = summed >= MIN_EDGE_WEIGHT
    u_lo, u_hi, summed = u_lo[keep], u_hi[keep], summed[keep]

    print(f"\n    -- Summary ------------------------------------------------")
    print(f"    Raw attribute pairs  : {len(keys):>10,}")
    print(f"    Unique undirected    : {len(uniq_keys):>10,}")
    print(f"    After weight filter  : {len(summed):>10,} (>= {MIN_EDGE_WEIGHT})")
    print(f"    Weight range         : {summed.min():.1f} .. {summed.max():.1f}")
    print(f"    Build time           : {time.time()-t0:>9.1f}s\n")

    src_bi = np.concatenate([u_lo, u_hi])
    dst_bi = np.concatenate([u_hi, u_lo])
    w_bi = np.concatenate([summed, summed])

    edge_index = torch.tensor(np.stack([src_bi, dst_bi]), dtype=torch.long)
    edge_weight = torch.tensor(w_bi, dtype=torch.float)
    return edge_index, edge_weight


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7 — STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def print_stats(data: Data) -> None:
    N = data.x.shape[0]
    E_dir = data.edge_index.shape[1]
    E = E_dir // 2

    deg = torch.zeros(N, dtype=torch.float)
    if E_dir:
        deg.scatter_add_(0, data.edge_index[0], torch.ones(E_dir))
    isolated = int((deg == 0).sum().item())

    density = (2 * E) / (N * (N - 1)) if N > 1 else 0.0

    if E_dir:
        adj = csr_matrix(
            (np.ones(E_dir, dtype=np.float32),
             (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(N, N))
        n_comp, _ = sc_connected_components(adj, directed=False)
    else:
        n_comp = N

    bar = "=" * 58
    print("\n[7/8] Graph Statistics")
    print(bar)
    print(f"  Nodes                  : {N:>12,}")
    print(f"  Edges (undirected)     : {E:>12,}")
    print(f"  Node features          : {data.x.shape[1]:>12,}")
    print(f"  Average node degree    : {deg.mean().item():>12.4f}")
    print(f"  Median node degree     : {int(deg.median().item()):>12,}")
    print(f"  Max node degree        : {int(deg.max().item()):>12,}")
    print(f"  Isolated nodes         : {isolated:>12,}  "
          f"({100*isolated/N:.2f}%)")
    print(f"  Connected components   : {n_comp:>12,}")
    print(f"  Graph density          : {density:>12.8f}")
    print(f"  Fraud rate             : {data.y.float().mean().item()*100:>11.3f}%")
    print(f"  Train nodes            : {int(data.train_mask.sum()):>12,}")
    print(f"  Val nodes              : {int(data.val_mask.sum()):>12,}")
    print(f"  Test nodes             : {int(data.test_mask.sum()):>12,}")
    print(f"  Train-only edges       : {int(data.train_edge_mask.sum())//2:>12,}")
    print(bar + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 8 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_subgraph(data: Data, n_sample: int, out_path: str, seed: int) -> None:
    print(f"[8/8] Visualising subgraph ({n_sample} nodes) ...")
    t0 = time.time()
    N = data.x.shape[0]
    E_dir = data.edge_index.shape[1]
    if E_dir == 0:
        print("    [!] No edges - skipping visualisation.")
        return

    deg = torch.zeros(N, dtype=torch.float)
    deg.scatter_add_(0, data.edge_index[0], torch.ones(E_dir))
    top_idx = set(deg.argsort(descending=True)[:n_sample].tolist())

    ei = data.edge_index.numpy()
    ew = data.edge_attr.numpy()
    mask = np.fromiter(
        (s in top_idx and d in top_idx for s, d in zip(ei[0], ei[1])),
        dtype=bool, count=ei.shape[1])

    G = nx.Graph()
    G.add_nodes_from(top_idx)
    for s, d, w in zip(ei[0][mask], ei[1][mask], ew[mask]):
        if int(s) < int(d):
            G.add_edge(int(s), int(d), weight=float(w))

    if len(G.edges()) == 0:
        print("    [!] No edges in subgraph - skipping.")
        return

    labels = data.y.numpy()
    nodes = list(G.nodes())
    colors = ["#e74c3c" if labels[n] == 1 else "#3498db" for n in nodes]
    edges = list(G.edges())
    widths = [G[u][v]["weight"] * 0.25 for u, v in edges]

    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    pos = nx.spring_layout(G, seed=seed, k=2.0 / np.sqrt(max(len(nodes), 1)))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=widths,
                           edge_color="#8ab4f8", alpha=0.20)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=colors,
                           node_size=18, alpha=0.90)
    ax.legend(handles=[mpatches.Patch(color="#e74c3c", label="Fraudulent"),
                       mpatches.Patch(color="#3498db", label="Legitimate")],
              loc="upper right", fontsize=13, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="white", framealpha=0.9)
    ax.set_title(f"IEEE-CIS Fraud Detection - Subgraph\n"
                 f"{len(nodes)} nodes | {len(G.edges())} edges | k={K_TEMPORAL}",
                 color="white", fontsize=14, pad=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved -> {out_path}  ({time.time()-t0:.1f}s)\n")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["temporal", "random"], default=SPLIT_MODE)
    ap.add_argument("--k", type=int, default=K_TEMPORAL)
    ap.add_argument("--max-group", type=int, default=MAX_GROUP_SIZE)
    args = ap.parse_args()

    _print_header(args.split)
    t_start = time.time()

    extract_data()

    print("[2/8] Loading and merging CSVs ...")
    t0 = time.time()
    df = load_raw_data(DATA_DIR)
    print(f"    Merged: {len(df):,} rows x {df.shape[1]} cols "
          f"({time.time()-t0:.1f}s)\n")

    y = df["isFraud"].values.astype(np.int64)
    dt = df["TransactionDT"].values.astype(np.int64)
    tx_ids = df["TransactionID"].values.astype(np.int64)
    N = len(y)
    print(f"[3/8] Transactions: {N:,}  (fraud: {y.sum():,} = {100*y.mean():.2f}%)")

    # ── Split FIRST, so preprocessing only ever sees training rows ─────────
    print(f"    Creating {args.split} train/val/test masks ...")
    train_mask, val_mask, test_mask = make_masks(
        y, dt, args.split, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    print(f"    Train: {int(train_mask.sum()):,}  "
          f"Val: {int(val_mask.sum()):,}  Test: {int(test_mask.sum()):,}\n")

    # ── Fit the preprocessor on training rows only ─────────────────────────
    print("[4/8] Fitting preprocessor on TRAINING ROWS ONLY ...")
    t0 = time.time()
    pre = FraudPreprocessor().fit(df, fit_mask=train_mask.numpy())
    X = pre.transform(df)
    pre.save(PIPELINE_PATH)
    print(f"    Node feature matrix: {X.shape[0]:,} x {X.shape[1]} "
          f"({time.time()-t0:.1f}s)")
    print(f"    Engineered features added: "
          f"{[c for c in pre.all_cols if c.endswith('_count') or c.startswith('TX_') or c.startswith('TransactionAmt_')][:8]} ...\n")

    # ── Edges from the RAW (pre-encoding) attribute values ─────────────────
    edge_cols = [c for c in EDGE_ATTRIBUTES if c in df.columns]
    missing = [c for c in EDGE_ATTRIBUTES if c not in df.columns]
    if missing:
        print(f"    [!] Edge attributes not in data: {missing}")
    edge_index, edge_weight = build_edges(
        df[edge_cols], dt, EDGE_ATTRIBUTES, args.k, args.max_group)

    # ── Assemble ───────────────────────────────────────────────────────────
    print("[6/8] Assembling PyTorch Geometric Data object ...")
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.edge_weight = data.edge_attr

    # Normalised weights are what the model consumes: raw additive weights run
    # 1..14 and would blow up the pre-activation scale of a mean aggregator.
    data.edge_weight_norm = edge_weight / edge_weight.max() if edge_weight.numel() else edge_weight

    # Edge masks that keep training from aggregating over the future.
    # An undirected edge between a train node and a test node would otherwise
    # let a training example read a transaction that had not happened yet.
    if edge_index.numel():
        src, dst = edge_index[0], edge_index[1]
        data.train_edge_mask = train_mask[src] & train_mask[dst]
        seen = train_mask | val_mask
        data.val_edge_mask = seen[src] & seen[dst]
    else:
        data.train_edge_mask = torch.zeros(0, dtype=torch.bool)
        data.val_edge_mask = torch.zeros(0, dtype=torch.bool)

    data.transaction_ids = torch.tensor(tx_ids, dtype=torch.long)
    data.feature_names = list(pre.all_cols)
    data.split_mode = args.split

    assert data.x.shape[0] == N, "x row count mismatch"
    assert data.y.shape[0] == N, "y length mismatch"
    assert data.edge_attr.shape[0] == data.edge_index.shape[1], \
        "edge_attr / edge_index length mismatch"
    print("    [OK] Data object validated\n")

    print_stats(data)
    visualize_subgraph(data, SUBGRAPH_NODES, OUTPUT_IMG, RANDOM_SEED)

    print(f"Saving graph to {OUTPUT_GRAPH} ...")
    torch.save(data, OUTPUT_GRAPH)
    print(f"    [OK] Saved ({os.path.getsize(OUTPUT_GRAPH)/1e6:.1f} MB)\n")

    print(f"Total pipeline time: {time.time()-t_start:.1f}s")
    print("\n[DONE] fraud_graph.pt + preprocessor.pkl are ready.")
    print("Next: py save_edge_attr_lookup.py && py train.py\n")


if __name__ == "__main__":
    main()
