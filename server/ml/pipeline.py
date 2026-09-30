"""
ml/pipeline.py — End-to-end inference pipeline for uploaded transaction CSVs.

Flow: validate → preprocess → build graph → create PyG Data → run GraphSAGE → return results

**Graph Attachment (Option A)**:
When edge_attr_lookup.pkl is available, uploaded transactions are attached to
the existing 590,540-node training graph via shared edge attributes. This gives
the model neighborhood context from training data, matching its transductive
training setup.

Combined graph layout:
    Nodes [0 .. n_training-1]           = training graph nodes
    Nodes [n_training .. n_training+N-1] = uploaded transaction nodes

Only probabilities for the uploaded N nodes are returned to the frontend.

Reuses the FraudPreprocessor (preprocessor.pkl) fitted during training and
the edge-construction logic from preprocessing_pipeline.py.
"""

import time
import uuid
import tracemalloc
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from itertools import combinations
from typing import Dict, Any, List, Optional, Tuple

from server.ml import loader
from graph_sampler import CSRGraph, NeighborSampleLoader

# Imported rather than re-declared: a third copy of this table was one edit away
# from silently disagreeing with the graph the model was trained on.
from preprocessing_pipeline import EDGE_ATTRIBUTES
from build_graph import K_TEMPORAL, MAX_GROUP_SIZE

MIN_EDGE_WEIGHT = 1.0
MAX_UPLOAD_ROWS = 10000
EVAL_NEIGHBORS = [60, 30]

# Minimum columns needed for meaningful prediction (subset of the full 434)
# The preprocessor handles missing columns by filling NaN, but we need at least
# a few transaction-level columns to produce useful predictions.
MINIMUM_COLUMNS = {"TransactionAmt"}

# Columns the preprocessor was fitted on (excluding TransactionID and isFraud)
# We don't require all 432 — the preprocessor fills missing ones with NaN/median.


def validate_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate an uploaded DataFrame.
    Returns a report dict with status, warnings, and column analysis.
    """
    warnings_list = []
    errors = []

    # Row count
    if len(df) == 0:
        errors.append("CSV is empty — no data rows found.")
    if len(df) > MAX_UPLOAD_ROWS:
        errors.append(f"CSV has {len(df):,} rows — maximum is {MAX_UPLOAD_ROWS:,}.")

    # Check minimum columns
    missing_required = MINIMUM_COLUMNS - set(df.columns)
    if missing_required:
        errors.append(f"Missing required column(s): {', '.join(sorted(missing_required))}")

    # Get preprocessor columns for comparison
    preprocessor = loader.get_preprocessor()
    if preprocessor is None:
        errors.append("Preprocessor not loaded — cannot process transactions.")
        return {"valid": False, "errors": errors, "warnings": [], "column_report": {}}

    # The raw column names an upload can actually contain. Comparing against
    # preprocessor.all_cols (which now includes derived features such as
    # card1_count and TX_hour) would report them all as missing.
    expected_cols = set(getattr(preprocessor, "input_cols",
                                preprocessor.all_cols))
    uploaded_cols = set(df.columns) - {"TransactionID", "isFraud"}
    matched_cols  = uploaded_cols & expected_cols
    missing_cols  = expected_cols - uploaded_cols
    extra_cols    = uploaded_cols - expected_cols

    if len(matched_cols) < 5:
        errors.append(
            f"Only {len(matched_cols)} columns match the training schema. "
            f"Need at least 5. Make sure this is IEEE-CIS format."
        )

    if missing_cols:
        warnings_list.append(
            f"{len(missing_cols)} expected columns missing — will be filled with defaults. "
            f"Examples: {', '.join(sorted(missing_cols)[:5])}"
        )
    if extra_cols:
        warnings_list.append(
            f"{len(extra_cols)} extra columns will be ignored. "
            f"Examples: {', '.join(sorted(extra_cols)[:5])}"
        )

    # Check for duplicates
    if "TransactionID" in df.columns:
        n_dup = df["TransactionID"].duplicated().sum()
        if n_dup > 0:
            warnings_list.append(f"{n_dup} duplicate TransactionIDs found.")

    # Missing value summary
    total_cells = len(df) * len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    if missing_cells > 0:
        pct = missing_cells / total_cells * 100
        warnings_list.append(f"{missing_cells:,} missing values ({pct:.1f}%) — will be imputed automatically.")

    return {
        "valid":          len(errors) == 0,
        "errors":         errors,
        "warnings":       warnings_list,
        "total_rows":     len(df),
        "total_columns":  len(df.columns),
        "matched_columns": len(matched_cols),
        "missing_columns": len(missing_cols),
        "extra_columns":   len(extra_cols),
        "has_labels":      "isFraud" in df.columns,
        "has_ids":         "TransactionID" in df.columns,
    }


def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Raw upload -> the model's node-feature matrix, via the fitted
    FraudPreprocessor.

    The preprocessor handles absent columns itself: numerics take the training
    median, categoricals take the "__missing__" ordinal code. That is exactly
    what the model saw for genuinely-null training rows, so it is the
    consistent choice.

    The previous version then overwrote every "missing" numeric column with 0.0
    ("the scaled mean"). That was redundant, and once the preprocessor started
    deriving features it became harmful: names like `card1_count`, `TX_hour` and
    `TransactionAmt_log` are not present in any upload, so they all looked
    missing and would have been zeroed out despite being computed correctly.
    Missing columns are reported, not rewritten.
    """
    preprocessor = loader.get_preprocessor()
    if preprocessor is None:
        raise RuntimeError("Preprocessor not loaded")

    feature_df = df.drop(columns=["TransactionID", "isFraud"], errors="ignore")

    # Compare against the RAW input schema, not the engineered feature names.
    expected = set(getattr(preprocessor, "input_cols", preprocessor.all_cols))
    missing = expected - set(feature_df.columns)
    if missing:
        print(f"[Preprocessing] {len(missing)} of {len(expected)} input columns "
              f"absent - imputed from training statistics.")
        print(f"[Preprocessing] Examples: {sorted(missing)[:10]}")
        if len(missing) > 0.5 * len(expected):
            print(f"[Preprocessing] WARNING: over half the schema is missing. "
                  f"Predictions will be driven by the few columns supplied.")

    X = preprocessor.transform(feature_df)
    return X.astype(np.float32)


def build_transaction_graph(
    df: pd.DataFrame,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Edges among the uploaded transactions only (mini-graph fallback, used when
    edge_attr_lookup.pkl is missing so there is no training graph to attach to).

    Delegates to build_intra_batch_edges with offset 0 so the fallback and the
    attached path build edges by the same rule. The old implementation here was
    all-pairs-within-a-group with a hard skip above 30 members, which produced
    no edges at all on realistic batches.
    """
    return build_intra_batch_edges(df, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH ATTACHMENT — Option A
#  Attach uploaded transactions to the training graph via shared attributes.
# ═══════════════════════════════════════════════════════════════════════════════

def build_cross_edges(
    df: pd.DataFrame,
    inverted_index: Dict[str, Dict],
    n_existing: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    Build cross-edges between uploaded transactions (indexed n_existing..n_existing+N-1)
    and training graph nodes (indexed 0..n_existing-1) using the inverted index.

    Parameters
    ----------
    df              : DataFrame of uploaded transactions (raw, with edge attrs)
    inverted_index  : {attr → {value → [training_node_indices]}}
    n_existing      : number of training graph nodes (590540)

    Returns
    -------
    edge_index      : LongTensor [2, 2E]  (bidirectional)
    edge_weight     : FloatTensor [2E]
    neighbor_counts : dict mapping uploaded row index → number of training neighbors
    """
    edge_weights: Dict[Tuple[int, int], float] = defaultdict(float)
    neighbor_sets: Dict[int, set] = defaultdict(set)  # track unique training neighbors

    n_new = len(df)
    df_reset = df.reset_index(drop=True)

    for attr, weight in EDGE_ATTRIBUTES.items():
        if attr not in df_reset.columns or attr not in inverted_index:
            continue

        attr_lookup = inverted_index[attr]

        for new_i in range(n_new):
            new_val = df_reset.at[new_i, attr]
            if pd.isna(new_val):
                continue

            # Must match save_edge_attr_lookup._normalise() exactly, or every
            # numeric lookup misses and the upload attaches to nothing.
            lookup_key = new_val
            if isinstance(new_val, (np.floating, float)):
                f = float(new_val)
                lookup_key = int(f) if f.is_integer() else f
            elif isinstance(new_val, np.integer):
                lookup_key = int(new_val)

            matches = attr_lookup.get(lookup_key, [])
            if not matches:
                continue
            # No length filter here any more. The index is built by
            # save_edge_attr_lookup.py to hold only the K most recent training
            # transactions per value, matching build_graph.py's edge rule.
            # The old `len(matches) > GROUP_CAP` guard discarded every busy
            # value, which is why uploads attached to almost nothing.

            new_node_idx = n_existing + new_i
            for train_idx in matches:
                src = min(new_node_idx, train_idx)
                dst = max(new_node_idx, train_idx)
                edge_weights[(src, dst)] += weight
                neighbor_sets[new_i].add(train_idx)

    # Filter by minimum weight
    filtered = [(e, w) for e, w in edge_weights.items() if w >= MIN_EDGE_WEIGHT]

    # Build neighbor count dict (per uploaded row)
    neighbor_counts = {i: len(neighbors) for i, neighbors in neighbor_sets.items()}

    if not filtered:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float),
            neighbor_counts,
        )

    src_list = [e[0] for e, _ in filtered]
    dst_list = [e[1] for e, _ in filtered]
    wts = [w for _, w in filtered]

    # Make undirected
    src_bi = np.array(src_list + dst_list, dtype=np.int64)
    dst_bi = np.array(dst_list + src_list, dtype=np.int64)
    wts_bi = np.array(wts + wts, dtype=np.float32)

    edge_index  = torch.tensor(np.stack([src_bi, dst_bi], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(wts_bi, dtype=torch.float)
    return edge_index, edge_weight, neighbor_counts


def build_intra_batch_edges(
    df: pd.DataFrame,
    n_existing: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Edges among the uploaded transactions, shifted to occupy nodes
    [n_existing .. n_existing + N - 1] in the combined graph.

    Uses the same rule as build_graph.py: order each attribute group by time and
    link every transaction to its K most recent peers. The previous version
    generated all pairs within a group and skipped any group larger than 30 —
    on a real 2,500-row batch every group exceeded 30, so it produced exactly
    zero edges (see attachment_info.intra_edges == 0 in the recorded run).
    """
    edge_weights: Dict[Tuple[int, int], float] = defaultdict(float)
    df_reset = df.reset_index(drop=True)
    n_new = len(df_reset)

    if "TransactionDT" in df_reset.columns:
        dt = pd.to_numeric(df_reset["TransactionDT"], errors="coerce")                .fillna(0).to_numpy(dtype=np.int64)
    else:
        dt = np.arange(n_new, dtype=np.int64)   # fall back to file order

    for attr, weight in EDGE_ATTRIBUTES.items():
        if attr not in df_reset.columns:
            continue
        col = df_reset[attr]
        pos = np.flatnonzero(col.notna().to_numpy())
        if pos.size < 2:
            continue

        codes, _ = pd.factorize(col.iloc[pos], sort=False)
        counts = np.bincount(codes)
        keep = counts[codes] <= MAX_GROUP_SIZE
        codes, pos = codes[keep], pos[keep]
        if codes.size < 2:
            continue

        order = np.lexsort((dt[pos], codes))
        s_codes, s_pos = codes[order], pos[order]

        for lag in range(1, K_TEMPORAL + 1):
            if s_codes.size <= lag:
                break
            same = s_codes[lag:] == s_codes[:-lag]
            if not same.any():
                continue
            for a, b in zip(s_pos[lag:][same], s_pos[:-lag][same]):
                src = n_existing + int(min(a, b))
                dst = n_existing + int(max(a, b))
                edge_weights[(src, dst)] += weight

    filtered = [(e, w) for e, w in edge_weights.items() if w >= MIN_EDGE_WEIGHT]
    if not filtered:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)

    src_list = [e[0] for e, _ in filtered]
    dst_list = [e[1] for e, _ in filtered]
    wts = [w for _, w in filtered]

    src_bi = np.array(src_list + dst_list, dtype=np.int64)
    dst_bi = np.array(dst_list + src_list, dtype=np.int64)
    wts_bi = np.array(wts + wts, dtype=np.float32)

    edge_index = torch.tensor(np.stack([src_bi, dst_bi], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(wts_bi, dtype=torch.float)
    return edge_index, edge_weight


def _normalised(w: torch.Tensor) -> torch.Tensor:
    """
    Scale raw additive edge weights into the same range the model was trained
    on. build_graph.py divides by the graph-wide maximum; reproducing that here
    keeps uploaded edges comparable to training edges.
    """
    if w.numel() == 0:
        return w
    data = loader.get_data()
    ref = None
    if data is not None and getattr(data, "edge_attr", None) is not None             and data.edge_attr.numel():
        ref = float(data.edge_attr.max())
    if not ref:
        ref = float(w.max()) or 1.0
    return w / ref


def _risk_level(prob: float, threshold: Optional[float] = None) -> str:
    if prob >= 0.80:
        return "CRITICAL"
    elif prob >= 0.60:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def run_inference(
    features: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Run the trained GraphSAGE model on the new graph.
    Returns fraud probability array of shape (N,).
    """
    model = loader.get_model()
    if model is None:
        raise RuntimeError("Model not loaded")

    device = loader.get_device()
    x = torch.tensor(features, dtype=torch.float32).to(device)
    ei = edge_index.to(device)
    ew = edge_weight.to(device) if edge_weight is not None else None

    model.eval()
    with torch.no_grad():
        logits = model(x, ei, ew)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    return probs


def run_attached_inference(
    features: np.ndarray,
    df: pd.DataFrame,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run inference with uploaded transactions attached to the training graph.

    Combined graph layout:
        Nodes [0 .. n_training-1]            = training graph (from fraud_graph.pt)
        Nodes [n_training .. n_training+N-1]  = uploaded transactions

    Returns
    -------
    probs          : np.ndarray of shape (N,) — probabilities for uploaded transactions ONLY
    attachment_info : dict with edge counts, timing, neighbor info
    """
    model = loader.get_model()
    data = loader.get_data()
    inverted_index = loader.get_edge_attr_lookup()

    if model is None or data is None:
        raise RuntimeError("Model or graph not loaded")
    if inverted_index is None:
        raise RuntimeError("Edge attribute lookup not loaded — cannot attach to training graph")

    device = loader.get_device()
    n_training = data.num_nodes  # 590,540
    n_uploaded = len(features)

    attachment_info = {
        "n_training_nodes": n_training,
        "n_uploaded_nodes": n_uploaded,
        "n_combined_nodes": n_training + n_uploaded,
    }

    # ── Step 1: Build cross-edges (uploaded ↔ training) ───────────────────
    print("[SCAN API] Graph construction started")
    t_cross = time.time()
    cross_ei, cross_ew, neighbor_counts = build_cross_edges(
        df, inverted_index, n_training
    )
    cross_time_ms = int((time.time() - t_cross) * 1000)

    n_cross_edges = cross_ei.shape[1] // 2 if cross_ei.shape[1] > 0 else 0
    attachment_info["cross_edges"] = n_cross_edges
    attachment_info["cross_edge_time_ms"] = cross_time_ms
    attachment_info["neighbor_counts"] = neighbor_counts

    nodes_with_neighbors = sum(1 for c in neighbor_counts.values() if c > 0)
    attachment_info["nodes_with_training_neighbors"] = nodes_with_neighbors
    attachment_info["nodes_without_training_neighbors"] = n_uploaded - nodes_with_neighbors

    print(f"[Graph Attachment] Cross-edges: {n_cross_edges:,} undirected "
          f"({cross_time_ms}ms)")
    print(f"[Graph Attachment] Uploaded nodes with training neighbors: "
          f"{nodes_with_neighbors}/{n_uploaded}")

    # ── Step 2: Build intra-batch edges (among uploaded transactions) ─────
    t_intra = time.time()
    intra_ei, intra_ew = build_intra_batch_edges(df, n_training)
    intra_time_ms = int((time.time() - t_intra) * 1000)

    n_intra_edges = intra_ei.shape[1] // 2 if intra_ei.shape[1] > 0 else 0
    attachment_info["intra_edges"] = n_intra_edges
    attachment_info["intra_edge_time_ms"] = intra_time_ms

    print(f"[Graph Attachment] Intra-batch edges: {n_intra_edges:,} undirected "
          f"({intra_time_ms}ms)")

    # ── Step 3: Combine features and edges ────────────────────────────────
    # The uploaded rows become nodes [n_training .. n_training + N - 1].
    t_combine = time.time()
    uploaded_x = torch.tensor(features, dtype=torch.float32)
    combined_x = torch.cat([data.x, uploaded_x], dim=0)

    train_ew = getattr(data, "edge_weight_norm", data.edge_attr)
    edge_parts = [data.edge_index]
    weight_parts = [train_ew]
    if cross_ei.shape[1] > 0:
        edge_parts.append(cross_ei)
        weight_parts.append(_normalised(cross_ew))
    if intra_ei.shape[1] > 0:
        edge_parts.append(intra_ei)
        weight_parts.append(_normalised(intra_ew))

    combined_ei = torch.cat(edge_parts, dim=1)
    combined_ew = torch.cat(weight_parts)
    combine_time_ms = int((time.time() - t_combine) * 1000)

    n_training_edges = data.edge_index.shape[1] // 2
    attachment_info["training_edges"] = n_training_edges
    attachment_info["total_edges"] = combined_ei.shape[1] // 2
    attachment_info["combine_time_ms"] = combine_time_ms

    print(f"[Graph Attachment] Combined edges: {combined_ei.shape[1]:,} directed "
          f"({n_training_edges:,} training + {n_cross_edges:,} cross + "
          f"{n_intra_edges:,} intra)")
    print("[SCAN API] Graph construction complete")

    # ── Step 4: Sampled inference over the uploaded nodes only ────────────
    # This used to run one forward pass over the entire combined graph and then
    # throw away 590,540 of the 593,040 results. Message passing materialises a
    # [num_edges, num_features] tensor, so on the rebuilt graph that is a
    # 4.5M x 451 float allocation — it no longer fits, and it never needed to
    # exist: only the uploaded rows' 2-hop neighbourhoods matter.
    print("[SCAN API] Inference started")
    t_infer = time.time()

    n_combined = n_training + n_uploaded
    csr = CSRGraph(combined_ei, combined_ew, n_combined)
    seeds = torch.arange(n_training, n_combined)
    dummy_y = torch.zeros(n_combined, dtype=torch.long)

    loader_ = NeighborSampleLoader(combined_x, dummy_y, csr, seeds,
                                   EVAL_NEIGHBORS, batch_size=2048,
                                   shuffle=False)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader_:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            chunks.append(
                torch.sigmoid(logits[:batch.batch_size]).squeeze(1).cpu())
    probs = torch.cat(chunks).numpy()

    infer_time_ms = int((time.time() - t_infer) * 1000)
    attachment_info["inference_time_ms"] = infer_time_ms
    print(f"[Graph Attachment] Inference complete: {infer_time_ms}ms "
          f"({n_uploaded:,} uploaded nodes scored)")
    print("[SCAN API] Inference complete")

    assert len(probs) == n_uploaded,         f"Expected {n_uploaded} probabilities, got {len(probs)}"

    del combined_x, combined_ei, combined_ew, csr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return probs, attachment_info


def generate_feature_importance(
    features: np.ndarray,
    probs: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 5,
) -> List[List[str]]:
    """
    Per-row "most unusual features", ranked by |z-score| against the training
    distribution. Cheap enough to run on a whole batch, unlike GNNExplainer.

    Only standardised numeric columns are eligible. Categorical columns hold
    raw OrdinalEncoder codes — integers running 0..cardinality — which are not
    z-scores at all. Ranking by raw magnitude let a category that happened to
    encode as 300 outrank every genuine outlier, so the "top features" shown
    next to a prediction were essentially a list of the highest-cardinality
    categorical columns, the same ones for every row.
    """
    preprocessor = loader.get_preprocessor()
    if preprocessor is None:
        return [[] for _ in range(len(features))]

    col_names = list(preprocessor.all_cols)
    num_cols = set(getattr(preprocessor, "num_cols", col_names))
    eligible = np.array([i for i, c in enumerate(col_names) if c in num_cols],
                        dtype=np.int64)
    if eligible.size == 0:
        eligible = np.arange(min(features.shape[1], len(col_names)))

    scores = np.abs(features[:, eligible])
    k = int(min(top_k, eligible.size))

    # argpartition is O(n) per row; a full argsort over 451 columns per row is
    # wasted work when only the top 5 are shown.
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    top = eligible[part[rows, order]]

    return [[col_names[idx] for idx in row] for row in top]


def run_full_pipeline(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: validate → preprocess → graph → infer → results.

    If edge_attr_lookup is available, uses graph attachment (combined-graph
    inference with the training graph). Otherwise falls back to mini-graph
    inference with a warning.
    """
    t0 = time.time()
    job_id = str(uuid.uuid4())[:8]

    # None means "use the threshold train.py tuned on the validation split",
    # which is what the model was actually evaluated at.
    if threshold is None:
        threshold = loader.get_threshold()

    # Start memory tracking
    tracemalloc.start()

    # ── Step 1: Validate ──────────────────────────────────────────────────
    validation = validate_csv(df)
    if not validation["valid"]:
        tracemalloc.stop()
        return {
            "job_id": job_id,
            "success": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "processing_time_ms": int((time.time() - t0) * 1000),
        }

    # Extract labels if present (for accuracy evaluation)
    true_labels = None
    if "isFraud" in df.columns:
        true_labels = df["isFraud"].values.astype(int)

    # Extract transaction IDs if present
    if "TransactionID" in df.columns:
        tx_ids = df["TransactionID"].values.tolist()
    else:
        tx_ids = [f"ROW_{i}" for i in range(len(df))]

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    df_reset = df.reset_index(drop=True)
    t_preprocess = time.time()
    print("[SCAN API] Preprocessing started")
    features = preprocess_features(df_reset)
    preprocess_time_ms = int((time.time() - t_preprocess) * 1000)
    print("[SCAN API] Preprocessing complete")

    # ── Step 3 & 4: Graph construction + inference ────────────────────────
    inverted_index = loader.get_edge_attr_lookup()
    use_attachment = inverted_index is not None and loader.get_data() is not None

    attachment_info = None
    neighbor_counts = {}

    if use_attachment:
        # ── Graph Attachment Mode ─────────────────────────────────────────
        print(f"[Pipeline] Using GRAPH ATTACHMENT mode (training graph + uploaded)")
        try:
            probs, attachment_info = run_attached_inference(
                features, df_reset, threshold
            )
            neighbor_counts = attachment_info.get("neighbor_counts", {})

            n_edges = attachment_info.get("total_edges", 0)
            n_cross = attachment_info.get("cross_edges", 0)
            n_intra = attachment_info.get("intra_edges", 0)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"[Pipeline] OOM during attached inference: {e}")
                print(f"[Pipeline] STOPPING — combined graph too large for available memory")
                tracemalloc.stop()
                return {
                    "job_id": job_id,
                    "success": False,
                    "errors": [
                        f"Out of memory during combined-graph inference. "
                        f"The training graph ({loader.get_data().num_nodes:,} nodes) + "
                        f"uploaded ({len(df):,} nodes) is too large. "
                        f"Error: {str(e)}"
                    ],
                    "warnings": [],
                    "processing_time_ms": int((time.time() - t0) * 1000),
                }
            raise  # Re-raise non-OOM errors

    else:
        # ── Fallback: Mini-graph mode (no graph attachment) ───────────────
        print(f"[Pipeline] Using MINI-GRAPH mode (no graph attachment — "
              f"edge_attr_lookup not available)")
        print("[SCAN API] Graph construction started")
        edge_index, edge_weight = build_transaction_graph(df_reset, len(df_reset))
        print("[SCAN API] Graph construction complete")
        n_edges = edge_index.shape[1] // 2 if edge_index.shape[1] > 0 else 0
        n_cross = 0
        n_intra = n_edges
        print("[SCAN API] Inference started")
        ew_norm = _normalised(edge_weight)
        probs = run_inference(features, edge_index, ew_norm, threshold)
        print("[SCAN API] Inference complete")

    # ── Step 4b: apply the ACTIVE model, not always the GNN ──────────────
    # Uploads used to be scored by the GNN alone and then cut at
    # loader.get_threshold() — which is the *active* model's threshold. With
    # the ensemble active that meant GNN scores judged against the ensemble's
    # 0.61, and uploaded rows disagreeing with the same rows scored in-graph.
    gnn_probs = probs
    active = loader.get_active_model()
    lgbm = loader.get_lgbm()
    if active in ("lightgbm", "ensemble") and lgbm is not None:
        lgbm_probs = lgbm.predict_proba(features)[:, 1].astype(np.float32)
        if active == "lightgbm":
            probs = lgbm_probs
        else:
            w = float(loader.get_ensemble_config().get("weight_gnn", 0.5))
            probs = w * gnn_probs + (1.0 - w) * lgbm_probs
    print(f"[Pipeline] Scored with model: {active}")

    # ── Step 5: Feature importance ────────────────────────────────────────
    print("[SCAN API] Explanation started")
    top_features = generate_feature_importance(features, probs, df_reset)
    print("[SCAN API] Explanation complete")

    # ── Step 6: Build results ─────────────────────────────────────────────
    results = []
    for i in range(len(probs)):
        prob = float(probs[i])
        is_fraud = prob >= threshold
        confidence = prob if is_fraud else (1.0 - prob)
        row = {
            "row_index":         i,
            "transaction_id":    str(tx_ids[i]),
            "fraud_probability": round(prob, 6),
            "is_fraud":          is_fraud,
            "risk_level":        _risk_level(prob, threshold),
            "confidence":        round(confidence, 6),
            "top_features":      top_features[i],
            "training_neighbors": neighbor_counts.get(i, 0),
        }
        if true_labels is not None:
            row["true_label"] = int(true_labels[i])
        results.append(row)

    fraud_count = sum(1 for r in results if r["is_fraud"])
    processing_time = int((time.time() - t0) * 1000)

    # Memory usage
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    response = {
        "job_id":       job_id,
        "success":      True,
        "total_rows":   len(results),
        "fraud_count":  fraud_count,
        "legit_count":  len(results) - fraud_count,
        "fraud_rate":   round(fraud_count / max(len(results), 1) * 100, 2),
        "graph_stats": {
            "nodes": len(results),
            "edges": n_edges,
            "cross_edges": n_cross,
            "intra_edges": n_intra,
            "mode": "attached" if use_attachment else "mini-graph",
        },
        "threshold":         round(float(threshold), 4),
        "model":             loader.get_active_model(),
        "processing_time_ms": processing_time,
        "preprocess_time_ms": preprocess_time_ms,
        "memory_mb": {
            "current": round(current_mem / 1e6, 1),
            "peak":    round(peak_mem / 1e6, 1),
        },
        "validation":        validation,
        "results":           results,
    }

    # Add attachment info if available
    if attachment_info is not None:
        response["graph_attachment"] = {
            "n_training_nodes":  attachment_info["n_training_nodes"],
            "n_uploaded_nodes":  attachment_info["n_uploaded_nodes"],
            "cross_edges":       attachment_info["cross_edges"],
            "intra_edges":       attachment_info["intra_edges"],
            "training_edges":    attachment_info["training_edges"],
            "total_edges":       attachment_info["total_edges"],
            "nodes_with_training_neighbors": attachment_info["nodes_with_training_neighbors"],
            "nodes_without_training_neighbors": attachment_info["nodes_without_training_neighbors"],
            "cross_edge_time_ms": attachment_info["cross_edge_time_ms"],
            "intra_edge_time_ms": attachment_info["intra_edge_time_ms"],
            "inference_time_ms":  attachment_info["inference_time_ms"],
        }

    # If true labels provided, add accuracy metrics
    if true_labels is not None:
        preds = np.array([1 if r["is_fraud"] else 0 for r in results])
        correct = int((preds == true_labels).sum())
        response["accuracy"] = round(correct / len(true_labels), 4)

    print("[SCAN API] Response created")
    return response
