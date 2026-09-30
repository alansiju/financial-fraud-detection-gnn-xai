"""
ml/loader.py - Singleton loader for the graph and model.
Loads once at startup; all API handlers share the same objects.
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.graphsage import GraphSAGE
from server.config import (GRAPH_PATH, MODEL_PATH, PREPROCESSOR_PATH,
                           EDGE_ATTR_LOOKUP_PATH, LOG_PATH, BASELINE_PATH,
                           MODEL_DEFAULTS, DEFAULT_FRAUD_THRESHOLD,
                           NODE_META_PATH, LGBM_PATH, ENSEMBLE_PATH)

# ── Singleton state ────────────────────────────────────────────────────────
_device: Optional[torch.device] = None
_data   = None          # torch_geometric Data object
_model: Optional[GraphSAGE] = None
_model_config: dict = {}
_test_indices: list = []
_training_log: list = []
_baseline_metrics: list = []
_feature_names: list = []
_preprocessor = None
_edge_attr_lookup = None   # DataFrame of edge attributes for training nodes
_tx_ids: Optional[np.ndarray] = None  # TransactionID array matching node indices
_tx_to_node: Optional[dict] = None   # Map of TransactionID -> node index
_threshold: float = 0.5              # tuned on validation, read from the checkpoint
_threshold_override: Optional[float] = None   # operator setting, if any
_edge_weights: dict = {}             # attr -> edge weight, from the lookup pickle
_node_probs: dict = {}               # model name -> probability per graph node
_node_meta = None                    # DataFrame of raw columns, row i == node i
_lgbm = None                         # LightGBM model, same features as the GNN
_ensemble: dict = {}                 # blend weight + per-model thresholds
_active_model: str = 'gnn'           # gnn | lightgbm | ensemble
_full_csr = None                     # CSR adjacency over the whole graph
_is_loaded: bool = False


def get_device() -> torch.device:
    return _device or torch.device("cpu")

def get_data():
    return _data

def get_model() -> Optional[GraphSAGE]:
    return _model

def get_model_config() -> dict:
    return _model_config

def get_test_indices() -> list:
    return _test_indices

def get_training_log() -> list:
    return _training_log

def get_baseline_metrics() -> list:
    return _baseline_metrics

def get_feature_names() -> list:
    return _feature_names

def get_preprocessor():
    return _preprocessor

def get_edge_attr_lookup():
    """Returns the edge-attribute lookup DataFrame (or None if not loaded)."""
    return _edge_attr_lookup

def get_tx_ids() -> Optional[np.ndarray]:
    return _tx_ids

def get_node_by_tx_id(tx_id: int) -> Optional[int]:
    if _tx_to_node is not None:
        return _tx_to_node.get(int(tx_id))
    return None

def get_tx_id_by_node(node_id: int) -> Optional[int]:
    if _tx_ids is not None and 0 <= int(node_id) < len(_tx_ids):
        return int(_tx_ids[int(node_id)])
    return None

def resolve_node_id(identifier: int) -> int:
    """
    Resolve identifier to a valid graph node index [0 .. num_nodes-1].
    - If identifier is a TransactionID in the dataset (e.g. 2991340), returns its node index (e.g. 4340).
    - If identifier is already a valid graph node index (< num_nodes), returns it directly.
    - Returns -1 if not found.
    """
    try:
        ident_int = int(identifier)
    except (ValueError, TypeError):
        return -1

    if _tx_to_node is not None and ident_int in _tx_to_node:
        return _tx_to_node[ident_int]
    if _data is not None and 0 <= ident_int < _data.num_nodes:
        return ident_int
    return -1

def get_available_models() -> list:
    """Which model variants can actually be served right now."""
    out = [{
        "id": "gnn",
        "name": _model_label(),
        "available": _model is not None,
        "threshold": round(_threshold_for("gnn"), 4),
        "description": "GraphSAGE over the transaction graph",
    }]
    if _lgbm is not None:
        out.append({
            "id": "lightgbm",
            "name": "LightGBM",
            "available": True,
            "threshold": round(_threshold_for("lightgbm"), 4),
            "description": "Gradient boosting on the same 469 features",
        })
    if _lgbm is not None and _model is not None and _ensemble:
        w = _ensemble.get("weight_gnn", 0.0)
        out.append({
            "id": "ensemble",
            "name": "Ensemble",
            "available": True,
            "threshold": round(_threshold_for("ensemble"), 4),
            "description": f"{w:.0%} GNN + {1-w:.0%} LightGBM, blend tuned on validation",
        })
    return out


def _model_label() -> str:
    cfg = _model_config or {}
    layers = cfg.get("num_layers", 2)
    skip = " +skip" if cfg.get("use_skip", True) else ""
    return f"GraphSAGE-{layers}L{skip}"


def get_active_model() -> str:
    return _active_model


def get_active_model_name() -> str:
    """Human-readable name of the model currently serving predictions."""
    for m in get_available_models():
        if m["id"] == _active_model:
            return m["name"]
    return _active_model


def set_active_model(name: str) -> str:
    """Switch which model serves predictions. Unknown or unavailable names raise."""
    global _active_model
    ids = {m["id"] for m in get_available_models() if m["available"]}
    if name not in ids:
        raise ValueError(f"model '{name}' is not available (have: {sorted(ids)})")
    _active_model = name
    return _active_model


def get_lgbm():
    return _lgbm


def get_ensemble_config() -> dict:
    return _ensemble


def _threshold_for(name: str) -> float:
    """Each variant's own validation-tuned threshold."""
    thr = (_ensemble.get("thresholds") or {}).get(name)
    if thr is not None:
        return float(thr)
    return _threshold


def get_node_meta():
    """
    Raw transaction columns aligned with graph node indices, or None if
    node_meta.pkl has not been generated. Used by the analytics endpoints so
    they can report real distributions instead of synthetic ones.
    """
    return _node_meta


def get_threshold() -> float:
    """
    The decision threshold in force: the operator's override if one is set,
    otherwise the value tuned on the validation split and stored in
    best_model.pt.

    Previously every caller assumed 0.5, which on this model's probability
    distribution flagged almost nothing.
    """
    if _threshold_override is not None:
        return _threshold_override
    return _threshold_for(_active_model)


def get_tuned_threshold(model: Optional[str] = None) -> float:
    """The validation-tuned threshold for a model, ignoring any override."""
    return _threshold_for(model or _active_model)


def set_threshold_override(value: Optional[float]) -> float:
    """
    Apply (or clear, with None) the operator threshold from the Settings page.

    The Settings page has always had a "Fraud Detection Threshold" slider
    captioned "transactions above this are flagged as fraud". Its value was
    written to the database and never read by any prediction code, so moving
    it changed nothing. This is what makes it take effect.
    """
    global _threshold_override
    if value is None:
        _threshold_override = None
    else:
        _threshold_override = float(max(0.01, min(0.99, value)))
    return get_threshold()


def get_edge_weights() -> dict:
    return _edge_weights


def get_full_csr():
    """CSR adjacency over the whole training graph (built once, on demand)."""
    global _full_csr
    if _full_csr is None and _data is not None:
        from graph_sampler import CSRGraph
        ew = getattr(_data, "edge_weight_norm", _data.edge_attr)
        _full_csr = CSRGraph(_data.edge_index, ew, _data.num_nodes)
    return _full_csr


def get_all_node_probs(model: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Fraud probability for every node under `model` (default: the active one),
    computed once per variant and cached.

    The graph and the weights are static, so there is no reason to redo this
    per request — and the old code ran a full-graph forward pass on every
    single-node prediction.
    """
    name = model or _active_model
    if name in _node_probs:
        return _node_probs[name]
    if _data is None:
        return None

    if name == "gnn":
        probs = _score_gnn()
    elif name == "lightgbm":
        probs = _score_lgbm()
    elif name == "ensemble":
        g, l = get_all_node_probs("gnn"), get_all_node_probs("lightgbm")
        if g is None or l is None:
            return None
        w = float(_ensemble.get("weight_gnn", 0.5))
        probs = w * g + (1.0 - w) * l
        print(f"[OK] Ensemble blended: {w:.2f} GNN + {1-w:.2f} LightGBM")
    else:
        return None

    if probs is not None:
        _node_probs[name] = probs
    return probs


def _score_gnn() -> Optional[np.ndarray]:
    if _model is None or _data is None:
        return None
    import torch as _torch
    from graph_sampler import NeighborSampleLoader

    csr = get_full_csr()
    idx = _torch.arange(_data.num_nodes)
    loader_ = NeighborSampleLoader(_data.x, _data.y, csr, idx, [60, 30],
                                   batch_size=8192, shuffle=False)
    print(f"[*] Scoring all {_data.num_nodes:,} graph nodes with the GNN ...")
    out = []
    _model.eval()
    with _torch.no_grad():
        for batch in loader_:
            batch = batch.to(_device)
            logits = _model(batch.x, batch.edge_index, batch.edge_attr)
            out.append(_torch.sigmoid(logits[:batch.batch_size]).squeeze(1).cpu())
    probs = _torch.cat(out).numpy()
    print(f"[OK] GNN scores cached (min {probs.min():.4f}, max {probs.max():.4f}, "
          f"median {float(np.median(probs)):.4f})")
    return probs


def _score_lgbm() -> Optional[np.ndarray]:
    if _lgbm is None or _data is None:
        return None
    print(f"[*] Scoring all {_data.num_nodes:,} graph nodes with LightGBM ...")
    probs = _lgbm.predict_proba(_data.x.numpy())[:, 1].astype(np.float32)
    print(f"[OK] LightGBM scores cached (min {probs.min():.4f}, "
          f"max {probs.max():.4f})")
    return probs


def is_loaded() -> bool:
    return _is_loaded


def load_all():
    """Called once during FastAPI startup."""
    global _device, _data, _model, _model_config, _test_indices
    global _training_log, _baseline_metrics, _feature_names, _preprocessor
    global _edge_attr_lookup, _tx_ids, _tx_to_node, _is_loaded
    global _threshold, _edge_weights, _node_probs, _full_csr, _node_meta
    global _lgbm, _ensemble, _active_model

    _device = torch.device("cpu")

    # ── Load graph ─────────────────────────────────────────────────────────
    print("[*] Loading fraud graph...")
    if not GRAPH_PATH.exists():
        print(f"[!] Graph not found at {GRAPH_PATH}")
        return
    _data = torch.load(GRAPH_PATH, map_location=_device, weights_only=False)
    print(f"[OK] Graph loaded: {_data.num_nodes:,} nodes, {_data.num_edges:,} edges")

    # ── Load model ─────────────────────────────────────────────────────────
    print("[*] Loading GraphSAGE model...")
    if not MODEL_PATH.exists():
        print(f"[!] Model not found at {MODEL_PATH}")
        return
    ckpt = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    cfg  = ckpt.get("config", {})
    _model_config = cfg

    in_channels = ckpt.get("in_channels", _data.x.shape[1])
    _model = GraphSAGE(
        in_channels = in_channels,
        hidden1     = cfg.get("hidden1",    MODEL_DEFAULTS["hidden1"]),
        hidden2     = cfg.get("hidden2",    MODEL_DEFAULTS["hidden2"]),
        hidden3     = cfg.get("hidden3",    MODEL_DEFAULTS["hidden3"]),
        mlp_hidden  = cfg.get("mlp_hidden", MODEL_DEFAULTS["mlp_hidden"]),
        dropout1    = cfg.get("dropout1",   MODEL_DEFAULTS["dropout1"]),
        dropout2    = cfg.get("dropout2",   MODEL_DEFAULTS["dropout2"]),
        dropout3    = cfg.get("dropout3",   MODEL_DEFAULTS["dropout3"]),
        num_layers  = cfg.get("num_layers", MODEL_DEFAULTS["num_layers"]),
        use_skip    = cfg.get("use_skip",   MODEL_DEFAULTS["use_skip"]),
    ).to(_device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    print(f"[OK] Model loaded: {_model.count_parameters():,} parameters")

    _threshold = float(ckpt.get("threshold", DEFAULT_FRAUD_THRESHOLD))
    if "threshold" in ckpt:
        print(f"[OK] Decision threshold from checkpoint: {_threshold:.2f} "
              f"(validation-tuned)")
    else:
        print(f"[!] Checkpoint has no tuned threshold - falling back to "
              f"{_threshold:.2f}. Re-run train.py to store one.")

    # ── Test indices ───────────────────────────────────────────────────────
    if hasattr(_data, "test_mask"):
        _test_indices = torch.where(_data.test_mask)[0].cpu().numpy().tolist()

    # ── Training log ───────────────────────────────────────────────────────
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            _training_log = json.load(f)

    # ── Baseline metrics ───────────────────────────────────────────────────
    if BASELINE_PATH.exists():
        import csv
        with open(BASELINE_PATH) as f:
            reader = csv.DictReader(f)
            _baseline_metrics = list(reader)

    # ── Feature names ──────────────────────────────────────────────────────
    n_feat = _data.x.shape[1]
    # Try to get from graph metadata, else generate generic names
    if hasattr(_data, "feature_names") and _data.feature_names:
        _feature_names = list(_data.feature_names)
    elif ckpt.get("feature_names"):
        _feature_names = list(ckpt["feature_names"])
    elif _preprocessor is not None and getattr(_preprocessor, "all_cols", None):
        _feature_names = list(_preprocessor.all_cols)
    else:
        _feature_names = [f"Feature_{i}" for i in range(n_feat)]

    # ── Load preprocessor ──────────────────────────────────────────────
    if PREPROCESSOR_PATH.exists():
        import pickle
        import preprocessing_pipeline
        
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'FraudPreprocessor':
                    return preprocessing_pipeline.FraudPreprocessor
                return super().find_class(module, name)
                
        with open(PREPROCESSOR_PATH, "rb") as f:
            _preprocessor = CustomUnpickler(f).load()
        for imp in (getattr(_preprocessor, "num_imp", None), getattr(_preprocessor, "cat_imp", None)):
            if imp is not None and hasattr(imp, "_fit_dtype") and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = imp._fit_dtype
        print(f"[OK] Preprocessor loaded: {_preprocessor.n_features} features")
    else:
        print(f"[!] Preprocessor not found at {PREPROCESSOR_PATH}")

    # ── Load edge attribute lookup (for graph attachment at inference) ─────
    if EDGE_ATTR_LOOKUP_PATH.exists():
        import pickle
        with open(EDGE_ATTR_LOOKUP_PATH, "rb") as f:
            lookup = pickle.load(f)
        # Support both old (DataFrame) and new (inverted_index) formats
        if "transaction_ids" in lookup:
            _tx_ids = lookup["transaction_ids"]
            _tx_to_node = {int(tid): idx for idx, tid in enumerate(_tx_ids)}
            print(f"[OK] TransactionID mapping loaded: {len(_tx_to_node):,} transactions mapped to node indices")

        _edge_weights = lookup.get("weights", {})
        if "inverted_index" in lookup:
            _edge_attr_lookup = lookup["inverted_index"]
            n_lookup_nodes = lookup.get("n_nodes", "?")
            print(f"[OK] Edge attr lookup loaded (inverted index): "
                  f"{n_lookup_nodes:,} training nodes, "
                  f"attrs={lookup['cols']}")
            # Verify node count matches loaded graph
            if _data is not None and n_lookup_nodes != _data.num_nodes:
                print(f"[WARNING] Lookup n_nodes ({n_lookup_nodes}) != "
                      f"graph nodes ({_data.num_nodes}). "
                      f"Rebuild with: py save_edge_attr_lookup.py")
        elif "data" in lookup:
            # Legacy DataFrame format — convert to inverted index on the fly
            print(f"[WARN] Legacy DataFrame lookup format detected. "
                  f"Consider re-running save_edge_attr_lookup.py for better performance.")
            legacy_df = lookup["data"]
            _edge_attr_lookup = {}
            for col in lookup.get("cols", legacy_df.columns.tolist()):
                if col not in legacy_df.columns:
                    continue
                col_data = legacy_df[col].dropna()
                groups = col_data.groupby(col_data).groups
                attr_index = {}
                for val, idx_index in groups.items():
                    idx_list = idx_index.tolist()
                    if len(idx_list) > 30:
                        continue
                    key = int(val) if isinstance(val, float) and val == int(val) else val
                    attr_index[key] = idx_list
                _edge_attr_lookup[col] = attr_index
            print(f"[OK] Converted legacy lookup: {len(legacy_df):,} nodes")
        else:
            print(f"[ERROR] edge_attr_lookup.pkl has unrecognized format. "
                  f"Keys: {list(lookup.keys())}. "
                  f"Re-run: py save_edge_attr_lookup.py")
            _edge_attr_lookup = None
    else:
        print(f"[WARNING] edge_attr_lookup.pkl not found at {EDGE_ATTR_LOOKUP_PATH}. "
              f"Graph attachment DISABLED — Dataset Scanner will use mini-graph only. "
              f"Run: py save_edge_attr_lookup.py")

    # ── Node metadata (real columns for the analytics endpoints) ──────────
    if NODE_META_PATH.exists():
        import pickle
        with open(NODE_META_PATH, "rb") as f:
            _node_meta = pickle.load(f)
        print(f"[OK] Node metadata loaded: {len(_node_meta):,} rows, "
              f"{len(_node_meta.columns)} columns")
        if _data is not None and len(_node_meta) != _data.num_nodes:
            print(f"[WARNING] node_meta rows ({len(_node_meta):,}) != graph "
                  f"nodes ({_data.num_nodes:,}). Re-run: py save_node_meta.py")
    else:
        print(f"[WARNING] node_meta.pkl not found at {NODE_META_PATH}. "
              f"Analytics endpoints will return 503 rather than synthetic "
              f"data. Run: py save_node_meta.py")

    # ── LightGBM + ensemble blend (optional) ──────────────────────────────
    if LGBM_PATH.exists():
        import pickle
        with open(LGBM_PATH, "rb") as f:
            _lgbm = pickle.load(f)
        print(f"[OK] LightGBM loaded "
              f"({getattr(_lgbm, 'best_iteration_', '?')} trees)")
    else:
        print(f"[i] LightGBM not found at {LGBM_PATH} — "
              f"run: py train_ensemble.py  (enables the ensemble)")

    if ENSEMBLE_PATH.exists():
        with open(ENSEMBLE_PATH) as f:
            _ensemble = json.load(f)
        w = _ensemble.get("weight_gnn")
        thr = _ensemble.get("thresholds", {})
        print(f"[OK] Ensemble config: {w:.0%} GNN + {1-w:.0%} LightGBM · "
              f"thresholds gnn={thr.get('gnn')} lgbm={thr.get('lightgbm')} "
              f"ens={thr.get('ensemble')}")
        # Prefer the strongest available variant by default.
        if _lgbm is not None and _model is not None:
            _active_model = "ensemble"
            print(f"[OK] Active model: ensemble "
                  f"(best validation PR-AUC of the three)")

    _is_loaded = True
    print("[OK] ML stack ready.")
