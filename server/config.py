"""
config.py - Central application configuration
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # repo root, wherever it is checked out
SERVER_DIR = Path(__file__).parent

GRAPH_PATH         = ROOT / "fraud_graph.pt"
MODEL_PATH         = ROOT / "outputs" / "models" / "best_model.pt"
PREPROCESSOR_PATH      = ROOT / "outputs" / "models" / "preprocessor.pkl"
EDGE_ATTR_LOOKUP_PATH  = ROOT / "outputs" / "models" / "edge_attr_lookup.pkl"
NODE_META_PATH         = ROOT / "outputs" / "models" / "node_meta.pkl"
LGBM_PATH              = ROOT / "outputs" / "models" / "lightgbm.pkl"
ENSEMBLE_PATH          = ROOT / "outputs" / "models" / "ensemble.json"
LOG_PATH           = ROOT / "outputs" / "results" / "training_log.json"
BASELINE_PATH      = ROOT / "outputs" / "results" / "baseline_metrics.csv"
PLOTS_DIR     = ROOT / "outputs" / "plots"
STATIC_DIR    = SERVER_DIR / "static"
UPLOADS_DIR   = SERVER_DIR / "uploads"
REPORTS_DIR   = SERVER_DIR / "reports"
DB_PATH       = SERVER_DIR / "fraud_app.db"

# Ensure dirs exist
for d in [STATIC_DIR, UPLOADS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── JWT ────────────────────────────────────────────────────────────────────
JWT_SECRET_KEY    = os.getenv("JWT_SECRET_KEY", "graphsage-fraud-secret-key-2024-msc")
JWT_ALGORITHM     = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours for demo

# ── Model defaults ─────────────────────────────────────────────────────────
# Only used when a checkpoint predates a config key. These MUST match the
# defaults in train.py -- when they drifted apart previously the server silently
# built a different network from the one that was trained.
MODEL_DEFAULTS = {
    "hidden1": 256,
    "hidden2": 128,
    "hidden3": 64,
    "mlp_hidden": 64,
    "dropout1": 0.2,
    "dropout2": 0.2,
    "dropout3": 0.2,
    "num_layers": 2,
    "use_skip": True,
}

# ── Application defaults ───────────────────────────────────────────────────
# Fallback only. The real threshold is tuned on the validation set by train.py
# and stored in best_model.pt; loader.get_threshold() returns that value.
# Hardcoding 0.5 here was the direct cause of "the model never flags anything":
# the trained model's probabilities peaked around 0.65 with a median of 0.19.
FALLBACK_FRAUD_THRESHOLD = 0.5
DEFAULT_FRAUD_THRESHOLD = FALLBACK_FRAUD_THRESHOLD
APP_VERSION = "1.0.0"
APP_TITLE   = "FraudShield AI"
APP_DESC    = "GraphSAGE-powered Financial Fraud Detection Platform"
