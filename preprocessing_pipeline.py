#!/usr/bin/env python3
"""
preprocessing_pipeline.py
==========================
THE single source of truth for turning raw IEEE-CIS transaction rows into the
model's node-feature matrix.

Both build_graph.py (training) and server/ml/pipeline.py (inference) call this
same fitted object, so training and serving can no longer drift apart.

What changed vs. the previous version
-------------------------------------
1. `fit()` now takes a `fit_mask`. Imputer medians, ordinal-encoder categories,
   count-encoding tables and scaler statistics are learned from TRAINING ROWS
   ONLY. Previously every statistic was fit on all 590,540 rows, so val/test
   information leaked into the features the model was scored on.

2. Raw `TransactionDT` is dropped and replaced by cyclical time features.
   `TransactionDT` is a monotonically increasing offset in seconds; under a
   chronological split its test-set values lie entirely outside the training
   range, so keeping it as a raw feature is both leaky (random split) and
   useless (time split).

3. Added the feature engineering that actually moves the needle on IEEE-CIS:
   log amount, the cents/decimal part of the amount, and count ("frequency")
   encoding of the high-cardinality identity columns.

4. Categoricals keep an ordinal code (for tree baselines) but every
   high-cardinality column also gets a count code, because a raw ordinal index
   imposes a meaningless ordering on a neural network.

Artifact: outputs/models/preprocessor.pkl
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(ROOT, "outputs", "models", "preprocessor.pkl")

# ── Attributes used to wire transactions together in the graph ──────────────
# (kept here so build_graph.py and the server import one definition)
EDGE_ATTRIBUTES: dict = {
    "card1":         3.0,
    "card2":         3.0,
    "DeviceInfo":    3.0,
    "P_emaildomain": 2.0,
    "addr1":         2.0,
    "addr2":         2.0,
    "R_emaildomain": 2.0,
    "DeviceType":    1.0,
    "id_30":         1.0,
    "id_31":         1.0,
}

# Columns that get a count/frequency code in addition to their ordinal code.
# These are the identity-like columns where "how often has this value been seen"
# is far more informative than "which arbitrary integer did the encoder assign".
COUNT_ENCODE_COLS = [
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType",
    "id_30", "id_31", "id_33", "ProductCD",
]

# Never feed these to the model.
DROP_COLS = ["TransactionID", "isFraud", "TransactionDT"]

# Group-aggregate features. For each (key, column) pair the training mean and
# standard deviation of `column` within each value of `key` become features, as
# does the row's ratio to that mean.
#
# This is the strongest signal on IEEE-CIS that raw columns do not carry: a
# $900 charge is unremarkable in general but very unusual on a card whose
# historical mean is $40. Statistics are computed from training rows only, so
# they cannot leak.
AGG_SPECS = [
    ("card1",        ["TransactionAmt", "D15", "D4"]),
    ("card1_addr1",  ["TransactionAmt", "D15"]),
    ("P_emaildomain", ["TransactionAmt"]),
]

# Composite keys built before aggregation.
COMPOSITE_KEYS = {
    "card1_addr1": ["card1", "addr1"],
}


class FraudPreprocessor:
    """
    Fit once on training rows, then apply to anything.

    Attributes exposed for downstream consumers
    -------------------------------------------
    input_cols : raw column names the fit dataframe had (used to validate uploads)
    all_cols   : engineered feature names, in the exact order of the output matrix
    num_cols   : engineered numeric feature names
    cat_cols   : engineered categorical feature names
    n_features : len(all_cols)
    """

    def __init__(self, add_time_features: bool = True, count_encode: bool = True):
        self.add_time_features = add_time_features
        self.count_encode = count_encode

        self.num_imp = SimpleImputer(strategy="median")
        self.cat_imp = SimpleImputer(strategy="constant", fill_value="__missing__")
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32
        )
        self.scaler = StandardScaler()

        self.input_cols: list = []
        self.num_cols: list = []
        self.cat_cols: list = []
        self.all_cols: list = []
        self.count_maps: dict = {}
        self.count_default: dict = {}
        self.agg_maps: dict = {}      # (key, col, stat) -> {key value: stat}
        self.agg_global: dict = {}    # (col, stat) -> training-wide fallback
        self.n_features: int | None = None
        self.fitted: bool = False

    # ── Input canonicalisation ─────────────────────────────────────────────
    @staticmethod
    def _canonical(df: pd.DataFrame) -> pd.DataFrame:
        """
        Put raw inputs into the exact dtypes the models were trained on.

        load_raw_data() reads TransactionAmt as float32; an uploaded CSV parses
        it as float64. The derived `TransactionAmt_decimal` (amt - floor(amt))
        exposes float32 rounding on larger amounts, so the same transaction got
        a slightly different feature at upload time than in training — enough
        to move LightGBM's score by up to 7e-3. Casting here makes transform()
        independent of how the caller happened to parse the file.

        Training input is already float32, so this is a no-op on the fitted
        statistics and on data.x.
        """
        if "TransactionAmt" not in df.columns:
            return df
        if df["TransactionAmt"].dtype == np.float32:
            return df
        out = df.copy()
        out["TransactionAmt"] = pd.to_numeric(out["TransactionAmt"],
                                              errors="coerce").astype(np.float32)
        return out

    # ── Feature engineering ────────────────────────────────────────────────
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive features from raw columns. Pure function of a single row, so it
        is identical at train and inference time and cannot leak.
        """
        out = df.copy()

        if self.add_time_features and "TransactionDT" in out.columns:
            dt = pd.to_numeric(out["TransactionDT"], errors="coerce")
            # IEEE-CIS TransactionDT is a second offset from a fixed reference.
            out["TX_hour"] = (dt // 3600) % 24
            out["TX_dayofweek"] = (dt // 86400) % 7
            # Cyclical encoding so 23:00 and 00:00 are adjacent.
            out["TX_hour_sin"] = np.sin(2 * np.pi * out["TX_hour"] / 24)
            out["TX_hour_cos"] = np.cos(2 * np.pi * out["TX_hour"] / 24)

        if "TransactionAmt" in out.columns:
            amt = pd.to_numeric(out["TransactionAmt"], errors="coerce")
            out["TransactionAmt_log"] = np.log1p(amt.clip(lower=0))
            # Long decimal tails are a well-known fraud tell on this dataset:
            # they come from foreign-currency conversion.
            out["TransactionAmt_decimal"] = (amt - np.floor(amt)).round(6)

        # Drop the raw timestamp and any label/id columns.
        out = out.drop(columns=[c for c in DROP_COLS if c in out.columns],
                       errors="ignore")
        return out

    @staticmethod
    def _composite(df: pd.DataFrame, name: str) -> pd.Series:
        """Build a composite grouping key, e.g. card1_addr1."""
        parts = COMPOSITE_KEYS[name]
        if not all(p in df.columns for p in parts):
            return pd.Series([np.nan] * len(df), index=df.index)
        out = df[parts[0]].astype(str)
        for p in parts[1:]:
            out = out + "_" + df[p].astype(str)
        return out

    def _key_series(self, df: pd.DataFrame, key: str) -> pd.Series:
        if key in COMPOSITE_KEYS:
            return self._composite(df, key)
        if key in df.columns:
            return df[key]
        return pd.Series([np.nan] * len(df), index=df.index)

    def _apply_aggregates(self, raw: pd.DataFrame,
                          out: pd.DataFrame) -> pd.DataFrame:
        """Attach the group-aggregate features to `out`, keyed off `raw`."""
        for key, cols in AGG_SPECS:
            ks = self._key_series(raw, key).astype(str)
            for col in cols:
                if col not in raw.columns:
                    continue
                vals = pd.to_numeric(raw[col], errors="coerce")
                for stat in ("mean", "std"):
                    mapping = self.agg_maps.get((key, col, stat))
                    if mapping is None:
                        continue
                    mapped = ks.map(mapping)
                    fallback = self.agg_global.get((col, stat), 0.0)
                    mapped = mapped.fillna(fallback).astype(np.float32)
                    out[f"{col}_{key}_{stat}"] = mapped
                    if stat == "mean":
                        # How far this transaction sits from what is normal for
                        # this card / address / domain.
                        denom = mapped.replace(0, np.nan)
                        out[f"{col}_{key}_ratio"] = (
                            (vals / denom).replace([np.inf, -np.inf], np.nan)
                            .astype(np.float32)
                        )
        return out

    def _apply_count_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map each configured column's values to their training frequency."""
        if not self.count_encode:
            return df
        out = df
        for col, mapping in self.count_maps.items():
            if col in out.columns:
                codes = out[col].map(mapping)
            else:
                codes = pd.Series(np.nan, index=out.index)
            # A value never seen in training gets count 0 — itself a signal.
            out[f"{col}_count"] = codes.fillna(0.0).astype(np.float32)
        return out

    # ── Fit ────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, fit_mask: np.ndarray | None = None):
        """
        Learn every statistic from the rows selected by `fit_mask`.

        Parameters
        ----------
        df       : raw merged dataframe (may still contain TransactionID/isFraud)
        fit_mask : boolean array over df's rows. None means "use every row",
                   which is only correct when df already is the training split.
        """
        self.input_cols = [c for c in df.columns if c not in ("TransactionID", "isFraud")]

        if fit_mask is None:
            fit_mask = np.ones(len(df), dtype=bool)
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.sum() == 0:
            raise ValueError("fit_mask selects zero rows")

        train_raw = self._canonical(df.loc[fit_mask])

        # Count maps must come from training rows only.
        self.count_maps = {}
        if self.count_encode:
            for col in COUNT_ENCODE_COLS:
                if col in train_raw.columns:
                    vc = train_raw[col].value_counts(dropna=True)
                    self.count_maps[col] = vc.to_dict()

        # Group aggregates, learned from training rows only.
        self.agg_maps = {}
        self.agg_global = {}
        for key, cols in AGG_SPECS:
            ks = self._key_series(train_raw, key).astype(str)
            for col in cols:
                if col not in train_raw.columns:
                    continue
                vals = pd.to_numeric(train_raw[col], errors="coerce")
                grouped = vals.groupby(ks)
                self.agg_maps[(key, col, "mean")] = grouped.mean().to_dict()
                self.agg_maps[(key, col, "std")] = grouped.std().fillna(0.0).to_dict()
                self.agg_global[(col, "mean")] = float(vals.mean())
                self.agg_global[(col, "std")] = float(vals.std())

        feats = self._engineer(train_raw)
        feats = self._apply_count_encoding(feats)
        feats = self._apply_aggregates(train_raw, feats)

        self.cat_cols = feats.select_dtypes(include="object").columns.tolist()
        self.num_cols = feats.select_dtypes(include=np.number).columns.tolist()
        self.all_cols = feats.columns.tolist()

        if self.num_cols:
            self.num_imp.fit(feats[self.num_cols])
            feats[self.num_cols] = self.num_imp.transform(feats[self.num_cols])
            self.scaler.fit(feats[self.num_cols])

        if self.cat_cols:
            self.cat_imp.fit(feats[self.cat_cols].astype(object))
            imputed = self.cat_imp.transform(feats[self.cat_cols].astype(object))
            self.encoder.fit(imputed)

        self.n_features = len(self.all_cols)
        self.fitted = True
        return self

    # ── Transform ──────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline. Returns float32 (N, n_features)."""
        if not self.fitted:
            raise RuntimeError("FraudPreprocessor.transform() called before fit()")

        df = self._canonical(df)
        feats = self._engineer(df)
        feats = self._apply_count_encoding(feats)
        feats = self._apply_aggregates(df, feats)

        # Align to the training schema: add anything absent, drop anything extra,
        # and restore the exact column order the model was trained on.
        for col in self.all_cols:
            if col not in feats.columns:
                feats[col] = np.nan
        feats = feats[self.all_cols]

        # Ensure compatibility across scikit-learn versions
        for imp in (self.num_imp, self.cat_imp):
            if hasattr(imp, "_fit_dtype") and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = imp._fit_dtype

        if self.num_cols:
            # Coerce first: a column read as text from a CSV must not silently
            # become a garbage ordinal.
            num_block = feats[self.num_cols].apply(pd.to_numeric, errors="coerce")
            num_block = pd.DataFrame(
                self.num_imp.transform(num_block),
                columns=self.num_cols, index=feats.index,
            )
            feats[self.num_cols] = self.scaler.transform(num_block)

        if self.cat_cols:
            cat_block = feats[self.cat_cols].astype(object)
            cat_block = self.cat_imp.transform(cat_block)
            feats[self.cat_cols] = self.encoder.transform(cat_block)

        return feats[self.all_cols].values.astype(np.float32)

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str = PIPELINE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        print(f"[Preprocessor] Saved to {path}  ({self.n_features} features)")

    @staticmethod
    def load(path: str = PIPELINE_PATH) -> "FraudPreprocessor":
        with open(path, "rb") as f:
            p = pickle.load(f)
        for imp in (getattr(p, "num_imp", None), getattr(p, "cat_imp", None)):
            if imp is not None and hasattr(imp, "_fit_dtype") and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = imp._fit_dtype
        print(f"[Preprocessor] Loaded from {path}")
        return p


def load_raw_data(root: str = ROOT) -> pd.DataFrame:
    """Load + merge + de-duplicate the IEEE-CIS CSVs. Used by every entry point."""
    tx = pd.read_csv(
        os.path.join(root, "train_transaction.csv"),
        dtype={"TransactionID": "int32", "isFraud": "int8", "TransactionAmt": "float32"},
        low_memory=False,
    )
    idt = pd.read_csv(
        os.path.join(root, "train_identity.csv"),
        dtype={"TransactionID": "int32"}, low_memory=False,
    )
    df = tx.merge(idt, on="TransactionID", how="left")
    return df.drop_duplicates(subset="TransactionID").reset_index(drop=True)


if __name__ == "__main__":
    # Standalone refit. build_graph.py normally does this itself so that the
    # graph and the preprocessor can never disagree.
    print("Loading training data to fit preprocessor...")
    df = load_raw_data()
    print(f"Fitting on {len(df):,} rows x {df.shape[1]} cols "
          f"(NOTE: no fit_mask -> fits on everything; "
          f"run build_graph.py for the leakage-free version)")
    p = FraudPreprocessor().fit(df)
    p.save()
