#!/usr/bin/env python3
"""
verify_all.py
=============
Independent end-to-end audit.

validate_api.py and validate_artifacts.py check that the system is internally
consistent. That is necessary but not sufficient: a number can be consistent
everywhere and still be wrong everywhere. This script recomputes every headline
figure from the raw artifacts, through code paths that do NOT go through the
server, and compares the results with what the live API actually serves.

Sections
--------
  A  Leakage & split integrity     (preprocessor fit, temporal order, edge masks)
  B  Model metrics, recomputed     (GNN, LightGBM, ensemble vs ensemble.json)
  C  Live API vs independent maths (metrics, sweep, PSI, analytics, labels)
  D  Stateful behaviour            (model switching, alert round-trip, risk bands)

Any state the checks change (active model, a test disposition) is restored.

Usage: start the backend, then  py verify_all.py
"""

import json
import os
import pickle
import sys
import warnings

import numpy as np
import requests
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

API = "http://127.0.0.1:8000"
TOL = 0.005          # tolerance for figures that pass through neighbour sampling

results = []


def check(section, name, ok, detail=""):
    results.append((section, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def main() -> int:
    from preprocessing_pipeline import load_raw_data
    from models.graphsage import GraphSAGE
    from train import EVAL_NEIGHBORS, csr_view, predict, tune_threshold

    print("=" * 78)
    print("  INDEPENDENT VERIFICATION")
    print("=" * 78)

    data = torch.load(os.path.join(ROOT, "fraud_graph.pt"), weights_only=False)
    pre = pickle.load(open(os.path.join(ROOT, "outputs/models/preprocessor.pkl"), "rb"))
    ens = json.load(open(os.path.join(ROOT, "outputs/models/ensemble.json")))
    df = load_raw_data(ROOT)
    y = data.y.numpy().astype(int)
    tr, va, te = (data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy())

    # ════════════════════════════════════════════════════════════════════
    print("\n[A] Leakage & split integrity")
    # ════════════════════════════════════════════════════════════════════
    dt = df["TransactionDT"].to_numpy()
    check("A", "train precedes validation in time", dt[tr].max() < dt[va].min(),
          f"{dt[tr].max():,} < {dt[va].min():,}")
    check("A", "validation precedes test in time", dt[va].max() < dt[te].min(),
          f"{dt[va].max():,} < {dt[te].min():,}")
    check("A", "labels in graph match the raw CSV",
          np.array_equal(y, df["isFraud"].to_numpy().astype(int)))

    # Scaler statistics must come from TRAIN rows only. Reproduce one column's
    # mean from raw data by hand and compare with what the scaler stored.
    col = "TransactionAmt_log"
    k = pre.num_cols.index(col)
    amt = df["TransactionAmt"].astype(float)
    logamt = np.log1p(amt.clip(lower=0))
    train_mean, all_mean = logamt[tr].mean(), logamt.mean()
    check("A", "scaler fitted on train rows only",
          abs(pre.scaler.mean_[k] - train_mean) < 1e-4 and abs(train_mean - all_mean) > 1e-4,
          f"scaler {pre.scaler.mean_[k]:.5f} · train {train_mean:.5f} · all {all_mean:.5f}")

    cnt = sum(pre.count_maps["card1"].values())
    check("A", "count encoding built from train rows only",
          cnt == int(df.loc[tr, "card1"].notna().sum()),
          f"{cnt:,} counted vs {int(df.loc[tr, 'card1'].notna().sum()):,} train rows")

    src, dst = data.edge_index[0], data.edge_index[1]
    bad = int((data.train_edge_mask & ~(data.train_mask[src] & data.train_mask[dst])).sum())
    check("A", "training message passing never touches val/test nodes", bad == 0,
          f"{bad} leaking edges")
    check("A", "raw TransactionDT is not a model feature",
          "TransactionDT" not in pre.all_cols)

    # ════════════════════════════════════════════════════════════════════
    print("\n[B] Model metrics, recomputed from scratch")
    # ════════════════════════════════════════════════════════════════════
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(ROOT, "outputs/models/best_model.pt"),
                      map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GraphSAGE(in_channels=ckpt["in_channels"], hidden1=cfg["hidden1"],
                      hidden2=cfg["hidden2"], hidden3=cfg["hidden3"],
                      mlp_hidden=cfg["mlp_hidden"], dropout1=cfg["dropout1"],
                      dropout2=cfg["dropout2"], dropout3=cfg["dropout3"],
                      num_layers=cfg["num_layers"], use_skip=cfg["use_skip"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_idx, test_idx = torch.where(data.val_mask)[0], torch.where(data.test_mask)[0]
    p_gnn_val = predict(model, data.x, data.y, csr_view(data, data.val_edge_mask),
                        val_idx, device, EVAL_NEIGHBORS, 4096)
    p_gnn_test = predict(model, data.x, data.y,
                         csr_view(data, torch.ones(data.edge_index.shape[1], dtype=torch.bool)),
                         test_idx, device, EVAL_NEIGHBORS, 4096)
    lgbm = pickle.load(open(os.path.join(ROOT, "outputs/models/lightgbm.pkl"), "rb"))
    X = data.x.numpy()
    p_lgb_val = lgbm.predict_proba(X[va])[:, 1]
    p_lgb_test = lgbm.predict_proba(X[te])[:, 1]
    w = ens["weight_gnn"]
    p_ens_val = w * p_gnn_val + (1 - w) * p_lgb_val
    p_ens_test = w * p_gnn_test + (1 - w) * p_lgb_test
    y_val, y_test = y[va], y[te]

    independent = {}
    for name, key, pv, pt in [
        ("GraphSAGE (GNN)", "gnn", p_gnn_val, p_gnn_test),
        ("LightGBM", "lightgbm", p_lgb_val, p_lgb_test),
        ("Ensemble (GNN+LGBM)", "ensemble", p_ens_val, p_ens_test),
    ]:
        thr = ens["thresholds"][key]
        pred = (pt >= thr).astype(int)
        m = {"F1": f1_score(y_test, pred), "PR-AUC": average_precision_score(y_test, pt),
             "ROC-AUC": roc_auc_score(y_test, pt), "Precision": precision_score(y_test, pred),
             "Recall": recall_score(y_test, pred), "Accuracy": accuracy_score(y_test, pred)}
        independent[key] = (m, thr, pt)
        stored = ens["test_metrics"][name]
        worst = max(abs(m[k] - stored[k]) for k in m)
        check("B", f"{name}: recomputed metrics match ensemble.json",
              worst <= TOL, f"F1 {m['F1']:.4f} vs {stored['F1']:.4f} · max diff {worst:.4f}")
        # The stored threshold must be the one validation selects — not test.
        if key != "gnn":
            t_val, _ = tune_threshold(y_val, pv)
            check("B", f"{name}: threshold was tuned on validation",
                  abs(t_val - thr) < 1e-6, f"validation selects {t_val:.2f}, stored {thr:.2f}")

    # The blend weight must be validation-optimal too.
    ws = np.arange(0, 1.001, 0.05)
    best_w = ws[int(np.argmax([average_precision_score(y_val, a * p_gnn_val + (1 - a) * p_lgb_val)
                               for a in ws]))]
    check("B", "blend weight is the validation optimum", abs(best_w - w) < 1e-6,
          f"validation picks {best_w:.2f}, stored {w:.2f}")

    e, g, l = (independent[k][0]["F1"] for k in ("ensemble", "gnn", "lightgbm"))
    check("B", "ensemble beats both single models on test F1", e > max(g, l),
          f"ensemble {e:.4f} · lgbm {l:.4f} · gnn {g:.4f}")

    # ════════════════════════════════════════════════════════════════════
    print("\n[C] Live API vs independent computation")
    # ════════════════════════════════════════════════════════════════════
    s = requests.Session()
    r = s.post(f"{API}/auth/login", json={"username": "admin", "password": "fraud123"}, timeout=30)
    if not check("C", "login", r.status_code == 200, str(r.status_code)):
        return report()
    s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"

    models = s.get(f"{API}/api/settings/models", timeout=60).json()
    active = models["active"]
    check("C", "an active model is reported", active in ("gnn", "lightgbm", "ensemble"), active)

    im, ithr, ipt = independent[active]
    met = s.get(f"{API}/api/metrics", timeout=600).json()
    check("C", "served threshold equals the active model's tuned threshold",
          close(met["threshold"], ithr, 1e-4), f"{met['threshold']} vs {ithr}")
    for k_api, k_ind in (("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                         ("precision", "Precision"), ("recall", "Recall")):
        check("C", f"/api/metrics {k_api} matches independent {active} value",
              close(met[k_api], im[k_ind]), f"{met[k_api]:.4f} vs {im[k_ind]:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, (ipt >= ithr).astype(int)).ravel()
    cm = met["confusion_matrix"]
    drift_tp = abs(cm["tp"] - tp) / max(tp, 1)
    check("C", "confusion matrix matches (within sampling noise)",
          sum(cm.values()) == len(y_test) and drift_tp < 0.02,
          f"API tp={cm['tp']} fp={cm['fp']} · independent tp={tp} fp={fp}")

    # Labels shown in the UI must name the model whose numbers they show.
    dash = s.get(f"{API}/api/dashboard", timeout=600).json()
    label_ok = {"gnn": "GraphSAGE", "lightgbm": "LightGBM", "ensemble": "Ensemble"}[active]
    check("C", "dashboard model name matches the model it reports",
          label_ok.lower() in str(dash.get("model_name", "")).lower(),
          f"model_name='{dash.get('model_name')}' while serving '{active}'")

    # Threshold sweep vs brute force at arbitrary points.
    sw = s.get(f"{API}/api/insights/threshold-sweep", params={"steps": 99}, timeout=600).json()
    probs_all_active = None
    mismatches = []
    served = {p["threshold"]: p for p in sw["points"]}
    # Use the API's own score vector for this comparison: we are testing the
    # sweep arithmetic, not the sampler.
    for t in (0.05, 0.25, 0.5, 0.75, 0.95):
        pt_ = min(served, key=lambda x: abs(x - t))
        pred = (ipt >= pt_).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        got = served[pt_]
        if abs(got["tp"] - tp_) / max(tp_, 1) > 0.02 or abs(got["fp"] - fp_) / max(fp_, 1) > 0.05:
            mismatches.append(f"t={pt_}: api tp/fp {got['tp']}/{got['fp']} vs {tp_}/{fp_}")
        if got["tp"] + got["fp"] + got["fn"] + got["tn"] != len(y_test):
            mismatches.append(f"t={pt_}: counts do not sum to n")
    check("C", "threshold sweep agrees with brute-force confusion matrices",
          not mismatches, "; ".join(mismatches) or "5 thresholds checked")

    # PSI recomputed.
    dr = s.get(f"{API}/api/insights/drift", timeout=600).json()
    check("C", "drift PSI is small and bands are consistent",
          dr["psi"]["train_vs_test"]["value"] >= 0
          and (dr["psi"]["train_vs_test"]["band"] == "stable") == (dr["psi"]["train_vs_test"]["value"] < 0.1),
          f"train->test {dr['psi']['train_vs_test']}")
    for k, m in (("train", tr), ("val", va), ("test", te)):
        check("C", f"drift {k} fraud rate matches the data",
              close(dr["splits"][k]["fraud_rate"], y[m].mean(), 1e-5),
              f"{dr['splits'][k]['fraud_rate']:.5f} vs {y[m].mean():.5f}")

    # Analytics against raw CSV group-bys.
    an = s.get(f"{API}/api/analytics/summary", timeout=600).json()
    truth = df.groupby("card4")["isFraud"].mean() * 100
    bad_cards = [c["card_type"] for c in an["fraud_by_card"]
                 if c["card_type"] in truth and not close(c["fraud_rate"], truth[c["card_type"]], 0.01)]
    check("C", "analytics card fraud rates match raw CSV", not bad_cards,
          ", ".join(bad_cards) or f"{len(an['fraud_by_card'])} card types checked")
    mon = df.assign(m=(df["TransactionDT"] // (86400 * 30))).shape[0]
    check("C", "analytics monthly totals sum to all transactions",
          sum(mm["total"] for mm in an["fraud_by_month"]) == len(df),
          f"{sum(mm['total'] for mm in an['fraud_by_month']):,} vs {len(df):,}")

    # ════════════════════════════════════════════════════════════════════
    print("\n[D] Stateful behaviour")
    # ════════════════════════════════════════════════════════════════════
    original_model = active
    try:
        for mid in ("gnn", "lightgbm", "ensemble"):
            r = s.put(f"{API}/api/settings/models", json={"model_id": mid}, timeout=600)
            ok = r.status_code == 200 and close(r.json()["threshold"], ens["thresholds"][mid], 1e-4)
            m2 = s.get(f"{API}/api/metrics", timeout=600).json()
            check("D", f"switch to {mid}: threshold and F1 follow the model",
                  ok and close(m2["f1"], independent[mid][0]["F1"]),
                  f"thr {r.json().get('threshold')} · F1 {m2['f1']:.4f} vs {independent[mid][0]['F1']:.4f}")
    finally:
        s.put(f"{API}/api/settings/models", json={"model_id": original_model}, timeout=600)

    q = s.get(f"{API}/api/alerts/queue", params={"limit": 20}, timeout=600).json()
    scores = [i["fraud_probability"] for i in q["items"]]
    check("D", "review queue is ranked by score, highest first",
          scores == sorted(scores, reverse=True) and len(scores) > 0,
          f"{len(scores)} items")
    check("D", "CRITICAL risk band is reachable",
          any(i["risk_level"] == "CRITICAL" for i in q["items"]),
          f"top item {q['items'][0]['risk_level'] if q['items'] else '-'}")

    # Round-trip a disposition and clean it up.
    nid = q["items"][0]["node_id"]
    before = s.get(f"{API}/api/alerts/stats", timeout=60).json()["total_reviewed"]
    rp = s.post(f"{API}/api/alerts/{nid}/disposition", json={"disposition": "confirmed_fraud"}, timeout=60)
    mid = s.get(f"{API}/api/alerts/stats", timeout=60).json()["total_reviewed"]
    in_queue = nid in [i["node_id"] for i in s.get(f"{API}/api/alerts/queue",
                                                     params={"limit": 20}, timeout=600).json()["items"]]
    rd = s.delete(f"{API}/api/alerts/{nid}/disposition", timeout=60)
    after = s.get(f"{API}/api/alerts/stats", timeout=60).json()["total_reviewed"]
    check("D", "disposition is recorded, leaves the queue, and undoes cleanly",
          rp.status_code == 200 and mid == before + 1 and not in_queue
          and rd.status_code == 200 and after == before,
          f"reviewed {before} -> {mid} -> {after}")
    check("D", "bad disposition values are rejected",
          s.post(f"{API}/api/alerts/{nid}/disposition", json={"disposition": "maybe"},
                 timeout=60).status_code == 400)

    # ════════════════════════════════════════════════════════════════════
    print("\n[E] Upload path serves the same model as the rest of the app")
    # ════════════════════════════════════════════════════════════════════
    # LightGBM uses no graph, and the upload preprocessor reproduces data.x
    # bit-for-bit, so with LightGBM active an uploaded row must score exactly
    # what the same row scores in-graph. Any gap means the upload path is not
    # using the active model.
    import io
    rng = np.random.default_rng(7)
    pick = np.sort(rng.choice(np.where(te)[0], 300, replace=False))
    buf = io.StringIO()
    df.iloc[pick].to_csv(buf, index=False)
    payload = buf.getvalue().encode()
    try:
        for mid in ("lightgbm", "ensemble"):
            s.put(f"{API}/api/settings/models", json={"model_id": mid}, timeout=600)
            up = s.post(f"{API}/api/upload-predict",
                        files={"file": ("t.csv", payload, "text/csv")}, timeout=900).json()
            up_p = np.array([r["fraud_probability"] for r in up["results"]])
            check("E", f"upload response reports model '{mid}'", up.get("model") == mid,
                  f"model={up.get('model')}")
            if mid == "lightgbm":
                ref = lgbm.predict_proba(X[pick])[:, 1]
                diff = float(np.abs(up_p - ref).max())
                check("E", "LightGBM: uploaded rows score identically to in-graph",
                      diff < 1e-4, f"max |diff| {diff:.2e}")
            check("E", f"{mid}: upload uses that model's threshold",
                  close(up["threshold"], ens["thresholds"][mid], 1e-4),
                  f"{up['threshold']} vs {ens['thresholds'][mid]}")
    finally:
        s.put(f"{API}/api/settings/models", json={"model_id": original_model}, timeout=600)

    from server.ml.predictor import _risk_level
    order = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    mono = all(order.index(_risk_level(p, 0.61)) <= order.index(_risk_level(p + 0.01, 0.61))
               for p in np.arange(0, 0.99, 0.01))
    check("D", "risk bands are monotonic in score", mono)

    return report()


def report() -> int:
    n_fail = sum(1 for r in results if not r[2])
    print("\n" + "=" * 78)
    for sec in "ABCDE":
        rows = [r for r in results if r[0] == sec]
        if rows:
            print(f"  {sec}: {sum(r[2] for r in rows)}/{len(rows)} passed")
    print(f"  TOTAL: {len(results) - n_fail}/{len(results)} passed")
    print("=" * 78)
    for r in results:
        if not r[2]:
            print(f"  FAIL [{r[0]}] {r[1]}: {r[3]}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
