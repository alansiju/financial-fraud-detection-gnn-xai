#!/usr/bin/env python3
"""
validate_api.py
===============
End-to-end validation of the running backend.

Starts nothing: point it at a server that is already up (start_backend.bat or
`uvicorn server.main:app`). It logs in, calls every endpoint, and asserts the
responses are real rather than merely well-formed.

The checks that matter are the ones that would have passed before the fixes
while the app was still wrong:

  * /api/metrics used to fall back to invented numbers on any exception, so it
    always returned 200. The check now asserts the reported threshold matches
    the model's and that the confusion matrix sums to the test-split size.
  * /api/dashboard did the same via _mock_dashboard().
  * /api/analytics returned rng.uniform() breakdowns that looked plausible.
    The check asserts the card-type fraud rates match the known real ones
    (discover ~7.7%, amex ~2.9%), which random data cannot reproduce.
  * /api/transactions listed invented devices and 2019 timestamps. The check
    asserts device values are the real IEEE-CIS set.
  * The Settings threshold slider was never read by prediction code. The check
    changes it and asserts predictions actually change.

Usage
-----
    py validate_api.py
    py validate_api.py --base http://127.0.0.1:8000 --user admin --password fraud123
"""

import argparse
import json
import sys

import requests

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name: str, ok: bool, detail: str = "", level: str = FAIL):
    status = PASS if ok else level
    results.append((status, name, detail))
    mark = {PASS: "  [PASS]", FAIL: "  [FAIL]", WARN: "  [WARN]"}[status]
    print(f"{mark} {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _find_connected_node(session, base, tries=40):
    """Return a node id whose depth-1 neighbourhood is non-empty."""
    for nid in range(0, tries * 50, 50):
        r = session.get(f"{base}/api/graph/{nid}", params={"depth": 1}, timeout=120)
        if r.status_code == 200 and r.json().get("node_count", 0) > 1:
            return nid
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="fraud123")
    args = ap.parse_args()
    B = args.base.rstrip("/")

    print("=" * 74)
    print(f"  API VALIDATION  ->  {B}")
    print("=" * 74)

    s = requests.Session()

    # ── Auth ────────────────────────────────────────────────────────────────
    print("\n[auth]")
    try:
        r = s.post(f"{B}/auth/login",
                   json={"username": args.user, "password": args.password},
                   timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"\n  Cannot reach {B}. Start the backend first:")
        print("    .venv\\Scripts\\python -m uvicorn server.main:app --port 8000")
        return 2

    if not check("login returns 200", r.status_code == 200,
                 f"got {r.status_code}: {r.text[:120]}"):
        return 1
    token = r.json().get("access_token")
    check("login returns a token", bool(token))
    s.headers["Authorization"] = f"Bearer {token}"

    # ── Dashboard ───────────────────────────────────────────────────────────
    print("\n[dashboard]")
    r = s.get(f"{B}/api/dashboard", timeout=300)
    check("GET /api/dashboard -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        d = r.json()
        check("dashboard reports real node count",
              d.get("graph_nodes") == 590540, f"graph_nodes={d.get('graph_nodes')}")
        check("dashboard is not the old mock",
              not (d.get("model_accuracy") == 0.974 and d.get("roc_auc") == 0.896),
              "returned the exact former mock values")
        check("dashboard exposes the active threshold",
              isinstance(d.get("threshold"), (int, float)),
              f"threshold={d.get('threshold')}")
        # The name must describe the model actually serving. The old assertion
        # (startswith "GraphSAGE-") encoded the mislabelling bug: with the
        # ensemble active, the ensemble's numbers were shown under the GNN name.
        served = s.get(f"{B}/api/settings/models", timeout=60).json()
        expected = next((m["name"] for m in served.get("models", [])
                         if m["id"] == served.get("active")), None)
        check("model name matches the model actually serving",
              expected is not None and d.get("model_name") == expected,
              f"model_name={d.get('model_name')} active={served.get('active')}")
        model_thr = d.get("threshold")

    # ── Metrics ─────────────────────────────────────────────────────────────
    print("\n[metrics]")
    r = s.get(f"{B}/api/metrics", timeout=300)
    check("GET /api/metrics -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        m = r.json()
        cm = m.get("confusion_matrix", {})
        total = sum(cm.get(k, 0) for k in ("tn", "fp", "fn", "tp"))
        check("confusion matrix covers the whole test split",
              total == 88581, f"sums to {total:,}, expected 88,581")
        check("metrics use the tuned threshold, not 0.5",
              abs(m.get("threshold", 0.5) - model_thr) < 1e-6,
              f"metrics={m.get('threshold')} dashboard={model_thr}")
        check("metrics are not the old mock",
              not (m.get("accuracy") == 0.974 and m.get("f1") == 0.76),
              "returned the exact former mock values")
        check("split mode is recorded", m.get("split_mode") == "temporal",
              f"split_mode={m.get('split_mode')}")
        check("baselines include gradient boosting",
              any("XGBoost" in str(b.get("Model", "")) for b in m.get("baselines", [])),
              "baseline list: "
              + ", ".join(str(b.get("Model")) for b in m.get("baselines", [])))

    # ── Analytics ───────────────────────────────────────────────────────────
    print("\n[analytics]")
    r = s.get(f"{B}/api/analytics/summary", timeout=300)
    check("GET /api/analytics/summary -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        a = r.json()
        check("analytics declares a real source",
              "node_meta" in str(a.get("source", "")), f"source={a.get('source')}")
        cards = {c["card_type"]: c for c in a.get("fraud_by_card", [])}
        # Ground truth from the raw CSV: discover 7.73%, amex 2.87%.
        disc = cards.get("discover", {}).get("fraud_rate")
        amex = cards.get("american express", {}).get("fraud_rate")
        check("discover fraud rate matches the real data (7.73%)",
              disc is not None and abs(disc - 7.73) < 0.1, f"got {disc}")
        check("amex fraud rate matches the real data (2.87%)",
              amex is not None and abs(amex - 2.87) < 0.1, f"got {amex}")
        devs = {d["device"] for d in a.get("fraud_by_device", [])}
        check("device categories are the real IEEE-CIS set",
              devs and devs <= {"desktop", "mobile", "(missing)"},
              f"got {sorted(devs)}")
        months = [m["month"] for m in a.get("fraud_by_month", [])]
        check("months are real dates, not a fixed Jan-Dec list",
              bool(months) and any(ch.isdigit() for ch in months[0]),
              f"first month={months[0] if months else None}")

    # ── Transactions ────────────────────────────────────────────────────────
    print("\n[transactions]")
    r = s.get(f"{B}/api/transactions", params={"page": 1, "limit": 5}, timeout=300)
    check("GET /api/transactions -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        t = r.json()
        rows = t.get("data", [])
        check("returns rows", len(rows) == 5, f"got {len(rows)}")
        if rows:
            row = rows[0]
            check("card types are real",
                  row.get("card_type") in
                  {"visa", "mastercard", "discover", "american express", "(unknown)"},
                  f"card_type={row.get('card_type')}")
            check("timestamps are not the invented 2019 dates",
                  not str(row.get("timestamp", "")).startswith("2019-"),
                  f"timestamp={row.get('timestamp')}")
            check("amount is a plausible transaction value",
                  isinstance(row.get("amount"), (int, float))
                  and 0 < row["amount"] < 100000,
                  f"amount={row.get('amount')}")

    r = s.get(f"{B}/api/transactions/facets", timeout=300)
    check("GET /api/transactions/facets -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        f = r.json()
        check("facets expose real device values",
              all(d["value"] in {"desktop", "mobile", "(missing)"}
                  for d in f.get("devices", [])),
              f"devices={[d['value'] for d in f.get('devices', [])]}")

    # ── Predict ─────────────────────────────────────────────────────────────
    print("\n[predict]")
    r = s.post(f"{B}/api/predict/node", json={"node_id": 4340}, timeout=300)
    check("POST /api/predict/node -> 200", r.status_code == 200, r.text[:160])
    base_prob = None
    if r.status_code == 200:
        p = r.json()
        base_prob = p.get("fraud_probability")
        check("prediction carries a probability",
              isinstance(base_prob, (int, float)), f"got {base_prob}")
        check("prediction uses the tuned threshold",
              abs(p.get("threshold", 0.5) - model_thr) < 1e-6,
              f"threshold={p.get('threshold')}")

    r = s.post(f"{B}/api/predict/batch",
               json={"node_ids": list(range(100, 140))}, timeout=300)
    check("POST /api/predict/batch -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        b = r.json()
        check("batch returns all requested nodes",
              b.get("total") == 40, f"total={b.get('total')}")

    # ── Graph ───────────────────────────────────────────────────────────────
    print("\n[graph]")
    # Pick a node that genuinely has neighbours -- roughly 13% of nodes are
    # isolated, and an isolated node would make this check vacuous.
    hub = _find_connected_node(s, B)
    r = s.get(f"{B}/api/graph/{hub}", params={"depth": 2}, timeout=300)
    check("GET /api/graph/{id} -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        g = r.json()
        check("subgraph expands beyond the centre node",
              g.get("node_count", 0) > 1 and g.get("edge_count", 0) > 0,
              f"node={hub} node_count={g.get('node_count')} "
              f"edge_count={g.get('edge_count')}")
        devs = {n.get("device") for n in g.get("nodes", [])}
        check("subgraph devices are real, not invented",
              devs <= {"desktop", "mobile", "(unknown)"}, f"got {sorted(devs)}")

    # ── Settings: the slider must actually change predictions ───────────────
    print("\n[settings -> prediction wiring]")
    r = s.get(f"{B}/api/settings", timeout=60)
    check("GET /api/settings -> 200", r.status_code == 200, r.text[:120])
    original = None
    was_overridden = False
    if r.status_code == 200:
        st = r.json()
        original = st.get("fraud_threshold")
        was_overridden = bool(st.get("is_overridden"))
        check("settings seed from the model's tuned threshold",
              st.get("model_tuned_threshold") is not None,
              f"model_tuned_threshold={st.get('model_tuned_threshold')}")

    # Force a threshold that must flip the verdict on a mid-probability node.
    if base_prob is not None:
        low = max(0.01, min(0.98, base_prob * 0.5))
        high = min(0.99, max(0.02, base_prob + (1 - base_prob) * 0.5))

        s.put(f"{B}/api/settings", json={"fraud_threshold": low}, timeout=60)
        r_low = s.post(f"{B}/api/predict/node", json={"node_id": 4340}, timeout=120)
        s.put(f"{B}/api/settings", json={"fraud_threshold": high}, timeout=60)
        r_high = s.post(f"{B}/api/predict/node", json={"node_id": 4340}, timeout=120)

        if r_low.status_code == 200 and r_high.status_code == 200:
            lo, hi = r_low.json(), r_high.json()
            # The API rounds thresholds to 4 decimal places in its response,
            # so compare at that resolution rather than exactly.
            check("threshold slider reaches the prediction path",
                  abs(lo["threshold"] - low) < 1e-3 and abs(hi["threshold"] - high) < 1e-3,
                  f"low={lo['threshold']} (asked {low:.4f}) "
                  f"high={hi['threshold']} (asked {high:.4f})")
            check("changing the threshold changes the verdict",
                  lo["is_fraud"] != hi["is_fraud"],
                  f"prob={base_prob} low_thr={low:.3f}->{lo['is_fraud']} "
                  f"high_thr={high:.3f}->{hi['is_fraud']}")

        # Restore. Writing the effective value back would persist it as an
        # explicit override (which is exactly how a stale 0.61 got stuck and
        # every other model was served at the ensemble's threshold), so reset
        # unless the original really was an override.
        if was_overridden and original is not None:
            s.put(f"{B}/api/settings", json={"fraud_threshold": original}, timeout=60)
        else:
            s.put(f"{B}/api/settings", json={"reset_threshold": True}, timeout=60)

    # ── Reports ─────────────────────────────────────────────────────────────
    print("\n[reports]")
    r = s.get(f"{B}/api/reports/csv", params={"limit": 50}, timeout=300)
    check("GET /api/reports/csv -> 200", r.status_code == 200, r.text[:120])
    if r.status_code == 200:
        lines = r.text.strip().split("\n")
        check("csv has a header and rows", len(lines) >= 2, f"{len(lines)} lines")

    # ── Summary ─────────────────────────────────────────────────────────────
    n_pass = sum(1 for x in results if x[0] == PASS)
    n_fail = sum(1 for x in results if x[0] == FAIL)
    n_warn = sum(1 for x in results if x[0] == WARN)

    print("\n" + "=" * 74)
    print(f"  {n_pass} passed, {n_fail} failed, {n_warn} warnings")
    print("=" * 74)
    if n_fail:
        print("\n  Failures:")
        for st, name, detail in results:
            if st == FAIL:
                print(f"    - {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
