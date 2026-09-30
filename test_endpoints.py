"""Quick smoke test for core endpoints (avoids the heavy transactions cache)."""
import urllib.request
import json
import sys

BASE = "http://localhost:8000"

def api(method, path, data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception as e:
        return 0, str(e)

passed = 0
failed = 0

def check(name, status, body, expected=200, keys=None):
    global passed, failed
    ok = (status == expected)
    if ok and keys and isinstance(body, dict):
        for k in keys:
            if k not in body:
                ok = False
                break
    if ok:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        detail = str(body)[:120] if not isinstance(body, dict) else str(list(body.keys()))[:120]
        print(f"  [FAIL] {name} -> status={status} {detail}")
    return ok

print("=" * 60)
print("FraudShield API Smoke Tests")
print("=" * 60)

# 1. Auth
print("\n--- Auth ---")
s, b = api("POST", "/auth/login", {"username": "admin", "password": "fraud123"})
check("POST /auth/login (admin)", s, b, 200, ["access_token", "username", "role", "email"])
token = b.get("access_token", "") if isinstance(b, dict) else ""

s, b = api("POST", "/auth/login", {"username": "analyst", "password": "demo456"})
check("POST /auth/login (analyst)", s, b, 200, ["access_token"])

s, b = api("POST", "/auth/login", {"username": "x", "password": "y"})
check("POST /auth/login (bad creds -> 401)", s, b, 401)

# 2. Dashboard
print("\n--- Dashboard ---")
s, b = api("GET", "/api/dashboard", token=token)
if check("GET /api/dashboard", s, b, 200, ["total_transactions", "fraud_detected", "roc_auc"]):
    print(f"       total={b['total_transactions']}, fraud={b['fraud_detected']}, auc={b.get('roc_auc')}")

# 3. Analytics
print("\n--- Analytics ---")
s, b = api("GET", "/api/analytics/summary", token=token)
if check("GET /api/analytics/summary", s, b, 200, ["fraud_by_month", "class_dist"]):
    print(f"       keys={list(b.keys())}")

# 4. Predict node
print("\n--- Predict ---")
s, b = api("POST", "/api/predict/node", {"node_id": 42038, "threshold": 0.5}, token=token)
if check("POST /api/predict/node", s, b, 200, ["node_id", "fraud_probability", "is_fraud"]):
    print(f"       node=42038, prob={b['fraud_probability']}, fraud={b['is_fraud']}, risk={b['risk_level']}")

# 5. Predict history
print("\n--- Predict History ---")
s, b = api("GET", "/api/predict/history?limit=5", token=token)
if isinstance(b, list):
    check("GET /api/predict/history", s, b, 200)
    print(f"       {len(b)} entries")
else:
    check("GET /api/predict/history", s, b, 200)

# 6. Graph
print("\n--- Graph ---")
s, b = api("GET", "/api/graph/42038?depth=1", token=token)
if check("GET /api/graph/42038", s, b, 200, ["center_node", "nodes", "edges"]):
    print(f"       {b['node_count']} nodes, {b['edge_count']} edges")

# 7. Metrics
print("\n--- Metrics ---")
s, b = api("GET", "/api/metrics", token=token)
if check("GET /api/metrics", s, b, 200, ["accuracy", "roc_auc", "confusion_matrix", "roc_curve"]):
    print(f"       acc={b['accuracy']}, auc={b['roc_auc']}, f1={b.get('f1')}")
    print(f"       ROC: {len(b['roc_curve'])} pts, PR: {len(b['pr_curve'])} pts, train: {len(b['training_curves'])} epochs")

# 8. Settings
print("\n--- Settings ---")
s, b = api("GET", "/api/settings", token=token)
check("GET /api/settings", s, b, 200, ["fraud_threshold", "theme"])

s, b = api("PUT", "/api/settings", {"fraud_threshold": 0.6}, token=token)
check("PUT /api/settings", s, b, 200)

# 9. Transactions (small page to avoid cache build hang)
print("\n--- Transactions ---")
s, b = api("GET", "/api/transactions?page=1&limit=3", token=token)
if check("GET /api/transactions", s, b, 200, ["total", "data", "total_pages"]):
    print(f"       total={b['total']}, pages={b['total_pages']}, returned={len(b['data'])}")
    if b["data"]:
        nid = b["data"][0]["id"]
        s2, b2 = api("GET", f"/api/transactions/{nid}", token=token)
        check(f"GET /api/transactions/{nid}", s2, b2, 200, ["node_id", "fraud_probability"])

# 10. Reports (just check CSV, fastest)
print("\n--- Reports ---")
try:
    req = urllib.request.Request(
        f"{BASE}/api/reports/csv?limit=5",
        headers={"Authorization": f"Bearer {token}"}
    )
    r = urllib.request.urlopen(req, timeout=30)
    size = len(r.read())
    if r.status == 200 and size > 50:
        passed += 1
        print(f"  [PASS] GET /api/reports/csv ({size} bytes)")
    else:
        failed += 1
        print(f"  [FAIL] GET /api/reports/csv -> {r.status} ({size} bytes)")
except Exception as e:
    failed += 1
    print(f"  [FAIL] GET /api/reports/csv -> {e}")

# Summary
print("\n" + "=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("ALL ENDPOINTS WORKING!")
print("=" * 60)
