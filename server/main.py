"""
server/main.py - FastAPI application factory
"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import APP_TITLE, APP_DESC, APP_VERSION, STATIC_DIR
from server.database import init_db
from server.ml import loader as ml_loader

# ── Routers ────────────────────────────────────────────────────────────────
from server.auth.router import router as auth_router
from server.api.dashboard    import router as dashboard_router
from server.api.transactions  import router as transactions_router
from server.api.predict       import router as predict_router
from server.api.explain       import router as explain_router
from server.api.graph         import router as graph_router
from server.api.analytics     import router as analytics_router
from server.api.metrics       import router as metrics_router
from server.api.reports       import router as reports_router
from server.api.settings      import router as settings_router
from server.api.upload_predict import router as upload_predict_router
from server.api.insights      import router as insights_router
from server.api.alerts        import router as alerts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load ML. Shutdown: cleanup."""
    init_db()
    ml_loader.load_all()
    _apply_saved_threshold()
    _warm_prediction_cache()
    yield
    print("Server shutting down.")


def _warm_prediction_cache() -> None:
    """
    Score every graph node once, before the server starts accepting requests.

    Doing this lazily on the first request meant that request sat for a minute
    or two holding a pooled database connection, and a few concurrent page
    loads then exhausted the pool ("QueuePool limit of size 5 overflow 10
    reached") so every endpoint failed. Paying the cost at startup keeps every
    request fast and makes "ready" actually mean ready.
    """
    import time
    try:
        if ml_loader.get_model() is None or ml_loader.get_data() is None:
            print("[!] Skipping prediction warm-up - model or graph missing.")
            return
        t0 = time.time()
        probs = ml_loader.get_all_node_probs()
        if probs is not None:
            print(f"[OK] Prediction cache warm: {len(probs):,} nodes "
                  f"in {time.time() - t0:.1f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] Prediction warm-up failed: {type(e).__name__}: {e}")


def _apply_saved_threshold() -> None:
    """
    Re-apply the operator's threshold override from the database so it survives
    a restart. Without this the Settings slider would silently revert to the
    model's tuned value on every boot.
    """
    try:
        from server.database import SessionLocal, AppSettings
        db = SessionLocal()
        try:
            row = db.query(AppSettings).first()
            if row is not None and row.fraud_threshold is not None:
                active = ml_loader.set_threshold_override(row.fraud_threshold)
                tuned = ml_loader.get_tuned_threshold()
                if abs(active - tuned) > 1e-6:
                    print(f"[OK] Operator threshold override applied: "
                          f"{active:.2f} (model tuned: {tuned:.2f})")
        finally:
            db.close()
    except Exception as e:
        print(f"[!] Could not apply saved threshold override: "
              f"{type(e).__name__}: {e}")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Routes ─────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(transactions_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(graph_router)
app.include_router(analytics_router)
app.include_router(metrics_router)
app.include_router(reports_router)
app.include_router(settings_router)
app.include_router(upload_predict_router)
app.include_router(insights_router)
app.include_router(alerts_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_loader.is_loaded(),
        "version": APP_VERSION,
    }
