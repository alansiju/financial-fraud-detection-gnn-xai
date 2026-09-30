"""
api/settings.py - App settings CRUD.

The stored `fraud_threshold` is reported alongside the model's own tuned
threshold so the UI can show when an operator override is diverging from the
value train.py selected on the validation split.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth.dependencies import get_current_user
from server.database import get_db, AppSettings
from server.ml import loader

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsUpdate(BaseModel):
    fraud_threshold: float = None
    # Clears a manual threshold so it follows the active model again.
    reset_threshold: bool = False
    theme: str = None
    model_name: str = None
    notifications_enabled: bool = None


@router.get("")
def get_settings(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    s = db.query(AppSettings).first()
    if not s:
        s = AppSettings()
        db.add(s)
        db.commit()
        db.refresh(s)
    model_thr = loader.get_tuned_threshold()
    # Deliberately NOT persisted. Writing the tuned value here turned a default
    # into a permanent override: it stuck after the active model changed, so
    # the ensemble (tuned to 0.61) was served at the GNN's 0.59. An unset
    # threshold now simply follows whichever model is active.
    effective = s.fraud_threshold if s.fraud_threshold is not None else model_thr

    return {
        "fraud_threshold": effective,
        "model_tuned_threshold": round(model_thr, 4),
        "active_model": loader.get_active_model(),
        "is_overridden": s.fraud_threshold is not None
                         and abs(s.fraud_threshold - model_thr) > 1e-6,
        "theme": s.theme,
        "model_name": s.model_name,
        "notifications_enabled": s.notifications_enabled,
        "updated_at": s.updated_at,
    }


@router.get("/models")
def list_models(current_user=Depends(get_current_user)):
    """
    Which model variants can serve predictions, and which one is active.

    Each carries its own validation-tuned threshold: the three are calibrated
    differently, so one shared cut-off would misreport at least two of them.
    """
    ens = loader.get_ensemble_config() or {}
    return {
        "active": loader.get_active_model(),
        "models": loader.get_available_models(),
        "blend": {
            "weight_gnn": ens.get("weight_gnn"),
            "weight_lgbm": ens.get("weight_lgbm"),
            "val_pr_auc": ens.get("val_pr_auc"),
            "verdict": ens.get("verdict"),
        } if ens else None,
        "test_metrics": ens.get("test_metrics"),
    }


class ModelSelection(BaseModel):
    model_id: str


@router.put("/models")
def select_model(
    body: ModelSelection,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Switch the serving model. Cached scores are kept per variant."""
    try:
        active = loader.set_active_model(body.model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # A manual threshold was chosen against the previous model's score
    # distribution; the models are calibrated differently, so carrying it over
    # would quietly serve the new model at an arbitrary cut-off. Clear it.
    loader.set_threshold_override(None)
    s = db.query(AppSettings).first()
    if s is not None and s.fraud_threshold is not None:
        s.fraud_threshold = None
        db.commit()
    return {
        "active": active,
        "threshold": round(loader.get_threshold(), 4),
        "tuned_threshold": round(loader.get_tuned_threshold(), 4),
    }


@router.put("")
def update_settings(
    body: SettingsUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    s = db.query(AppSettings).first()
    if not s:
        s = AppSettings()
        db.add(s)
    if body.reset_threshold:
        s.fraud_threshold = None
        loader.set_threshold_override(None)
    elif body.fraud_threshold is not None:
        s.fraud_threshold = max(0.01, min(0.99, body.fraud_threshold))
        # Make the slider actually do something.
        loader.set_threshold_override(s.fraud_threshold)
    if body.theme is not None and body.theme in ("dark", "light"):
        s.theme = body.theme
    if body.model_name is not None:
        s.model_name = body.model_name
    if body.notifications_enabled is not None:
        s.notifications_enabled = body.notifications_enabled
    s.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(s)
    return {
        "message": "Settings updated",
        "fraud_threshold": s.fraud_threshold,
        "model_tuned_threshold": round(loader.get_tuned_threshold(), 4),
        "active_threshold": round(loader.get_threshold(), 4),
        "theme": s.theme,
    }
