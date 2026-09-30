"""
database.py - SQLAlchemy SQLite setup + seed
"""
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from server.config import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    # Widened from the SQLAlchemy defaults (5 + 10 overflow): the ML
    # endpoints can hold a connection across a slow call, and the
    # dashboard fans out several requests at once.
    pool_size=20,
    max_overflow=30,
    pool_timeout=60,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), unique=True, nullable=False)
    email         = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role          = Column(String(20), default="analyst")  # admin / analyst
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_login    = Column(DateTime, nullable=True)


class AppSettings(Base):
    __tablename__ = "app_settings"
    id                = Column(Integer, primary_key=True, index=True)
    # No default: a fresh install adopts the model's validation-tuned
    # threshold (see api/settings.get_settings). Defaulting to 0.5 baked in
    # the exact value that made the model flag nothing.
    fraud_threshold   = Column(Float, default=None, nullable=True)
    theme             = Column(String(20), default="dark")
    model_name        = Column(String(100), default="GraphSAGE-v1")
    notifications_enabled = Column(Boolean, default=True)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id            = Column(Integer, primary_key=True, index=True)
    node_id       = Column(Integer, nullable=True)
    input_data    = Column(Text, nullable=True)   # JSON string
    fraud_prob    = Column(Float)
    is_fraud      = Column(Boolean)
    confidence    = Column(Float)
    threshold_used = Column(Float)
    created_at    = Column(DateTime, default=datetime.utcnow)
    username      = Column(String(50), nullable=True)



class AlertReview(Base):
    """
    An analyst's decision on one alert.

    Keeps the model's score and the ground-truth label alongside the human
    call, so the precision analysts actually observe can be compared with the
    precision the test split predicted.
    """
    __tablename__ = "alert_reviews"

    id             = Column(Integer, primary_key=True, index=True)
    node_id        = Column(Integer, unique=True, index=True, nullable=False)
    transaction_id = Column(Integer, index=True)
    disposition    = Column(String(24))   # confirmed_fraud | dismissed | escalated
    note           = Column(Text)
    reviewer       = Column(String(64), index=True)
    fraud_prob     = Column(Float)
    true_label     = Column(Integer)
    threshold_used = Column(Float)
    model_used     = Column(String(24))
    reviewed_at    = Column(DateTime, default=datetime.utcnow)


# ── DB init + seed ─────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed demo users if not exist
        from server.auth.utils import get_password_hash
        if not db.query(User).filter_by(username="admin").first():
            db.add(User(
                username="admin",
                email="admin@fraudshield.ai",
                hashed_password=get_password_hash("fraud123"),
                role="admin",
            ))
        if not db.query(User).filter_by(username="analyst").first():
            db.add(User(
                username="analyst",
                email="analyst@fraudshield.ai",
                hashed_password=get_password_hash("demo456"),
                role="analyst",
            ))
        # Seed default settings
        if not db.query(AppSettings).first():
            db.add(AppSettings())
        db.commit()
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
