"""auth/router.py - Login, logout, me endpoints"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.database import get_db, User
from server.auth.utils import verify_password, create_access_token
from server.auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    email: str


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=req.username, is_active=True).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(
        access_token=token,
        username=user.username,
        role=user.role,
        email=user.email,
    )


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "last_login": current_user.last_login,
        "created_at": current_user.created_at,
    }


@router.post("/logout")
def logout():
    # Stateless JWT — client just discards token
    return {"message": "Logged out successfully"}
