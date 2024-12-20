# api/routers/session_routes.py

from fastapi import APIRouter, HTTPException
from typing import Dict
from ..utils.session_manager import session_manager

router = APIRouter(prefix="/session", tags=["Session Management"])

@router.post("/new")
async def create_session() -> Dict[str, str]:
    """Create a new session."""
    session_id = session_manager.create_session()
    return {"session_id": session_id}

@router.get("/{session_id}/status")
async def get_session_status(session_id: str) -> Dict[str, bool]:
    """Check if a session is valid."""
    session = session_manager.get_session(session_id)
    return {"valid": session is not None}