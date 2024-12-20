# api/utils/session_manager.py

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from models.federated.private_fl_manager import PrivateFederatedLearningManager

class Session:
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created: datetime = datetime.now()
        self.last_active: datetime = datetime.now()
        self.fl_manager: Optional[PrivateFederatedLearningManager] = None

class SessionManager:
    def __init__(self, session_timeout: int = 30):  # timeout in minutes
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout)
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session = Session()
        self.sessions[session.id] = session
        return session.id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID and update its last_active time."""
        session = self.sessions.get(session_id)
        if session:
            session.last_active = datetime.now()
        return session
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_active > self.session_timeout
        ]
        for sid in expired_sessions:
            del self.sessions[sid]

# Global session manager instance
session_manager = SessionManager()