# api/utils/session_manager.py

import uuid
import pickle
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
from models.federated.private_fl_manager import PrivateFederatedLearningManager

class Session:
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created: datetime = datetime.now()
        self.last_active: datetime = datetime.now()
        self.fl_manager: Optional[PrivateFederatedLearningManager] = None
    
    def to_dict(self) -> dict:
        """Convert session to dictionary for persistence."""
        return {
            'id': self.id,
            'created': self.created,
            'last_active': self.last_active,
            'fl_manager': self.fl_manager
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        """Create session from dictionary."""
        session = cls()
        session.id = data['id']
        session.created = data['created']
        session.last_active = data['last_active']
        session.fl_manager = data['fl_manager']
        return session

class SessionManager:
    def __init__(self, session_timeout: int = 30):  # timeout in minutes
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout)
        self.storage_dir = Path("storage/sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_sessions()
    
    def _load_persisted_sessions(self):
        """Load all valid sessions from disk."""
        for file in self.storage_dir.glob("*.session"):
            try:
                with open(file, 'rb') as f:
                    session_data = pickle.load(f)
                    if not self._is_session_expired(session_data['last_active']):
                        session = Session.from_dict(session_data)
                        self.sessions[session.id] = session
            except Exception as e:
                print(f"Error loading session {file}: {e}")
    
    def _is_session_expired(self, last_active: datetime) -> bool:
        """Check if a session is expired."""
        return datetime.now() - last_active > self.session_timeout
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session = Session()
        self.sessions[session.id] = session
        self._persist_session(session)
        return session.id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID and update its last_active time."""
        session = self.sessions.get(session_id)
        
        if not session:
            # Try to load from persistent storage
            session_data = self._load_session(session_id)
            if session_data and not self._is_session_expired(session_data['last_active']):
                session = Session.from_dict(session_data)
                self.sessions[session_id] = session
        
        if session:
            session.last_active = datetime.now()
            self._persist_session(session)
        
        return session
    
    def _persist_session(self, session: Session):
        """Save session to disk."""
        try:
            file_path = self.storage_dir / f"{session.id}.session"
            with open(file_path, 'wb') as f:
                pickle.dump(session.to_dict(), f)
        except Exception as e:
            print(f"Error saving session {session.id}: {e}")
    
    def _load_session(self, session_id: str) -> Optional[dict]:
        """Load session from disk."""
        try:
            file_path = self.storage_dir / f"{session_id}.session"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
        return None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory and disk."""
        current_time = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_active > self.session_timeout
        ]
        
        for sid in expired_sessions:
            # Remove from memory
            del self.sessions[sid]
            
            # Remove from disk
            try:
                file_path = self.storage_dir / f"{sid}.session"
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Error deleting session file {sid}: {e}")

# Global session manager instance
session_manager = SessionManager()