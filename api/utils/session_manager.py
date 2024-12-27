# api/utils/session_manager.py

import uuid
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
from models.federated.private_fl_manager import PrivateFederatedLearningManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Session:
    """Represents a training session with associated FL manager."""
    
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created: datetime = datetime.now()
        self.last_active: datetime = datetime.now()
        self.fl_manager: Optional[PrivateFederatedLearningManager] = None
    
    def to_dict(self) -> dict:
        """Convert session to dictionary for persistence."""
        try:
            return {
                'id': self.id,
                'created': self.created,
                'last_active': self.last_active,
                'fl_manager': pickle.dumps(self.fl_manager)  # Serialize FL manager
            }

    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        """Create session from dictionary."""
        session = cls()
        session.id = data['id']
        session.created = data['created']
        session.last_active = data['last_active']
        session.fl_manager = pickle.loads(data['fl_manager']) if data['fl_manager'] else None
        return session

class SessionManager:
    """Manages training sessions with persistence and cleanup."""
    
    def __init__(self, session_timeout: int = 30):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout)
        self.storage_dir = Path("storage/sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_sessions()
        logger.info("SessionManager initialized with timeout: %d minutes", session_timeout)
    
    def _load_persisted_sessions(self) -> None:
        """Load valid sessions from disk with error handling."""
        loaded = 0
        errors = 0
        for file in self.storage_dir.glob("*.session"):
            try:
                with open(file, 'rb') as f:
                    session_data = pickle.load(f)
                    if not self._is_session_expired(session_data['last_active']):
                        session = Session.from_dict(session_data)
                        self.sessions[session.id] = session
                        loaded += 1
                    else:
                        # Clean up expired session file
                        file.unlink()
            except Exception as e:
                logger.error("Error loading session %s: %s", file, str(e))
                errors += 1
                try:
                    file.unlink()  # Clean up corrupted session file
                except:
                    pass
        
        logger.info("Loaded %d sessions, encountered %d errors", loaded, errors)
    
    def _is_session_expired(self, last_active: datetime) -> bool:
        """Check if a session is expired."""
        return datetime.now() - last_active > self.session_timeout
    
    def create_session(self) -> str:
        """Create a new session with logging."""
        session = Session()
        self.sessions[session.id] = session
        self._persist_session(session)
        logger.info("Created new session: %s", session.id)
        return session.id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get and validate session, updating last_active time."""
        session = self.sessions.get(session_id)
        
        if not session:
            # Try to load from persistent storage
            session_data = self._load_session(session_id)
            if session_data and not self._is_session_expired(session_data['last_active']):
                session = Session.from_dict(session_data)
                self.sessions[session_id] = session
                logger.debug("Loaded session from storage: %s", session_id)
            else:
                logger.warning("Invalid or expired session requested: %s", session_id)
                return None
        
        if session:
            session.last_active = datetime.now()
            self._persist_session(session)
        
        return session
    
    def _persist_session(self, session: Session) -> None:
        """Save session to disk with proper error handling."""
        try:
            file_path = self.storage_dir / f"{session.id}.session"
            with open(file_path, 'wb') as f:
                pickle.dump(session.to_dict(), f)
            logger.debug("Persisted session: %s", session.id)
        except Exception as e:
            logger.error("Error saving session %s: %s", session.id, str(e))
            # Attempt cleanup of potentially corrupted file
            try:
                file_path = self.storage_dir / f"{session.id}.session"
                if file_path.exists():
                    file_path.unlink()
            except:
                pass
    
    def _load_session(self, session_id: str) -> Optional[dict]:
        """Load session from disk with error handling."""
        try:
            file_path = self.storage_dir / f"{session_id}.session"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error("Error loading session %s: %s", session_id, str(e))
            # Clean up corrupted session file
            try:
                file_path.unlink()
            except:
                pass
        return None
    
    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions from memory and disk."""
        current_time = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_active > self.session_timeout
        ]
        
        cleaned = 0
        errors = 0
        for sid in expired_sessions:
            try:
                # Remove from memory
                del self.sessions[sid]
                
                # Remove from disk
                file_path = self.storage_dir / f"{sid}.session"
                if file_path.exists():
                    file_path.unlink()
                cleaned += 1
            except Exception as e:
                logger.error("Error cleaning up session %s: %s", sid, str(e))
                errors += 1
        
        if expired_sessions:
            logger.info("Cleaned up %d expired sessions (%d errors)", cleaned, errors)

# Global session manager instance with 30-minute timeout
session_manager = SessionManager(session_timeout=30)