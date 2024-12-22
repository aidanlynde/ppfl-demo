# api/utils/session_store.py
import json
import pickle
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class PersistentSessionStore:
    def __init__(self, storage_dir: str = "storage/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, dict] = self._load_sessions()

    def _load_sessions(self) -> Dict[str, dict]:
        """Load all sessions from disk."""
        sessions = {}
        for file in self.storage_dir.glob("*.session"):
            try:
                with open(file, 'rb') as f:
                    session_data = pickle.load(f)
                    sessions[file.stem] = session_data
            except Exception as e:
                print(f"Error loading session {file}: {e}")
        return sessions

    def save_session(self, session_id: str, session_data: dict):
        """Save session to disk."""
        try:
            file_path = self.storage_dir / f"{session_id}.session"
            with open(file_path, 'wb') as f:
                pickle.dump(session_data, f)
        except Exception as e:
            print(f"Error saving session {session_id}: {e}")

    def load_session(self, session_id: str) -> Optional[dict]:
        """Load session from disk."""
        try:
            file_path = self.storage_dir / f"{session_id}.session"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
        return None

    def delete_session(self, session_id: str):
        """Delete session from disk."""
        try:
            file_path = self.storage_dir / f"{session_id}.session"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")