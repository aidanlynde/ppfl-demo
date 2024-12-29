import os
import uuid
import pickle
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

from sqlalchemy import create_engine, text

from models.federated.private_fl_manager import PrivateFederatedLearningManager
from api.utils.logger_config import logger

class Session:
    """
    Represents a training session with associated FL manager.
    We store this in the database by pickling the entire object.
    """
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created: datetime = datetime.now()
        self.last_active: datetime = datetime.now()
        self.fl_manager: Optional[PrivateFederatedLearningManager] = None

    def to_dict(self) -> dict:
        """Convert this Session into a dict of primitive data + a pickle of fl_manager."""
        return {
            'id': self.id,
            'created': self.created,
            'last_active': self.last_active,
            # We store the fl_manager as a separate pickle so we can nest it inside.
            'fl_manager': pickle.dumps(self.fl_manager) if self.fl_manager else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        """Create a Session object from a dict that was originally produced by to_dict()."""
        s = cls()
        s.id = data['id']
        s.created = data['created']
        s.last_active = data['last_active']
        if data['fl_manager'] is not None:
            s.fl_manager = pickle.loads(data['fl_manager'])
        return s

class SessionManager:
    """
    Postgres-backed Session Manager.
    Stores each session (including the pickled fl_manager) in a 'sessions' table.
    """

    def __init__(self, session_timeout: int = 30):
        # We do a time-based expiration, in minutes
        self.session_timeout = timedelta(minutes=session_timeout)

        # Read DATABASE_URL from environment (Railway env var)
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set in the environment variables")

        # Create a SQLAlchemy engine
        self.engine = create_engine(database_url, echo=False)

        # On initialization, ensure the 'sessions' table exists
        self._create_table_if_not_exists()

        logger.info(
            "Postgres-backed SessionManager initialized with a %d-minute timeout",
            session_timeout
        )

    def _create_table_if_not_exists(self):
        """
        Create a table named 'sessions' if it doesn't already exist.
        Columns:
         - session_id (text primary key)
         - data (bytea) => we'll store a pickled dict from Session.to_dict()
         - created (timestamp)
         - last_active (timestamp)
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            data BYTEA NOT NULL,
            created TIMESTAMP NOT NULL,
            last_active TIMESTAMP NOT NULL
        );
        """
        with self.engine.begin() as conn:
            conn.execute(text(create_sql))

    def create_session(self) -> str:
        """
        Create a new Session object, insert it into the DB, and return the session_id.
        """
        new_session = Session()  # from our Session class
        self._persist_session(new_session)
        logger.info("Created new session: %s", new_session.id)
        return new_session.id

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Load a session from DB by session_id, check if it's expired, update last_active, re-persist.
        Returns None if not found or expired.
        """
        session = self._load_session_from_db(session_id)
        if not session:
            logger.warning("Session %s not found in DB", session_id)
            return None

        # Check if expired
        if self._is_session_expired(session.last_active):
            logger.info("Session %s is expired; deleting...", session_id)
            self._delete_session(session_id)
            return None

        # Update last_active
        session.last_active = datetime.now()
        self._persist_session(session)
        return session

    def _persist_session(self, session: Session) -> None:
        """
        Insert or update the given session in the DB (upsert by session_id).
        """
        # Convert session to dict, then pickle that dict
        session_dict = session.to_dict()
        pickled_data = pickle.dumps(session_dict)

        with self.engine.begin() as conn:
            # We do an UPSERT by session_id
            conn.execute(
                text("""
                    INSERT INTO sessions (session_id, data, created, last_active)
                    VALUES (:sid, :data, :created, :last_active)
                    ON CONFLICT (session_id) DO UPDATE
                      SET data = EXCLUDED.data,
                          last_active = EXCLUDED.last_active
                """),
                {
                    "sid": session.id,
                    "data": pickled_data,
                    "created": session.created,
                    "last_active": session.last_active
                }
            )

    def _load_session_from_db(self, session_id: str) -> Optional[Session]:
        """
        Fetch the 'data' (pickled dict) from 'sessions' table by session_id, unpickle, and return a Session.
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT data, created, last_active FROM sessions WHERE session_id = :sid"),
                {"sid": session_id}
            ).fetchone()

        if not row:
            return None

        pickled_data = row[0]
        try:
            session_dict = pickle.loads(pickled_data)
            session_obj = Session.from_dict(session_dict)
            return session_obj
        except Exception as e:
            logger.error("Error unpickling session %s: %s", session_id, e)
            # If corrupted, delete it
            self._delete_session(session_id)
            return None

    def _delete_session(self, session_id: str):
        """Remove a session row from DB."""
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM sessions WHERE session_id = :sid"),
                {"sid": session_id}
            )

    def _is_session_expired(self, last_active: datetime) -> bool:
        """Check if the session has been inactive for longer than session_timeout."""
        return (datetime.now() - last_active) > self.session_timeout

    def cleanup_expired_sessions(self):
        """
        Optional method: Remove expired sessions from DB. You can call this periodically if you like.
        """
        # Example: We'll load all sessions, check which are expired, and delete them.
        # For large scale, you’d do a direct SQL query with a timestamp condition.
        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT session_id, data, last_active FROM sessions")).fetchall()

        expired_count = 0
        for row in rows:
            sid = row[0]
            pickled_data = row[1]
            last_active = row[2]

            if self._is_session_expired(last_active):
                self._delete_session(sid)
                expired_count += 1

        if expired_count > 0:
            logger.info("Cleaned up %d expired sessions", expired_count)


# Finally, instantiate a global SessionManager (like you did before).
# We can keep the same name `session_manager` so imports don't break.
from sqlalchemy import create_engine

session_manager = SessionManager(session_timeout=30)
