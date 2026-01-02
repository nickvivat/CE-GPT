"""
Session management module for user sessions.
Handles session creation, retrieval, update, and deletion.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import json

from ..utils.database import Base, get_db_session
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Session(Base):
    """Database model for user sessions."""
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    session_metadata = Column(JSONB, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy reserved name
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationship to chat messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class SessionManager:
    """Manages user sessions."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.default_ttl_hours = config.session.default_ttl_hours
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        ttl = ttl_hours if ttl_hours else self.default_ttl_hours
        expires_at = datetime.utcnow() + timedelta(hours=ttl)
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "expires_at": expires_at,
            "session_metadata": metadata or {},
            "is_active": True
        }
        
        try:
            with get_db_session() as db:
                db_session = Session(**session_data)
                db.add(db_session)
                db.flush()  # Flush to get ID without committing
                db.refresh(db_session)
                # Expunge to detach from session before returning
                db.expunge(db_session)
                logger.info(f"Created session: {session_id}")
                return db_session
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        try:
            with get_db_session() as db:
                session = db.query(Session).filter(
                    Session.session_id == session_id,
                    Session.is_active == True
                ).first()
                
                if session:
                    # Check if session is expired
                    if datetime.utcnow() > session.expires_at:
                        logger.info(f"Session {session_id} has expired")
                        self.delete_session(session_id)
                        return None
                    
                    # Update last accessed time
                    session.updated_at = datetime.utcnow()
                    # Flush to persist the update before expunging
                    db.flush()
                    # Expunge to detach from session before returning
                    db.expunge(session)
                    return session
                return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        extend_ttl: bool = False
    ) -> bool:
        """Update session metadata or extend TTL."""
        try:
            with get_db_session() as db:
                session = db.query(Session).filter(
                    Session.session_id == session_id,
                    Session.is_active == True
                ).first()
                
                if not session:
                    return False
                
                if metadata is not None:
                    # Merge metadata
                    current_metadata = session.session_metadata or {}
                    current_metadata.update(metadata)
                    session.session_metadata = current_metadata
                
                if extend_ttl:
                    session.expires_at = datetime.utcnow() + timedelta(hours=self.default_ttl_hours)
                
                session.updated_at = datetime.utcnow()
                logger.info(f"Updated session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session (soft delete by setting is_active=False)."""
        try:
            with get_db_session() as db:
                session = db.query(Session).filter(
                    Session.session_id == session_id
                ).first()
                
                if not session:
                    return False
                
                session.is_active = False
                logger.info(f"Deleted session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Delete all expired sessions."""
        try:
            with get_db_session() as db:
                now = datetime.utcnow()
                expired_sessions = db.query(Session).filter(
                    Session.expires_at < now,
                    Session.is_active == True
                ).all()
                
                count = len(expired_sessions)
                for session in expired_sessions:
                    session.is_active = False
                
                if count > 0:
                    logger.info(f"Cleaned up {count} expired sessions")
                return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_active_sessions(self, limit: int = 100) -> List[Session]:
        """Get active sessions."""
        try:
            with get_db_session() as db:
                sessions = db.query(Session).filter(
                    Session.is_active == True
                ).order_by(Session.updated_at.desc()).limit(limit).all()
                # Expunge all sessions to detach from session before returning
                for session in sessions:
                    db.expunge(session)
                return sessions
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    def get_most_recent_active_session(
        self,
        user_id: Optional[str] = None,
        max_age_minutes: int = 30
    ) -> Optional[Session]:
        """
        Get the most recent active session that hasn't expired.
        
        Args:
            user_id: Optional user ID to filter sessions. If None, returns most recent session regardless of user.
            max_age_minutes: Maximum age in minutes for a session to be considered for reuse (default: 30).
                            Sessions older than this will not be reused.
        
        Returns:
            The most recent active session, or None if no valid session is found.
        """
        try:
            with get_db_session() as db:
                now = datetime.utcnow()
                max_age_threshold = now - timedelta(minutes=max_age_minutes)
                
                query = db.query(Session).filter(
                    Session.is_active == True,
                    Session.expires_at > now,  # Not expired
                    Session.updated_at >= max_age_threshold  # Recently used
                )
                
                # Filter by user_id if provided
                if user_id:
                    query = query.filter(Session.user_id == user_id)
                
                session = query.order_by(Session.updated_at.desc()).first()
                
                if session:
                    # Update last accessed time
                    session.updated_at = datetime.utcnow()
                    db.flush()
                    # Expunge to detach from session before returning
                    db.expunge(session)
                    logger.info(f"Reusing existing session: {session.session_id}")
                    return session
                
                return None
        except Exception as e:
            logger.error(f"Failed to get most recent active session: {e}")
            return None

