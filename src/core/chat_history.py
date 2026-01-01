"""
Chat history management module.
Handles storage and retrieval of chat messages.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError

from ..utils.database import Base, get_db_session
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatMessage(Base):
    """Database model for chat messages."""
    __tablename__ = "chat_messages"
    
    message_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    message_metadata = Column(JSONB, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy reserved name
    sequence_number = Column(Integer, nullable=False, index=True)
    
    # Unique constraint to prevent duplicate sequence numbers per session
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence_number', name='uq_session_sequence'),
    )
    
    # Relationship to session
    session = relationship("Session", back_populates="messages")


class ChatHistoryManager:
    """Manages chat message history."""
    
    def __init__(self):
        """Initialize the chat history manager."""
        self.max_messages = config.session.max_messages_per_session
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatMessage]:
        """Add a message to chat history with retry logic for sequence number collisions."""
        import uuid
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with get_db_session() as db:
                    # Get next sequence number atomically within the same transaction
                    sequence_number = self._get_next_sequence_number_in_session(db, session_id)
                    
                    message_data = {
                        "message_id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "role": role,
                        "content": content,
                        "message_metadata": metadata or {},
                        "sequence_number": sequence_number
                    }
                    
                    message = ChatMessage(**message_data)
                    db.add(message)
                    db.flush()  # Flush to get ID without committing
                    db.refresh(message)
                    
                    # This ensures we can see the newly added message when counting
                    self._cleanup_old_messages_in_session(db, session_id)
                    
                    # Expunge to detach from session before returning
                    db.expunge(message)
                    
                    logger.debug(f"Added {role} message to session {session_id} with sequence {sequence_number}")
                    return message
            except IntegrityError as e:
                # Handle unique constraint violation (sequence number collision)
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Sequence number collision for session {session_id}, attempt {attempt + 1}/{max_retries}. Retrying..."
                    )
                    continue
                else:
                    logger.error(f"Failed to add message after {max_retries} retries due to sequence number collision: {e}")
                    return None
            except Exception as e:
                logger.error(f"Failed to add message to session {session_id}: {e}")
                return None
        
        return None
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ChatMessage]:
        """Get messages for a session."""
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(ChatMessage.sequence_number.asc())
                
                if offset > 0:
                    query = query.offset(offset)
                
                if limit:
                    query = query.limit(limit)
                
                messages = query.all()
                # Expunge all messages to detach from session before returning
                for msg in messages:
                    db.expunge(msg)
                return messages
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []
    
    def get_recent_messages(
        self,
        session_id: str,
        n: int = 10
    ) -> List[ChatMessage]:
        """Get the most recent N messages for a session."""
        try:
            with get_db_session() as db:
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(
                    ChatMessage.sequence_number.desc()
                ).limit(n).all()
                
                # Expunge all messages to detach from session before returning
                for msg in messages:
                    db.expunge(msg)
                # Return in chronological order
                return list(reversed(messages))
        except Exception as e:
            logger.error(f"Failed to get recent messages for session {session_id}: {e}")
            return []
    
    def clear_history(self, session_id: str) -> bool:
        """Clear all messages for a session."""
        try:
            with get_db_session() as db:
                deleted = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).delete()
                logger.info(f"Cleared {deleted} messages for session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to clear history for session {session_id}: {e}")
            return False
    
    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        try:
            with get_db_session() as db:
                count = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).count()
                return count
        except Exception as e:
            logger.error(f"Failed to get message count for session {session_id}: {e}")
            return 0
    
    def _get_next_sequence_number_in_session(self, db, session_id: str) -> int:
        """Get the next sequence number for a session (using existing session)."""
        try:
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.sequence_number.desc()).first()
            
            if last_message:
                return last_message.sequence_number + 1
            return 1
        except Exception as e:
            logger.error(f"Failed to get next sequence number: {e}")
            return 1
    
    def _get_next_sequence_number(self, session_id: str) -> int:
        """Get the next sequence number for a session (opens new session)."""
        try:
            with get_db_session() as db:
                return self._get_next_sequence_number_in_session(db, session_id)
        except Exception as e:
            logger.error(f"Failed to get next sequence number: {e}")
            return 1
    
    def _cleanup_old_messages_in_session(self, db, session_id: str):
        """Remove old messages if exceeding the limit (using existing session)."""
        try:
            count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).count()
            
            if count > self.max_messages:
                excess = count - self.max_messages
                # Delete oldest messages
                db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(
                    ChatMessage.sequence_number.asc()
                ).limit(excess).delete(synchronize_session=False)
                logger.info(f"Cleaned up {excess} old messages for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}")
    
    def _cleanup_old_messages(self, session_id: str):
        """Remove old messages if exceeding the limit (opens new session)."""
        try:
            count = self.get_message_count(session_id)
            if count > self.max_messages:
                excess = count - self.max_messages
                with get_db_session() as db:
                    # Delete oldest messages
                    db.query(ChatMessage).filter(
                        ChatMessage.session_id == session_id
                    ).order_by(
                        ChatMessage.sequence_number.asc()
                    ).limit(excess).delete(synchronize_session=False)
                    logger.info(f"Cleaned up {excess} old messages for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}")

