"""
Chat history management module.
Handles storage and retrieval of chat messages with optimized performance and concurrency handling.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, UniqueConstraint, Index, func, select, text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from cachetools import TTLCache
import threading
import hashlib

from ..utils.database import Base, get_db_session
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatMessage(Base):
    """Database model for chat messages with soft delete support."""
    __tablename__ = "chat_messages"
    
    message_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    message_metadata = Column(JSONB, default=dict)
    sequence_number = Column(Integer, nullable=False, index=True)
    deleted_at = Column(DateTime, nullable=True, index=True)
    
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence_number', name='uq_session_sequence'),
        Index('idx_session_sequence', 'session_id', 'sequence_number'),
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_session_deleted', 'session_id', 'deleted_at'),
    )
    
    session = relationship("Session", back_populates="messages")
    
    @property
    def is_deleted(self) -> bool:
        """Check if message is soft-deleted."""
        return self.deleted_at is not None


class ChatHistoryManager:
    """Manages chat message history with optimized performance and concurrency handling."""
    
    def __init__(self):
        """Initialize the chat history manager."""
        self.max_messages = config.session.max_messages_per_session
        self._recent_messages_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        self._cache_lock = threading.Lock()
        self._cleanup_threshold = int(self.max_messages * 0.9)
        self._cleanup_target = int(self.max_messages * 0.8)
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatMessage]:
        """Add a message to chat history with thread-safe sequence number generation."""
        import uuid
        import time
        
        max_retries = 5
        base_delay = 0.01
        
        for attempt in range(max_retries):
            try:
                with get_db_session() as db:
                    lock_id = self._get_session_lock_id(session_id)
                    db.execute(text("SELECT pg_advisory_xact_lock(:lock_id)").bindparams(lock_id=lock_id))
                    
                    sequence_number = self._get_next_sequence_number_in_session(db, session_id)
                    
                    message = ChatMessage(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role=role,
                        content=content,
                        message_metadata=metadata or {},
                        sequence_number=sequence_number
                    )
                    db.add(message)
                    db.flush()
                    db.refresh(message)
                    
                    self._cleanup_old_messages_in_session(db, session_id)
                    db.expunge(message)
                    
                    logger.debug(f"Added {role} message to session {session_id} with sequence {sequence_number}")
                
                self._invalidate_cache(session_id)
                return message
            except IntegrityError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (time.time() % 0.01)
                    logger.warning(
                        f"Sequence number collision for session {session_id}, attempt {attempt + 1}/{max_retries}. "
                        f"Retrying after {delay:.3f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Failed to add message after {max_retries} retries due to sequence number collision "
                        f"for session {session_id}: {e}"
                    )
                    return None
            except Exception as e:
                logger.error(f"Failed to add message to session {session_id}: {e}", exc_info=True)
                return None
        
        return None
    
    def add_message_pair(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[ChatMessage], Optional[ChatMessage]]:
        """Add user and assistant messages atomically in a single transaction."""
        import uuid
        import time
        
        max_retries = 5
        base_delay = 0.01
        
        for attempt in range(max_retries):
            try:
                with get_db_session() as db:
                    lock_id = self._get_session_lock_id(session_id)
                    db.execute(text("SELECT pg_advisory_xact_lock(:lock_id)").bindparams(lock_id=lock_id))
                    
                    user_seq = self._get_next_sequence_number_in_session(db, session_id)
                    assistant_seq = user_seq + 1
                    
                    user_message = ChatMessage(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="user",
                        content=user_content,
                        message_metadata=user_metadata or {},
                        sequence_number=user_seq
                    )
                    
                    assistant_message = ChatMessage(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="assistant",
                        content=assistant_content,
                        message_metadata=assistant_metadata or {},
                        sequence_number=assistant_seq
                    )
                    
                    db.add_all([user_message, assistant_message])
                    db.flush()
                    db.refresh(user_message)
                    db.refresh(assistant_message)
                    
                    self._cleanup_old_messages_in_session(db, session_id)
                    db.expunge(user_message)
                    db.expunge(assistant_message)
                    
                    logger.debug(f"Added message pair to session {session_id} with sequences {user_seq}, {assistant_seq}")
                
                self._invalidate_cache(session_id)
                return user_message, assistant_message
            except IntegrityError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (time.time() % 0.01)
                    logger.warning(
                        f"Sequence number collision for message pair in session {session_id}, "
                        f"attempt {attempt + 1}/{max_retries}. Retrying after {delay:.3f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Failed to add message pair after {max_retries} retries due to sequence number collision "
                        f"for session {session_id}: {e}"
                    )
                    return None, None
            except Exception as e:
                logger.error(f"Failed to add message pair to session {session_id}: {e}", exc_info=True)
                return None, None
        
        return None, None
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        include_deleted: bool = False
    ) -> List[ChatMessage]:
        """Get messages for a session, excluding soft-deleted messages by default."""
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
                
                if not include_deleted:
                    query = query.filter(ChatMessage.deleted_at.is_(None))
                
                query = query.order_by(ChatMessage.sequence_number.asc())
                
                if offset > 0:
                    query = query.offset(offset)
                
                if limit:
                    query = query.limit(limit)
                
                messages = query.all()
                for msg in messages:
                    db.expunge(msg)
                return messages
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}", exc_info=True)
            return []
    
    def get_recent_messages(
        self,
        session_id: str,
        n: int = 10,
        include_deleted: bool = False
    ) -> List[ChatMessage]:
        """Get the most recent N messages for a session with caching."""
        cache_key = f"{session_id}:{n}:{include_deleted}"
        with self._cache_lock:
            if cache_key in self._recent_messages_cache:
                logger.debug(f"Cache hit for recent messages: {cache_key}")
                return self._recent_messages_cache[cache_key]
        
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
                
                if not include_deleted:
                    query = query.filter(ChatMessage.deleted_at.is_(None))
                
                messages = query.order_by(
                    ChatMessage.sequence_number.desc()
                ).limit(n).all()
                
                for msg in messages:
                    db.expunge(msg)
                
                result = list(reversed(messages))
                
                with self._cache_lock:
                    self._recent_messages_cache[cache_key] = result
                
                return result
        except Exception as e:
            logger.error(f"Failed to get recent messages for session {session_id}: {e}", exc_info=True)
            return []
    
    def clear_history(self, session_id: str, soft_delete: bool = True) -> bool:
        """
        Clear all messages for a session.
        
        Args:
            session_id: Session identifier
            soft_delete: If True, soft-delete messages (preserve for audit). If False, hard-delete.
        """
        try:
            with get_db_session() as db:
                if soft_delete:
                    updated = db.query(ChatMessage).filter(
                        ChatMessage.session_id == session_id,
                        ChatMessage.deleted_at.is_(None)
                    ).update(
                        {ChatMessage.deleted_at: datetime.utcnow()},
                        synchronize_session=False
                    )
                    logger.info(f"Soft-deleted {updated} messages for session {session_id}")
                else:
                    deleted = db.query(ChatMessage).filter(
                        ChatMessage.session_id == session_id
                    ).delete(synchronize_session=False)
                    logger.info(f"Hard-deleted {deleted} messages for session {session_id}")
                
                self._invalidate_cache(session_id)
                return True
        except Exception as e:
            logger.error(f"Failed to clear history for session {session_id}: {e}", exc_info=True)
            return False
    
    def get_message_count(self, session_id: str, include_deleted: bool = False) -> int:
        """Get the number of messages in a session."""
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
                
                if not include_deleted:
                    query = query.filter(ChatMessage.deleted_at.is_(None))
                
                return query.count()
        except Exception as e:
            logger.error(f"Failed to get message count for session {session_id}: {e}", exc_info=True)
            return 0
    
    def _get_session_lock_id(self, session_id: str) -> int:
        """Generate consistent integer lock ID from session_id for PostgreSQL advisory locks."""
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        return abs(hash_value) % (2**63 - 1)
    
    def _get_next_sequence_number_in_session(self, db, session_id: str) -> int:
        """
        Get the next sequence number for a session with row-level locking.
        
        Note: Does NOT filter by deleted_at because the unique constraint applies to all rows.
        Must be called within a transaction that holds an advisory lock.
        """
        try:
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).with_for_update(
                skip_locked=False
            ).order_by(
                ChatMessage.sequence_number.desc()
            ).first()
            
            if last_message:
                return last_message.sequence_number + 1
            return 1
        except Exception as e:
            logger.error(f"Failed to get next sequence number: {e}", exc_info=True)
            return 1
    
    def _cleanup_old_messages_in_session(self, db, session_id: str):
        """Remove old messages if exceeding threshold (90% of max_messages)."""
        try:
            count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id,
                ChatMessage.deleted_at.is_(None)
            ).count()
            
            if count > self._cleanup_threshold:
                if count > self.max_messages:
                    messages_to_delete = count - self.max_messages
                    target_count = self.max_messages
                else:
                    messages_to_delete = count - self._cleanup_target
                    target_count = self._cleanup_target
                
                if messages_to_delete > 0:
                    messages_to_keep = db.query(ChatMessage.sequence_number).filter(
                        ChatMessage.session_id == session_id,
                        ChatMessage.deleted_at.is_(None)
                    ).order_by(
                        ChatMessage.sequence_number.desc()
                    ).limit(target_count).all()
                    
                    if messages_to_keep:
                        min_seq_to_keep = min(msg[0] for msg in messages_to_keep)
                        
                        deleted = db.query(ChatMessage).filter(
                            ChatMessage.session_id == session_id,
                            ChatMessage.deleted_at.is_(None),
                            ChatMessage.sequence_number < min_seq_to_keep
                        ).delete(synchronize_session=False)
                        
                        if deleted > 0:
                            logger.info(
                                f"Cleaned up {deleted} old messages for session {session_id} "
                                f"(count was {count}, target is {target_count}, limit is {self.max_messages})"
                            )
        except Exception as e:
            logger.error(f"Failed to cleanup old messages for session {session_id}: {e}", exc_info=True)
    
    def _invalidate_cache(self, session_id: str):
        """Invalidate cache entries for a session."""
        with self._cache_lock:
            keys_to_remove = [
                key for key in self._recent_messages_cache.keys()
                if key.startswith(f"{session_id}:")
            ]
            for key in keys_to_remove:
                self._recent_messages_cache.pop(key, None)
    
    def clear_cache(self):
        """Clear all cached messages."""
        with self._cache_lock:
            self._recent_messages_cache.clear()
            logger.debug("Cleared chat history cache")
