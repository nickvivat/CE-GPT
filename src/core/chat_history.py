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

# Import deferred to avoid circular import (history_compressor does not depend on chat_history)
def _compress_messages(messages, recent_count, summary_max_tokens, llm_client):
    from .history_compressor import compress
    return compress(messages, recent_count, summary_max_tokens, llm_client)


class ChatMessage(Base):
    """Database model for chat messages."""
    __tablename__ = "chat_messages"
    
    message_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    message_metadata = Column(JSONB, default=dict)
    sequence_number = Column(Integer, nullable=False, index=True)
    
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence_number', name='uq_session_sequence'),
        Index('idx_session_sequence', 'session_id', 'sequence_number'),
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
    )
    
    session = relationship("Session", back_populates="messages")


class SessionCompressionSummary(Base):
    """Stored compressed summary for a session (hybrid cache: DB persistence).
    Key is (session_id, message_count, messages_length) so we don't reuse a summary
    computed from a different message set (e.g. filtered vs unfiltered list).
    """
    __tablename__ = "session_compression_summaries"

    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), primary_key=True)
    message_count = Column(Integer, primary_key=True)
    messages_length = Column(Integer, primary_key=True)  # len(messages) actually compressed; distinguishes filtered vs unfiltered
    summary_text = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ChatHistoryManager:
    """Manages chat message history with optimized performance and concurrency handling."""
    
    def __init__(self):
        """Initialize the chat history manager."""
        self.max_messages = config.session.max_messages_per_session
        self._recent_messages_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        self._compression_summary_cache: TTLCache = TTLCache(maxsize=500, ttl=600)
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
        offset: int = 0
    ) -> List[ChatMessage]:
        """Get messages for a session."""
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
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
        n: int = 10
    ) -> List[ChatMessage]:
        """Get the most recent N messages for a session with caching."""
        cache_key = f"{session_id}:{n}"
        with self._cache_lock:
            if cache_key in self._recent_messages_cache:
                logger.debug(f"Cache hit for recent messages: {cache_key}")
                return self._recent_messages_cache[cache_key]
        
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
                
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
    
    def get_messages_for_compression(
        self,
        session_id: str,
        max_messages: int
    ) -> List[ChatMessage]:
        """Get the last max_messages for a session in chronological order (for history compression)."""
        return self.get_recent_messages(session_id, n=max_messages)
    
    def _compression_cache_key(
        self, session_id: str, message_count: int, messages_length: int
    ) -> str:
        """Cache key for compression summary; includes messages_length so filtered vs unfiltered lists don't share entries."""
        return f"compression:{session_id}:{message_count}:{messages_length}"
    
    def get_compressed_summary_cached(
        self, session_id: str, message_count: int, messages_length: int
    ) -> Optional[str]:
        """Return cached or DB-stored summary for (session_id, message_count, messages_length), or None."""
        cache_key = self._compression_cache_key(session_id, message_count, messages_length)
        with self._cache_lock:
            if cache_key in self._compression_summary_cache:
                return self._compression_summary_cache[cache_key]
        try:
            with get_db_session() as db:
                row = db.query(SessionCompressionSummary).filter(
                    SessionCompressionSummary.session_id == session_id,
                    SessionCompressionSummary.message_count == message_count,
                    SessionCompressionSummary.messages_length == messages_length,
                ).first()
                if row:
                    summary = row.summary_text
                    with self._cache_lock:
                        self._compression_summary_cache[cache_key] = summary
                    return summary
                # Backward compat: old rows may have messages_length=0; use when messages_length == message_count
                if messages_length == message_count:
                    row_legacy = db.query(SessionCompressionSummary).filter(
                        SessionCompressionSummary.session_id == session_id,
                        SessionCompressionSummary.message_count == message_count,
                        SessionCompressionSummary.messages_length == 0,
                    ).first()
                    if row_legacy:
                        summary = row_legacy.summary_text
                        with self._cache_lock:
                            self._compression_summary_cache[cache_key] = summary
                        return summary
        except Exception as e:
            logger.debug("Failed to load compression summary from DB: %s", e)
        return None
    
    def store_compressed_summary(
        self,
        session_id: str,
        message_count: int,
        messages_length: int,
        summary: str,
    ) -> None:
        """Persist summary to DB and in-memory cache (hybrid storage)."""
        cache_key = self._compression_cache_key(session_id, message_count, messages_length)
        try:
            with get_db_session() as db:
                row = db.query(SessionCompressionSummary).filter(
                    SessionCompressionSummary.session_id == session_id,
                    SessionCompressionSummary.message_count == message_count,
                    SessionCompressionSummary.messages_length == messages_length,
                ).first()
                if row:
                    row.summary_text = summary
                else:
                    db.add(SessionCompressionSummary(
                        session_id=session_id,
                        message_count=message_count,
                        messages_length=messages_length,
                        summary_text=summary,
                    ))
            with self._cache_lock:
                self._compression_summary_cache[cache_key] = summary
        except Exception as e:
            logger.warning("Failed to store compression summary: %s", e)
    
    def get_or_compute_compressed_history(
        self,
        session_id: str,
        messages: List[ChatMessage],
        recent_count: int,
        summary_max_tokens: int,
        llm_client: Any,
        message_count_for_cache: Optional[int] = None,
    ):
        """Return compressed history from cache/DB or compute and store (hybrid).

        Cache key is (session_id, message_count_for_cache, len(messages)) so we never
        reuse a summary computed from a different message set (e.g. filtered vs
        unfiltered list). message_count_for_cache is the DB count; len(messages) is
        the actual list length being compressed.
        """
        from .history_compressor import CompressedHistory
        messages_length = len(messages)
        cache_key_count = message_count_for_cache if message_count_for_cache is not None else messages_length
        if messages_length <= recent_count:
            return _compress_messages(
                messages, recent_count, summary_max_tokens, llm_client
            )
        cached_summary = self.get_compressed_summary_cached(
            session_id, cache_key_count, messages_length
        )
        if cached_summary is not None:
            recent = messages[-recent_count:]
            return CompressedHistory(summary=cached_summary, recent_messages=recent)
        result = _compress_messages(
            messages, recent_count, summary_max_tokens, llm_client
        )
        if result.summary:
            self.store_compressed_summary(
                session_id, cache_key_count, messages_length, result.summary
            )
        return result
    
    def clear_history(self, session_id: str) -> bool:
        """
        Clear all messages for a session.
        
        Args:
            session_id: Session identifier
        """
        try:
            with get_db_session() as db:
                deleted = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).delete(synchronize_session=False)
                deleted_summaries = db.query(SessionCompressionSummary).filter(
                    SessionCompressionSummary.session_id == session_id
                ).delete(synchronize_session=False)
                logger.info(f"Deleted {deleted} messages and {deleted_summaries} compression summaries for session {session_id}")
                
                self._invalidate_cache(session_id)
                self._invalidate_compression_cache(session_id)
                return True
        except Exception as e:
            logger.error(f"Failed to clear history for session {session_id}: {e}", exc_info=True)
            return False
    
    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        try:
            with get_db_session() as db:
                query = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
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
                ChatMessage.session_id == session_id
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
                        ChatMessage.session_id == session_id
                    ).order_by(
                        ChatMessage.sequence_number.desc()
                    ).limit(target_count).all()
                    
                    if messages_to_keep:
                        min_seq_to_keep = min(msg[0] for msg in messages_to_keep)
                        
                        deleted = db.query(ChatMessage).filter(
                            ChatMessage.session_id == session_id,
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
        self._invalidate_compression_cache(session_id)
    
    def _invalidate_compression_cache(self, session_id: str):
        """Invalidate in-memory compression summaries for a session."""
        with self._cache_lock:
            prefix = f"compression:{session_id}:"
            keys_to_remove = [
                key for key in self._compression_summary_cache.keys()
                if isinstance(key, str) and key.startswith(prefix)
            ]
            for key in keys_to_remove:
                self._compression_summary_cache.pop(key, None)
    
    def clear_cache(self):
        """Clear all cached messages and compression summaries."""
        with self._cache_lock:
            self._recent_messages_cache.clear()
            self._compression_summary_cache.clear()
            logger.debug("Cleared chat history and compression caches")
