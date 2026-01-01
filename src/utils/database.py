"""
Database management module for session and chat history storage.
Provides PostgreSQL connection management and database initialization.
"""

import os
from typing import Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

# Base class for database models
Base = declarative_base()

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        try:
            database_url = config.database.get_url()
            _engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                echo=config.database.echo,
                pool_pre_ping=True,  # Verify connections before using
            )
            logger.info(f"Database engine created: {database_url.split('@')[-1]}")
        except ValueError as e:
            # Re-raise ValueError with clear message
            logger.error(f"Database configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        logger.info("Session factory created")
    return _SessionLocal


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def init_database():
    """Initialize database tables."""
    try:
        engine = get_engine()
        # Import models to register them with Base
        from ..core.session_manager import Session as SessionModel
        from ..core.chat_history import ChatMessage
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def check_database_connection() -> bool:
    """Check if database connection is healthy."""
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def close_database():
    """Close database connections."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
        _engine = None
        logger.info("Database engine closed")
    _SessionLocal = None

