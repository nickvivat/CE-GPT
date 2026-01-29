#!/usr/bin/env python3
"""
Migration script for chat history improvements.
Ensures composite indexes exist on chat_messages:
  - (session_id, sequence_number)
  - (session_id, timestamp)
Run this script if you have an existing database that needs these indexes.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text, inspect
from src.utils.database import get_engine, check_database_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def index_exists(engine, index_name: str) -> bool:
    """Check if an index exists on chat_messages."""
    inspector = inspect(engine)
    indexes = [idx['name'] for idx in inspector.get_indexes('chat_messages')]
    return index_name in indexes


def migrate_database():
    """Ensure chat_messages has the required composite indexes."""
    try:
        logger.info("Checking database connection...")
        if not check_database_connection():
            logger.error("Database connection failed. Please check your DATABASE_URL configuration.")
            sys.exit(1)
        
        engine = get_engine()
        
        with engine.connect() as conn:
            trans = conn.begin()
            
            try:
                indexes_to_create = [
                    ('idx_session_sequence', 'CREATE INDEX idx_session_sequence ON chat_messages (session_id, sequence_number)'),
                    ('idx_session_timestamp', 'CREATE INDEX idx_session_timestamp ON chat_messages (session_id, timestamp)'),
                ]
                
                for index_name, create_sql in indexes_to_create:
                    if not index_exists(engine, index_name):
                        logger.info(f"Creating index {index_name}...")
                        conn.execute(text(create_sql))
                        logger.info(f"✓ Created index {index_name}")
                    else:
                        logger.info(f"✓ Index {index_name} already exists")
                
                trans.commit()
                logger.info("Database migration completed successfully!")
                
            except Exception as e:
                trans.rollback()
                logger.error(f"Migration failed, rolled back: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to migrate database: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    migrate_database()
