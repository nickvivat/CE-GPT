#!/usr/bin/env python3
"""
Migration script for chat history improvements.
Adds deleted_at column and composite indexes to existing chat_messages table.
Run this script if you have an existing database that needs to be migrated.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text, inspect
from src.utils.database import get_engine, check_database_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def column_exists(engine, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def index_exists(engine, index_name: str) -> bool:
    """Check if an index exists."""
    inspector = inspect(engine)
    indexes = [idx['name'] for idx in inspector.get_indexes('chat_messages')]
    return index_name in indexes


def migrate_database():
    """Migrate existing database to add new features."""
    try:
        logger.info("Checking database connection...")
        if not check_database_connection():
            logger.error("Database connection failed. Please check your DATABASE_URL configuration.")
            sys.exit(1)
        
        engine = get_engine()
        
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Check if deleted_at column exists
                if not column_exists(engine, 'chat_messages', 'deleted_at'):
                    logger.info("Adding deleted_at column to chat_messages table...")
                    conn.execute(text("""
                        ALTER TABLE chat_messages 
                        ADD COLUMN deleted_at TIMESTAMP NULL
                    """))
                    logger.info("✓ Added deleted_at column")
                else:
                    logger.info("✓ deleted_at column already exists")
                
                # Create indexes if they don't exist
                indexes_to_create = [
                    ('idx_session_sequence', 'CREATE INDEX idx_session_sequence ON chat_messages (session_id, sequence_number)'),
                    ('idx_session_timestamp', 'CREATE INDEX idx_session_timestamp ON chat_messages (session_id, timestamp)'),
                    ('idx_session_deleted', 'CREATE INDEX idx_session_deleted ON chat_messages (session_id, deleted_at)'),
                ]
                
                for index_name, create_sql in indexes_to_create:
                    if not index_exists(engine, index_name):
                        logger.info(f"Creating index {index_name}...")
                        conn.execute(text(create_sql))
                        logger.info(f"✓ Created index {index_name}")
                    else:
                        logger.info(f"✓ Index {index_name} already exists")
                
                # Commit transaction
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
