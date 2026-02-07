#!/usr/bin/env python3
"""
Migration script for session_compression_summaries: add messages_length to the key.
Run this if you have an existing database created before this column was added.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text, inspect
from src.utils.database import get_engine, check_database_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def column_exists(engine, table: str, column: str) -> bool:
    """Check if a column exists on the table."""
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns(table)]
    return column in cols


def migrate_database():
    """Add messages_length column and update PK if needed."""
    try:
        logger.info("Checking database connection...")
        if not check_database_connection():
            logger.error("Database connection failed. Check DATABASE_URL.")
            sys.exit(1)

        engine = get_engine()
        inspector = inspect(engine)
        if "session_compression_summaries" not in inspector.get_table_names():
            logger.info("Table session_compression_summaries does not exist; nothing to migrate.")
            return

        if column_exists(engine, "session_compression_summaries", "messages_length"):
            logger.info("Column messages_length already exists; nothing to migrate.")
            return

        with engine.connect() as conn:
            trans = conn.begin()
            try:
                logger.info("Adding column messages_length...")
                conn.execute(text(
                    "ALTER TABLE session_compression_summaries "
                    "ADD COLUMN messages_length INTEGER NOT NULL DEFAULT 0"
                ))
                logger.info("Dropping old primary key...")
                conn.execute(text(
                    "ALTER TABLE session_compression_summaries "
                    "DROP CONSTRAINT session_compression_summaries_pkey"
                ))
                logger.info("Adding new primary key (session_id, message_count, messages_length)...")
                conn.execute(text(
                    "ALTER TABLE session_compression_summaries "
                    "ADD PRIMARY KEY (session_id, message_count, messages_length)"
                ))
                trans.commit()
                logger.info("Migration completed successfully.")
            except Exception as e:
                trans.rollback()
                raise
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    migrate_database()
