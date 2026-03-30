#!/usr/bin/env python3
"""
Database initialization script.
Run this script to create the database tables.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.database import init_database, check_database_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Initialize the database."""
    try:
        logger.info("Checking database connection...")
        if not check_database_connection():
            logger.error("Database connection failed. Please check your DATABASE_URL configuration.")
            sys.exit(1)
        
        logger.info("Initializing database tables...")
        init_database()
        logger.info("Database initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

