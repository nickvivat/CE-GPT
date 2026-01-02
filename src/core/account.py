"""
Account management module for user accounts.
Handles account creation and retrieval for testing purposes.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.orm import relationship

from ..utils.database import Base, get_db_session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Account(Base):
    """Database model for user accounts."""
    __tablename__ = "accounts"
    
    acc_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    email = Column(String, nullable=True)
    role = Column(String, nullable=True)
    birth_date = Column(DateTime, nullable=True)
    gender = Column(String, nullable=True)
    year = Column(String, nullable=True)
    path_image = Column(Text, nullable=True)
    full_name_en = Column(String, nullable=True)
    full_name_th = Column(String, nullable=True)
    
    # Relationship to sessions (if needed in the future)
    # sessions = relationship("Session", back_populates="account")


def get_or_create_test_account(acc_id: str = "test_user") -> Optional[str]:
    """
    Get or create a test account for testing purposes.
    
    Args:
        acc_id: Account ID to use for the test account (default: "test_user")
    
    Returns:
        The account ID (acc_id) if successful, None otherwise
    """
    try:
        with get_db_session() as db:
            # Try to get existing account
            account = db.query(Account).filter(Account.acc_id == acc_id).first()
            
            if account:
                logger.info(f"Using existing test account: {acc_id}")
                return account.acc_id
            
            # Create new test account
            account = Account(
                acc_id=acc_id,
                email=f"{acc_id}@test.com",
                role="student",
                full_name_en="Test User",
                full_name_th="ผู้ใช้ทดสอบ"
            )
            db.add(account)
            db.flush()
            logger.info(f"Created test account: {acc_id}")
            return account.acc_id
    except Exception as e:
        logger.error(f"Failed to get or create test account: {e}")
        return None

