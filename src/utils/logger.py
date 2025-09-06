"""
Logging configuration and setup for the Multilingual RAG System.

This module provides centralized logging configuration with proper formatting,
file handling, and console output.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from .config import config


def setup_logger(
    name: str = "multilingual_rag",
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    enable_console: Optional[bool] = None
) -> logging.Logger:
    """
    Set up a logger with proper configuration.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        log_level: Logging level (optional)
        enable_console: Whether to enable console logging (optional)
    
    Returns:
        Configured logger instance
    """
    # Use configuration defaults if not specified
    log_level = log_level or config.debug.log_level
    enable_console = enable_console if enable_console is not None else True
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s"
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "multilingual_rag") -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    Set the log level for a logger.
    
    Args:
        logger: Logger instance
        level: Log level string
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper()))


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: Optional[str] = None
) -> None:
    """
    Add a file handler to an existing logger.
    
    Args:
        logger: Logger instance
        log_file: Path to log file
        level: Log level (optional)
    """
    if level is None:
        level = config.debug.log_level
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Use detailed formatter for file logging
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


def remove_file_handler(logger: logging.Logger, log_file: str) -> None:
    """
    Remove a file handler from a logger.
    
    Args:
        logger: Logger instance
        log_file: Path to log file
    """
    handlers_to_remove = []
    
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            if handler.baseFilename == os.path.abspath(log_file):
                handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        logger.removeHandler(handler)


# Create default logger
default_logger = setup_logger()
