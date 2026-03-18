"""
Error handling and retry logic for the RAG system.
Implements retry mechanisms, circuit breaker pattern, and comprehensive error handling.
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, Type, Union, List
from enum import Enum
import requests
import torch
import numpy as np

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in the RAG system."""
    NETWORK = "network"
    MODEL_LOADING = "model_loading"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_SEARCH = "vector_search"
    RERANKING = "reranking"
    RESPONSE_GENERATION = "response_generation"
    DATA_PROCESSING = "data_processing"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state


class RetryHandler:
    """Retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Function failed after {self.max_retries + 1} attempts: {e}")
                    raise e
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        raise last_exception


def handle_errors(error_type: ErrorType, fallback_value: Any = None, 
                  log_level: str = "ERROR") -> Callable:
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Bypass fallback for policy rejections or guardrail triggers
                if (isinstance(e, ValueError) and str(e) == "ABUSIVE_QUERY") or \
                   (type(e).__name__ == "GuardrailException"):
                    raise e
                    
                # Log the error with context
                log_func = getattr(logger, log_level.lower())
                log_func(f"{error_type.value.upper()} error in {func.__name__}: {e}")
                
                # Handle specific error types
                if error_type == ErrorType.NETWORK:
                    if isinstance(e, requests.exceptions.RequestException):
                        logger.error(f"Network error details: {type(e).__name__}")
                    elif isinstance(e, requests.exceptions.Timeout):
                        logger.error("Request timed out")
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        logger.error("Connection failed")
                
                elif error_type == ErrorType.MODEL_LOADING:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        logger.error("CUDA out of memory - consider using CPU or smaller batch size")
                    elif isinstance(e, FileNotFoundError):
                        logger.error("Model file not found - check model path")
                
                elif error_type == ErrorType.EMBEDDING_GENERATION:
                    if isinstance(e, ValueError):
                        logger.error("Invalid input for embedding generation")
                    elif isinstance(e, RuntimeError):
                        logger.error("Runtime error during embedding generation")
                
                elif error_type == ErrorType.VECTOR_SEARCH:
                    if isinstance(e, IndexError):
                        logger.error("Vector index error - check if index is built")
                    elif isinstance(e, ValueError):
                        logger.error("Invalid search parameters")
                
                # Return fallback value if provided
                if fallback_value is not None:
                    logger.info(f"Returning fallback value for {func.__name__}")
                    return fallback_value
                
                # Re-raise the exception
                raise e
        
        return wrapper
    return decorator


# Common validators
def is_positive_number(value: Any) -> bool:
    """Check if value is a positive number."""
    return isinstance(value, (int, float)) and value > 0


def is_valid_language(value: Any) -> bool:
    """Check if value is a valid language code."""
    return isinstance(value, str) and value in ["en", "th"]


def is_valid_similarity_threshold(value: Any) -> bool:
    """Check if value is a valid similarity threshold."""
    return isinstance(value, (int, float)) and 0 <= value <= 1


# Error recovery strategies
def recover_from_model_error(error: Exception, fallback_model: str = None) -> str:
    """Recover from model loading errors."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        logger.info("Falling back to CPU for model loading")
        return "cpu"
    elif isinstance(error, FileNotFoundError) and fallback_model:
        logger.info(f"Falling back to alternative model: {fallback_model}")
        return fallback_model
    else:
        raise error


def recover_from_network_error(error: Exception, retry_count: int = 0) -> bool:
    """Recover from network errors."""
    if isinstance(error, requests.exceptions.Timeout):
        if retry_count < 3:
            logger.info(f"Network timeout, retrying... (attempt {retry_count + 1})")
            return True
    elif isinstance(error, requests.exceptions.ConnectionError):
        if retry_count < 5:
            logger.info(f"Connection error, retrying... (attempt {retry_count + 1})")
            return True
    
    return False


# Performance monitoring decorator
def monitor_performance(operation_name: str = None) -> Callable:
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = None
            
            try:
                # Try to get memory usage if psutil is available
                try:
                    import psutil
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass
                
                result = func(*args, **kwargs)
                
                # Log performance metrics
                execution_time = time.time() - start_time
                op_name = operation_name or func.__name__
                
                logger.info(f"Performance: {op_name} completed in {execution_time:.4f}s")
                
                if start_memory is not None:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024
                        memory_delta = end_memory - start_memory
                        logger.info(f"Memory usage: {memory_delta:+.2f}MB")
                    except:
                        pass
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                op_name = operation_name or func.__name__
                logger.error(f"Performance: {op_name} failed after {execution_time:.4f}s: {e}")
                raise e
        
        return wrapper
    return decorator
