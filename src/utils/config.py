"""
Configuration settings for the Multilingual RAG System.

This module contains all configuration constants and settings used throughout the system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for AI models used in the system."""
    
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b-it-qat"
    
    def __post_init__(self):
        """Validate model configurations."""
        if not self.embedding_model:
            raise ValueError("Embedding model must be specified")
        if not self.reranker_model:
            raise ValueError("Reranker model must be specified")
        if not self.ollama_url:
            raise ValueError("Ollama URL must be specified")
        if not self.ollama_model:
            raise ValueError("Ollama model must be specified")
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create configuration from environment variables."""
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")
        )


@dataclass
class SearchConfig:
    """Configuration for search and retrieval parameters."""
    
    top_k: int = 50
    top_k_rerank: int = 20
    similarity_threshold: float = 0.1
    rerank_threshold: float = 0.05
    
    def __post_init__(self):
        """Validate search configurations."""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k_rerank <= 0:
            raise ValueError("top_k_rerank must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if not 0 <= self.rerank_threshold <= 1:
            raise ValueError("rerank_threshold must be between 0 and 1")
    
    @classmethod
    def from_env(cls) -> 'SearchConfig':
        """Create configuration from environment variables."""
        return cls(
            top_k=int(os.getenv("TOP_K", "50")),
            top_k_rerank=int(os.getenv("TOP_K_RERANK", "20")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.1")),
            rerank_threshold=float(os.getenv("RERANK_THRESHOLD", "0.05"))
        )


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""
    
    batch_size: int = 32
    max_workers: int = 4
    chunk_overlap: int = 0
    max_chunk_size: Optional[int] = None
    
    def __post_init__(self):
        """Validate processing configurations."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
    
    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        """Create configuration from environment variables."""
        return cls(
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "0")),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "0")) if os.getenv("MAX_CHUNK_SIZE") else None
        )


@dataclass
class CacheConfig:
    """Configuration for caching parameters."""
    
    embeddings_dir: str = "cache/embeddings"
    chunks_dir: str = "cache/chunks"
    max_cache_size: int = 1024  # MB
    cache_ttl: int = 3600  # seconds
    
    def __post_init__(self):
        """Validate cache configurations."""
        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create configuration from environment variables."""
        return cls(
            embeddings_dir=os.getenv("CACHE_EMBEDDINGS_DIR", "cache/embeddings"),
            chunks_dir=os.getenv("CACHE_CHUNKS_DIR", "cache/chunks"),
            max_cache_size=int(os.getenv("CACHE_MAX_SIZE", "1024")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600"))
        )


@dataclass
class APIConfig:
    """Configuration for API settings."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    max_requests: int = 1000
    timeout: int = 30
    
    def __post_init__(self):
        """Validate API configurations."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            workers=int(os.getenv("API_WORKERS", "1")),
            max_requests=int(os.getenv("API_MAX_REQUESTS", "1000")),
            timeout=int(os.getenv("API_TIMEOUT", "30"))
        )


@dataclass
class DebugConfig:
    """Configuration for debug and development settings."""
    
    debug: bool = False
    log_level: str = "INFO"
    enable_profiling: bool = False
    enable_tracing: bool = False
    
    def __post_init__(self):
        """Validate debug configurations."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
    
    @classmethod
    def from_env(cls) -> 'DebugConfig':
        """Create configuration from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true",
            enable_tracing=os.getenv("ENABLE_TRACING", "false").lower() == "true"
        )


@dataclass
class Config:
    """Main configuration class that combines all configuration sections."""
    
    models: ModelConfig = field(default_factory=ModelConfig.from_env)
    search: SearchConfig = field(default_factory=SearchConfig.from_env)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig.from_env)
    cache: CacheConfig = field(default_factory=CacheConfig.from_env)
    api: APIConfig = field(default_factory=APIConfig.from_env)
    debug: DebugConfig = field(default_factory=DebugConfig.from_env)
    
    def __post_init__(self):
        """Validate overall configuration."""
        # Ensure cache directories exist
        os.makedirs(self.cache.embeddings_dir, exist_ok=True)
        os.makedirs(self.cache.chunks_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "models": self.models.__dict__,
            "search": self.search.__dict__,
            "processing": self.processing.__dict__,
            "cache": self.cache.__dict__,
            "api": self.api.__dict__,
            "debug": self.debug.__dict__
        }
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        self.models = ModelConfig.from_env()
        self.search = SearchConfig.from_env()
        self.processing = ProcessingConfig.from_env()
        self.cache = CacheConfig.from_env()
        self.api = APIConfig.from_env()
        self.debug = DebugConfig.from_env()


# Global configuration instance
config = Config()

# Update configuration from environment variables
config.update_from_env()
