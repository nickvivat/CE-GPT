"""
Pydantic models for API request/response validation.
Defines the data structures used in the CE RAG API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime


class LanguageEnum(str, Enum):
    """Supported languages for the RAG system."""
    THAI = "th"
    ENGLISH = "en"
    AUTO = "auto"


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query for response generation")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of sources to retrieve")
    language: str = Field(default="auto", description="Language preference (auto, en, th)")
    use_reranking: bool = Field(default=True, description="Whether to use reranking for better results")
    include_sources: bool = Field(default=True, description="Whether to include source information")
    stream: bool = Field(default=True, description="Whether to stream the response for better UX")


class GenerateResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Generated AI response")
    session_id: Optional[str] = Field(None, description="Session ID if session was used")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents used for generation")
    language_detected: str = Field(..., description="Detected language of the query")
    generation_time_ms: float = Field(..., description="Response generation time in milliseconds")
    total_sources: int = Field(..., description="Total number of sources used")


class SystemStatus(BaseModel):
    """System status and health information."""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="System version")
    total_chunks: int = Field(..., description="Total number of chunks")
    vector_store_count: int = Field(..., description="Vector store document count")
    reranker_enabled: bool = Field(..., description="Reranker status")
    query_enhancement_enabled: bool = Field(..., description="Query enhancement status")
    response_generation_enabled: bool = Field(..., description="Response generation status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    last_data_update: Optional[datetime] = Field(None, description="Last data update timestamp")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class PerformanceMetrics(BaseModel):
    """Performance monitoring metrics."""
    operation_stats: Dict[str, Dict[str, Any]] = Field(..., description="Operation statistics")
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    export_timestamp: datetime = Field(default_factory=datetime.now, description="Export timestamp")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")


class SessionRequest(BaseModel):
    """Request model for creating a session."""
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session metadata")
    ttl_hours: Optional[int] = Field(None, ge=1, le=168, description="Session TTL in hours (1-168)")


class SessionResponse(BaseModel):
    """Response model for session information."""
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    is_active: bool = Field(..., description="Whether the session is active")


class Message(BaseModel):
    """Chat message model."""
    message_id: str = Field(..., description="Message identifier")
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    sequence_number: int = Field(..., description="Message sequence number")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[Message] = Field(..., description="List of messages")
    total_count: int = Field(..., description="Total number of messages")


class SessionUpdateRequest(BaseModel):
    """Request model for updating a session."""
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    extend_ttl: bool = Field(False, description="Whether to extend session TTL")
