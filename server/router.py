"""
Main API router for the CE RAG System.
Defines all API endpoints and their handlers.
"""

import time
import os
import json
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.responses import Response
import asyncio

from .models import (
    GenerateRequest, GenerateResponse, SystemStatus, 
    PerformanceMetrics, HealthCheck, ErrorResponse,
    SessionRequest, SessionResponse, SessionUpdateRequest,
    ChatHistoryResponse, Message
)
from src.core.rag import RAGSystem
from src.core.session_manager import SessionManager
from src.core.chat_history import ChatHistoryManager
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.performance_monitor import performance_monitor
from src.utils.database import init_database, check_database_connection

logger = get_logger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["CE RAG System"])

# Global RAG system instance
rag_system: RAGSystem = None
system_start_time = time.time()

# Global session and chat history managers
session_manager: SessionManager = None
chat_history_manager: ChatHistoryManager = None


def get_rag_system() -> RAGSystem:
    """Dependency to get the RAG system instance."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


def initialize_rag_system():
    """Initialize the RAG system globally."""
    global rag_system, session_manager, chat_history_manager
    try:
        logger.info("Initializing database...")
        # Initialize database and create tables
        init_database()
        
        # Check database connection
        if not check_database_connection():
            logger.warning("Database connection check failed, but continuing...")
        
        # Initialize session and chat history managers
        session_manager = SessionManager()
        chat_history_manager = ChatHistoryManager()
        logger.info("Session and chat history managers initialized")
        
        logger.info("Initializing RAG system for API...")
        # RAG system will auto-load data and build vector index
        rag_system = RAGSystem(
            use_reranker=True,
            use_query_enhancement=True,
            auto_load_data=True,
            chat_history_manager=chat_history_manager
        )
        
        # Verify that data was loaded successfully
        if not rag_system.chunks or len(rag_system.chunks) == 0:
            raise Exception("No data loaded by RAG system")
        
        logger.info("RAG system ready for API requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Reset all components to maintain consistent state
        rag_system = None
        session_manager = None
        chat_history_manager = None
        raise e


def get_session_manager() -> SessionManager:
    """Dependency to get the session manager instance."""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    return session_manager


def get_chat_history_manager() -> ChatHistoryManager:
    """Dependency to get the chat history manager instance."""
    if chat_history_manager is None:
        raise HTTPException(status_code=503, detail="Chat history manager not initialized")
    return chat_history_manager


@router.get("/health", response_model=HealthCheck, summary="Health Check")
async def health_check():
    """Check system health and status."""
    try:
        uptime = time.time() - system_start_time
        return HealthCheck(
            status="healthy" if rag_system is not None else "unhealthy",
            version="1.0.0",
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="error",
            version="1.0.0",
            uptime_seconds=time.time() - system_start_time
        )


@router.get("/status", response_model=SystemStatus, summary="System Status")
async def get_system_status(rag: RAGSystem = Depends(get_rag_system)):
    """Get detailed system status and statistics."""
    try:
        status_data = rag.get_system_status()
        uptime = time.time() - system_start_time
        
        return SystemStatus(
            status="operational",
            version="1.0.0",
            total_chunks=status_data.get('total_chunks', 0),
            vector_store_count=status_data.get('vector_store_count', 0),
            reranker_enabled=status_data.get('reranker_enabled', False),
            query_enhancement_enabled=status_data.get('query_enhancement_enabled', False),
            response_generation_enabled=status_data.get('response_generation_enabled', False),
            uptime_seconds=uptime,
            performance_metrics=status_data.get('performance_metrics', {})
        )
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")




@router.post("/generate", summary="Generate AI Response")
async def generate_response(
    request: GenerateRequest,
    sm: SessionManager = Depends(get_session_manager),
    chm: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Generate an AI response based on the query and retrieved sources."""
    try:
        start_time = time.time()
        
        # Get RAG system
        rag_system = get_rag_system()
        
        # Handle session
        session_id = request.session_id
        if session_id:
            # Validate session exists - if provided, it must be valid
            session = sm.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        elif config.session.auto_create:
            # Auto-create session only when no session_id is provided and auto_create is enabled
            session = sm.create_session(metadata={"language": request.language})
            session_id = session.session_id
            logger.info(f"Auto-created session: {session_id}")
        
        # Determine language (same logic as search endpoint)
        language = None
        if request.language != "auto":
            language = request.language
        
        # First, search for relevant courses to get sources
        search_results = await rag_system.search(
            query=request.query,
            top_k=request.top_k,
            language=language,
            use_reranking=request.use_reranking,
            session_id=session_id
        )
        
        logger.info(f"Found {len(search_results)} search results for query: {request.query}")
        
        # Generate response using the search results
        response = await rag_system.generate_response(
            query=request.query,
            top_k=request.top_k,
            language=language,
            use_reranking=request.use_reranking,
            stream=request.stream,
            search_results=search_results,
            session_id=session_id
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        # Store messages in chat history if session exists (always store, even if response is empty)
        if session_id:
            user_msg = chm.add_message(
                session_id=session_id,
                role="user",
                content=request.query,
                metadata={"language": request.language}
            )
            if not user_msg:
                logger.error(f"Failed to store user message for session {session_id}")
            
            assistant_msg = chm.add_message(
                session_id=session_id,
                role="assistant",
                content=response or "",  # Store empty string if response is None/empty
                metadata={
                    "sources": len(search_results),
                    "language": rag_system._detect_language(request.query),
                    "generation_time_ms": generation_time
                }
            )
            if not assistant_msg:
                logger.error(f"Failed to store assistant message for session {session_id}")
            
            # Update session timestamp
            if not sm.update_session(session_id, extend_ttl=True):
                logger.error(f"Failed to update session {session_id} timestamp")
        
        # Prepare sources for response
        sources = []
        if request.include_sources and search_results:
            sources = [
                {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "rerank_score": result.get("rerank_score", 0.0),
                    "chunk_id": result.get("chunk_id", ""),
                    "original_index": result.get("original_index", 0)
                }
                for result in search_results
            ]
        
        return GenerateResponse(
            query=request.query,
            response=response,
            session_id=session_id,
            sources=sources,
            language_detected=rag_system._detect_language(request.query),
            generation_time_ms=generation_time,
            total_sources=len(search_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream", summary="Generate AI Response with Real-time Streaming")
async def generate_response_stream(
    request: GenerateRequest,
    sm: SessionManager = Depends(get_session_manager),
    chm: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Generate an AI response with real-time streaming using Server-Sent Events."""
    try:
        # Get RAG system
        rag_system = get_rag_system()
        
        # Handle session
        session_id = request.session_id
        if session_id:
            # Validate session exists - if provided, it must be valid
            session = sm.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        elif config.session.auto_create:
            # Auto-create session only when no session_id is provided and auto_create is enabled
            session = sm.create_session(metadata={"language": request.language})
            session_id = session.session_id
            logger.info(f"Auto-created session: {session_id}")
        
        # Determine language
        language = None
        if request.language != "auto":
            language = request.language
        
        # Search for relevant courses to get sources
        search_results = await rag_system.search(
            query=request.query,
            top_k=request.top_k,
            language=language,
            use_reranking=request.use_reranking,
            session_id=session_id
        )
        
        logger.info(f"Found {len(search_results)} search results for streaming query: {request.query}")
        
        # Store user message immediately before streaming starts to ensure it's saved
        user_message_saved = False
        if session_id:
            user_msg = chm.add_message(
                session_id=session_id,
                role="user",
                content=request.query,
                metadata={"language": request.language}
            )
            if user_msg:
                user_message_saved = True
            else:
                logger.error(f"Failed to store user message for session {session_id}")
        
        async def generate_stream():
            """Generator function for streaming response chunks."""
            response_text = ""
            assistant_message_saved = False
            try:
                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Starting generation...'})}\n\n"
                
                # Send search results info
                yield f"data: {json.dumps({'type': 'search_results', 'count': len(search_results)})}\n\n"
                
                if request.include_sources and search_results:
                    sources = [
                        {
                            "content": result.get("content", ""),
                            "metadata": result.get("metadata", {}),
                            "similarity_score": result.get("similarity_score", 0.0),
                            "rerank_score": result.get("rerank_score", 0.0),
                            "chunk_id": result.get("chunk_id", ""),
                            "original_index": result.get("original_index", 0)
                        }
                        for result in search_results
                    ]
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                # Get streaming response from RAG system and iterate over it
                async for chunk in rag_system.generate_response_stream(
                    query=request.query,
                    top_k=request.top_k,
                    language=language,
                    use_reranking=request.use_reranking,
                    search_results=search_results,
                    session_id=session_id
                ):
                    if chunk:
                        response_text += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
                completion_data = {
                    'type': 'complete', 
                    'message': 'Generation complete',
                    'language_detected': rag_system._detect_language(request.query),
                    'total_sources': len(search_results),
                    'session_id': session_id
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                # Store assistant message in finally block to ensure it's saved even if stream is interrupted
                # Always store, even if response_text is empty
                if session_id and not assistant_message_saved:
                    assistant_msg = chm.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=response_text or "",  # Store empty string if response_text is empty
                        metadata={
                            "sources": len(search_results),
                            "language": rag_system._detect_language(request.query)
                        }
                    )
                    if assistant_msg:
                        assistant_message_saved = True
                    else:
                        logger.error(f"Failed to store assistant message for session {session_id}")
                    
                    if not sm.update_session(session_id, extend_ttl=True):
                        logger.error(f"Failed to update session {session_id} timestamp")
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up streaming generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PerformanceMetrics, summary="Performance Metrics")
async def get_performance_metrics(rag: RAGSystem = Depends(get_rag_system)):
    """Get system performance metrics."""
    try:
        operation_stats = performance_monitor.get_operation_stats()
        
        return PerformanceMetrics(
            operation_stats=operation_stats,
            system_metrics=rag.get_system_status().get('performance_metrics', {})
        )
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.post("/performance/export", summary="Export Performance Data")
async def export_performance_data(
    background_tasks: BackgroundTasks,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Export performance data to file."""
    try:
        export_path = f"performance_export_{int(time.time())}.json"
        
        def export_task():
            try:
                rag.export_performance_data(export_path)
                logger.info(f"Performance data exported to {export_path}")
            except Exception as e:
                logger.error(f"Export failed: {e}")
        
        background_tasks.add_task(export_task)
        
        return {"message": "Export started", "file": export_path}
        
    except Exception as e:
        logger.error(f"Failed to start export: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start export: {str(e)}")


@router.post("/cache/clear", summary="Clear Cache")
async def clear_cache(rag: RAGSystem = Depends(get_rag_system)):
    """Clear system cache."""
    try:
        success = rag.clear_embedding_cache()
        if success:
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")


@router.post("/conversation/clear", summary="Clear Conversation Context")
async def clear_conversation_context(rag: RAGSystem = Depends(get_rag_system)):
    """Clear conversation context."""
    try:
        rag.clear_conversation_context()
        return {"message": "Conversation context cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear conversation context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation context: {str(e)}")


# Session management endpoints
@router.post("/sessions", response_model=SessionResponse, summary="Create Session")
async def create_session(
    request: SessionRequest,
    sm: SessionManager = Depends(get_session_manager)
):
    """Create a new session."""
    try:
        session = sm.create_session(
            user_id=request.user_id,
            metadata=request.metadata,
            ttl_hours=request.ttl_hours
        )
        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            metadata=session.session_metadata or {},
            is_active=session.is_active
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse, summary="Get Session")
async def get_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager)
):
    """Get session information."""
    try:
        session = sm.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            metadata=session.session_metadata or {},
            is_active=session.is_active
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}", response_model=SessionResponse, summary="Update Session")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    sm: SessionManager = Depends(get_session_manager)
):
    """Update session metadata or extend TTL."""
    try:
        success = sm.update_session(
            session_id=session_id,
            metadata=request.metadata,
            extend_ttl=request.extend_ttl
        )
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sm.get_session(session_id)
        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            metadata=session.session_metadata or {},
            is_active=session.is_active
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", summary="Delete Session")
async def delete_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager)
):
    """Delete a session."""
    try:
        success = sm.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse, summary="Get Chat History")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    sm: SessionManager = Depends(get_session_manager),
    chm: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Get chat history for a session."""
    try:
        # Validate session exists
        session = sm.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = chm.get_messages(session_id, limit=limit, offset=offset)
        total_count = chm.get_message_count(session_id)
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[
                Message(
                    message_id=msg.message_id,
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    sequence_number=msg.sequence_number,
                    metadata=msg.message_metadata or {}
                )
                for msg in messages
            ],
            total_count=total_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}/history", summary="Clear Chat History")
async def clear_chat_history(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
    chm: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Clear chat history for a session."""
    try:
        # Validate session exists
        session = sm.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        success = chm.clear_history(session_id)
        if success:
            return {"message": "Chat history cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear chat history")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", summary="API Root")
async def api_root():
    """API root endpoint."""
    return {
        "message": "CE RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "status": "/api/v1/status",
            "generate": "/api/v1/generate",
            "generate_stream": "/api/v1/generate/stream",
            "sessions": "/api/v1/sessions",
            "performance": "/api/v1/performance",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }
