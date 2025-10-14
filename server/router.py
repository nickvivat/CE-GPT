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
    PerformanceMetrics, HealthCheck, ErrorResponse
)
from src.core.rag import RAGSystem
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.performance_monitor import performance_monitor

logger = get_logger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["CE RAG System"])

# Global RAG system instance
rag_system: RAGSystem = None
system_start_time = time.time()


def get_rag_system() -> RAGSystem:
    """Dependency to get the RAG system instance."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


def initialize_rag_system():
    """Initialize the RAG system globally."""
    global rag_system
    try:
        logger.info("Initializing RAG system for API...")
        # RAG system will auto-load data and build vector index
        rag_system = RAGSystem(use_reranker=True, use_query_enhancement=True, auto_load_data=True)
        
        # Verify that data was loaded successfully
        if not rag_system.chunks or len(rag_system.chunks) == 0:
            raise Exception("No data loaded by RAG system")
        
        logger.info("RAG system initialized successfully for API")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_system = None
        raise e


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
async def generate_response(request: GenerateRequest):
    """Generate an AI response based on the query and retrieved sources."""
    try:
        start_time = time.time()
        
        # Get RAG system
        rag_system = get_rag_system()
        
        # Determine language (same logic as search endpoint)
        language = None
        if request.language != "auto":
            language = request.language
        
        # First, search for relevant courses to get sources
        search_results = await rag_system.search(
            query=request.query,
            top_k=request.top_k,
            language=language,
            use_reranking=request.use_reranking
        )
        
        logger.info(f"Found {len(search_results)} search results for query: {request.query}")
        
        # Generate response using the search results
        response = await rag_system.generate_response(
            query=request.query,
            top_k=request.top_k,
            language=request.language,
            use_reranking=request.use_reranking,
            stream=request.stream,
            search_results=search_results,
        )
        
        generation_time = (time.time() - start_time) * 1000
        
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
            sources=sources,
            language_detected=rag_system._detect_language(request.query),
            generation_time_ms=generation_time,
            total_sources=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream", summary="Generate AI Response with Real-time Streaming")
async def generate_response_stream(request: GenerateRequest):
    """Generate an AI response with real-time streaming using Server-Sent Events."""
    try:
        # Get RAG system
        rag_system = get_rag_system()
        
        # Determine language
        language = None
        if request.language != "auto":
            language = request.language
        
        # Search for relevant courses to get sources
        search_results = await rag_system.search(
            query=request.query,
            top_k=request.top_k,
            language=language,
            use_reranking=request.use_reranking
        )
        
        logger.info(f"Found {len(search_results)} search results for streaming query: {request.query}")
        
        async def generate_stream():
            """Generator function for streaming response chunks."""
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
                    language=request.language,
                    use_reranking=request.use_reranking,
                    search_results=search_results
                ):
                    if chunk:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
                completion_data = {
                    'type': 'complete', 
                    'message': 'Generation complete',
                    'language_detected': rag_system._detect_language(request.query),
                    'total_sources': len(search_results)
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
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
            "performance": "/api/v1/performance",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }
