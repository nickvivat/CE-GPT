"""
Main FastAPI application for the CE RAG System.
Provides web API interface with Swagger documentation.
"""

import time
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from .router import router, initialize_rag_system
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting CE RAG API server...")
    try:
        # Initialize RAG system
        initialize_rag_system()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.warning("API will start but may not function properly")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CE RAG API server...")


# Create FastAPI application
app = FastAPI(
    title="CE RAG System API",
    description="""
    **Computer Engineering RAG System API**
    
    A multilingual Retrieval-Augmented Generation (RAG) system designed specifically for Computer Engineering courses at KMITL.
    
    ## Features
    
    - 🌐 **Multilingual Support**: Thai and English language processing
    - 🔍 **Intelligent Search**: Vector-based semantic search with reranking
    - 🤖 **LLM Integration**: Ollama-based query enhancement and response generation
    - 📊 **Performance Monitoring**: Real-time system metrics and performance tracking
    - 🛡️ **Error Handling**: Robust error handling with circuit breaker patterns
    - 💬 **Conversation Context**: Maintains chat history for better query understanding
    
    ## Quick Start
    
    1. **Health Check**: `GET /api/v1/health`
    2. **Search Courses**: `POST /api/v1/search`
    3. **Generate Response**: `POST /api/v1/generate`
    4. **System Status**: `GET /api/v1/status`
    
    ## Authentication
    
    Currently, this API does not require authentication for development purposes.
    For production deployment, consider implementing API key authentication.
    
    ## Rate Limiting
    
    Basic rate limiting is implemented to prevent abuse.
    Contact administrators for higher rate limits if needed.
    """,
    version="2.0.0",
    contact={
        "name": "CE RAG System Team",
        "email": "support@kmitl.ac.th",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


def custom_openapi():
    """Customize OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Customize OpenAPI schema
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Mount static files
app.mount("/static", StaticFiles(directory="server/static"), name="static")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": str(exc) if config.debug.debug else "An unexpected error occurred",
            "timestamp": time.time()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": time.time()
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Include API router
app.include_router(router)


# Root endpoint - serve the web interface
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the web interface."""
    try:
        with open("server/static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>CE RAG System</title></head>
            <body>
                <h1>🎓 CE RAG System</h1>
                <p>Web interface not found. Please check the server configuration.</p>
                <p><a href="/docs">API Documentation</a></p>
                <p><a href="/health">Health Check</a></p>
            </body>
        </html>
        """, status_code=404)


# Health check endpoint (root level)
@app.get("/health", tags=["Health"])
async def health_check():
    """Quick health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5500"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
