# Computer Engineering RAG System

A **Multilingual Retrieval-Augmented Generation (RAG)** system designed specifically for Computer Engineering courses at KMITL. This system provides intelligent course search, query enhancement, and contextual responses in both Thai and English.

## Features

### Core Capabilities
- **Multilingual Processing**: Native support for Thai and English language processing
- **Semantic Retrieval**: Advanced vector-based retrieval with intelligent reranking using Qdrant vector database
- **LLM Integration**: Ollama-based query enhancement and response generation
- **Conversation Context**: Maintains chat history for improved query understanding
- **Performance Monitoring**: Real-time system metrics and comprehensive logging
- **Data Support**: Handles both course and professor information

### Technical Features
- **RESTful API**: FastAPI-based API with automatic OpenAPI documentation
- **Error Handling**: Robust error handling with circuit breaker patterns
- **Caching**: Intelligent caching for embeddings and chunks
- **Streaming**: Real-time streaming responses using Server-Sent Events
- **Docker Support**: Containerized deployment with Docker and docker-compose
- **Scalability**: Designed for production deployment with monitoring

## Architecture

```
CE-GPT/
├── src/                    # Core RAG system
│   ├── core/              # Main RAG components
│   │   ├── rag.py         # System orchestration
│   │   ├── embedder.py    # Text embedding generation
│   │   ├── reranker.py    # Result reranking
│   │   ├── query.py       # Query enhancement
│   │   ├── generator.py   # Response generation
│   │   ├── llm_client.py  # LLM client interface
│   │   └── vector_store.py # Qdrant vector database management
│   ├── preprocess/        # Data processing
│   │   ├── data_processor.py
│   │   ├── course_processor.py
│   │   └── professor_processor.py
│   └── utils/             # Utilities
│       ├── config.py      # Configuration management
│       ├── logger.py      # Logging system
│       ├── error_handler.py
│       ├── performance_monitor.py
│       └── performance_logger.py
├── server/                # API server
│   ├── app.py            # FastAPI application
│   ├── router.py         # API endpoints
│   └── models.py         # Pydantic models
├── data/                 # Data storage
│   ├── raw/              # Raw course and professor data
│   └── processed/        # Processed data
├── cache/                # Caching directory
│   ├── embeddings/       # Cached embeddings
│   └── chunks/           # Cached chunks
├── logs/                 # Performance logs
├── prompt/               # Prompt templates
└── docker-compose.yml    # Docker Compose configuration
```

## Installation

### Prerequisites
- Python 3.11 or higher
- Ollama with Gemma3 model
- Qdrant vector database
- 8GB+ RAM recommended
- 2GB+ disk space for models

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CE-GPT
   ```

2. **Install dependencies**
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

3. **Set up Qdrant**
   
   For local Qdrant instance:
   ```bash
   # Using Docker
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
   
   Or use Qdrant Cloud and set the `QDRANT_URL` environment variable.

4. **Configure environment variables**
   
   Create a `.env` file or set environment variables:
   ```bash
   # Qdrant Configuration
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION_NAME=CE-GPT
   QDRANT_API_KEY=  # Optional, for Qdrant Cloud
   
   # Ollama Configuration
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=gemma3:4b-it-qat
   
   # Model Configuration
   EMBEDDING_MODEL=google/embeddinggemma-300m
   RERANKER_MODEL=BAAI/bge-reranker-v2-m3
   ```

5. **Start Ollama service and download model**
   
   Start the Ollama service (required for model downloads):
   ```bash
   ollama serve
   ```
   
   In a new terminal, download the required model (Ollama must be running):
   ```bash
   ollama pull gemma3:4b-it-qat
   ```
   
   Note: The `ollama pull` command requires the Ollama service to be running. You can run `ollama serve` in one terminal and `ollama pull` in another, or run `ollama serve` in the background.

6. **Start the API server**
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build Docker image manually**
   ```bash
   docker build -t ce-gpt .
   docker run -p 8000:8000 --env-file .env ce-gpt
   ```

   Note: When using Docker, ensure Ollama is accessible. For local Ollama, use `http://host.docker.internal:11434` as the `OLLAMA_URL`.

## API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### System Health
```http
GET /health
GET /api/v1/health
```
Quick health check endpoints for monitoring.

#### System Status
```http
GET /api/v1/status
```
Detailed system status including performance metrics, chunk counts, and component status.

#### Response Generation
```http
POST /api/v1/generate
Content-Type: application/json

{
  "query": "What are the prerequisites for calculus?",
  "language": "en",
  "top_k": 5,
  "use_reranking": true,
  "include_sources": true,
  "stream": false
}
```

#### Streaming Response
```http
POST /api/v1/generate/stream
Content-Type: application/json

{
  "query": "Explain machine learning concepts",
  "language": "en",
  "use_reranking": true
}
```
Returns Server-Sent Events (SSE) stream with real-time response chunks.

#### Performance Metrics
```http
GET /api/v1/performance
```
Retrieve detailed performance statistics and system metrics.

#### Performance Export
```http
POST /api/v1/performance/export
```
Export performance data to JSON file.

#### Cache Management
```http
POST /api/v1/cache/clear
POST /api/v1/conversation/clear
```
Clear system cache and conversation context.

## Usage Examples

### Python Client
```python
import requests

# Generate AI response
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "query": "What are the career prospects in computer engineering?",
    "language": "en",
    "top_k": 5,
    "use_reranking": True,
    "include_sources": True
})

result = response.json()
print(result['response'])
print(f"Sources: {result['total_sources']}")
```

### Streaming Response
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/v1/generate/stream",
    json={"query": "Explain algorithms", "language": "en"},
    stream=True
)

for line in response.iter_lines():
    if line:
        data = line.decode('utf-8')
        if data.startswith('data: '):
            event = json.loads(data[6:])
            if event.get('type') == 'chunk':
                print(event.get('content', ''), end='', flush=True)
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/api/v1/status

# Generate response
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain algorithms",
    "language": "en",
    "top_k": 5,
    "use_reranking": true
  }'
```

## Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_WORKERS=1
DEBUG=false
LOG_LEVEL=INFO

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=CE-GPT
QDRANT_API_KEY=  # Optional, for Qdrant Cloud

# Model Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b-it-qat
EMBEDDING_MODEL=google/embeddinggemma-300m
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Processing Configuration
BATCH_SIZE=32
MAX_WORKERS=4
TOP_K=50
TOP_K_RERANK=20
SIMILARITY_THRESHOLD=0.1
RERANK_THRESHOLD=0.5

# Cache Configuration
CACHE_EMBEDDINGS_DIR=cache/embeddings
CACHE_CHUNKS_DIR=cache/chunks
CACHE_MAX_SIZE=1024
CACHE_TTL=3600
```

### Configuration File
The system uses `src/utils/config.py` for centralized configuration management with environment variable overrides. All configuration classes support loading from environment variables.

## Development

### Code Quality
- **Formatting**: Black code formatter
- **Linting**: Flake8 for code quality
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings and comments

### Development Commands
```bash
# Format code
black src/ server/

# Lint code
flake8 src/ server/

# Run tests
pytest

# Start development server
uvicorn server.app:app --reload --log-level debug
```

### Project Structure
- **Modular Design**: Clean separation of concerns with core, preprocessing, and utility modules
- **Error Handling**: Comprehensive error handling with retry logic and circuit breakers
- **Logging**: Structured logging with performance monitoring and CSV export
- **Vector Store**: Qdrant-based vector database with support for course and professor data
- **Testing**: Unit tests and integration tests (tests directory)

## Performance

### Monitoring
- Real-time performance metrics via `/api/v1/performance` endpoint
- System resource monitoring (CPU, memory, disk)
- API response time tracking
- Error rate monitoring
- CSV logging for embedding search, reranking, query enhancement, and response generation

### Optimization
- Intelligent caching strategies for embeddings and chunks
- Batch processing for embeddings
- Circuit breaker patterns for fault tolerance
- Memory-efficient data processing
- Async processing capabilities

### Scalability
- Horizontal scaling support with Qdrant
- Database connection pooling
- Async processing capabilities
- Resource usage optimization

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Start Qdrant with Docker
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

2. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama service
   ollama serve
   ```

3. **Model Loading Errors**
   ```bash
   # Verify model availability
   ollama list
   
   # Pull required model
   ollama pull gemma3:4b-it-qat
   ```

4. **API Connection Issues**
   ```bash
   # Check API server status
   curl http://localhost:8000/health
   
   # Verify port availability (Linux/Mac)
   netstat -tulpn | grep :8000
   
   # Verify port availability (Windows)
   netstat -ano | findstr :8000
   ```

5. **Vector Store Empty**
   - Ensure data files exist in `data/processed/`
   - Check that RAG system auto-loads data on initialization
   - Verify Qdrant collection exists and contains data

6. **CORS Issues**
   - Ensure frontend is configured to connect to the API server
   - Check CORS settings in `server/app.py`
   - Configure `allow_origins` appropriately for production

### Debug Mode
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export API_RELOAD=true
uvicorn server.app:app --reload --log-level debug
```

## Data Processing

The system processes two types of data:
- **Course Data**: Course information from `data/raw/course_detail.json`
- **Professor Data**: Professor information from `data/raw/professor_detail.json`

Processed data is stored in `data/processed/` and automatically loaded into the vector store on system initialization.

## Performance Logging

Performance metrics are logged to CSV files in the `logs/` directory:
- `embedding_search_performance.csv`
- `reranking_performance.csv`
- `query_enhancement_performance.csv`
- `response_generation_performance.csv`
- `overall_rag_performance.csv`

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Maintainer**: CE RAG System Team  
**Contact**: nickvivat@hotmail.com
