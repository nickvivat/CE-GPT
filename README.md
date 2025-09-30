# Computer Engineering RAG System

A **Multilingual Retrieval-Augmented Generation (RAG)** system designed specifically for Computer Engineering courses at KMITL. This system provides intelligent course search, query enhancement, and contextual responses in both Thai and English.

## ✨ Features

### Core Capabilities
- **Multilingual Processing**: Native support for Thai and English language processing
- **Semantic Retrieval**: Advanced vector-based retrieval with intelligent reranking
- **LLM Integration**: Ollama-based query enhancement and response generation
- **Conversation Context**: Maintains chat history for improved query understanding
- **Performance Monitoring**: Real-time system metrics and comprehensive logging

### Technical Features
- **RESTful API**: FastAPI-based API with automatic OpenAPI documentation
- **Error Handling**: Robust error handling with circuit breaker patterns
- **Caching**: Intelligent caching for improved performance
- **Streaming**: Real-time streaming responses for enhanced user experience
- **Scalability**: Designed for production deployment with monitoring

## 🏗️ Architecture

```
CE RAG/
├── src/                    # Core RAG system
│   ├── core/              # Main RAG components
│   │   ├── rag.py         # System orchestration
│   │   ├── embedder.py    # Text embedding generation
│   │   ├── reranker.py    # Result reranking
│   │   ├── query.py       # Query enhancement
│   │   ├── generator.py   # Response generation
│   │   └── vector_store.py # Vector database management
│   ├── preprocess/        # Data processing
│   │   ├── data_processor.py
│   │   ├── course_processor.py
│   │   └── professor_processor.py
│   └── utils/             # Utilities
│       ├── config.py      # Configuration management
│       ├── logger.py      # Logging system
│       ├── error_handler.py
│       └── performance_monitor.py
├── server/                # API server
│   ├── app.py            # FastAPI application
│   ├── router.py         # API endpoints
│   └── models.py         # Pydantic models
└── data/                 # Data storage
    ├── raw/              # Raw course data
    └── processed/        # Processed data

```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Ollama with Gemma3 model
- 8GB+ RAM recommended
- 2GB+ disk space for models

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CE-RAG
   ```

2. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Start Ollama service**
   ```bash
   ollama serve
   ```

4. **Download required model**
   ```bash
   ollama pull gemma3:4b-it-qat
   ```

5. **Start the API server**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📚 API Documentation

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
Detailed system status including performance metrics and configuration.


#### Response Generation
```http
POST /api/v1/generate
Content-Type: application/json

{
  "query": "What are the prerequisites for calculus?",
  "language": "en",
  "use_reranking": true,
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

#### Performance Metrics
```http
GET /api/v1/performance
```
Retrieve detailed performance statistics and system metrics.

#### Cache Management
```http
POST /api/v1/cache/clear
POST /api/v1/conversation/clear
```
Clear system cache and conversation context.

## 💻 Usage Examples

### Python Client
```python
import requests

# Generate AI response
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "query": "What are the career prospects in computer engineering?",
    "language": "en",
    "use_reranking": True
})

ai_response = response.json()
print(ai_response['response'])
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health


# Generate response
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain algorithms", "language": "en"}'
```

### CLI Demo
```bash
python examples/demo.py
```

## ⚙️ Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
DEBUG=false

# Model Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b-it-qat
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Processing Configuration
BATCH_SIZE=32
MAX_WORKERS=4
TOP_K=50
TOP_K_RERANK=20
```

### Configuration File
The system uses `src/utils/config.py` for centralized configuration management with environment variable overrides.

## 🔧 Development

### Code Quality
- **Formatting**: Black code formatter
- **Linting**: Flake8 for code quality
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings and comments

### Development Commands
```bash
# Format code
black src/ examples/ server/

# Lint code
flake8 src/ examples/ server/

# Run tests
pytest

# Start development server
uvicorn server:app --reload --log-level debug
```

### Project Structure
- **Modular Design**: Clean separation of concerns
- **Error Handling**: Comprehensive error handling with retry logic
- **Logging**: Structured logging with performance monitoring
- **Testing**: Unit tests and integration tests

## 📊 Performance

### Monitoring
- Real-time performance metrics
- System resource monitoring (CPU, memory, disk)
- API response time tracking
- Error rate monitoring

### Optimization
- Intelligent caching strategies
- Batch processing for embeddings
- Circuit breaker patterns for fault tolerance
- Memory-efficient data processing

### Scalability
- Horizontal scaling support
- Database connection pooling
- Async processing capabilities
- Resource usage optimization

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama service
   ollama serve
   ```

2. **Model Loading Errors**
   ```bash
   # Verify model availability
   ollama list
   
   # Pull required model
   ollama pull gemma3:4b-it-qat
   ```

3. **API Connection Issues**
   ```bash
   # Check API server status
   curl http://localhost:8000/health
   
   # Verify port availability
   netstat -tulpn | grep :8000
   ```

4. **CORS Issues**
   - Ensure frontend is configured to connect to the API server
   - Check CORS settings in `server/app.py`

### Debug Mode
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export API_RELOAD=true
uvicorn server:app --reload --log-level debug
```


---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Maintainer**: CE RAG System Team
**Contact**: nickvivat@hotmail.com