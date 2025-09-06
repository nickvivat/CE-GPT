# 🎓 Computer Engineering RAG System

A **Multilingual Retrieval-Augmented Generation (RAG)** system designed specifically for Computer Engineering courses at KMITL. This system provides intelligent course search, query enhancement, and contextual responses in both Thai and English.

## ✨ Features

- **🌐 Multilingual Support**: Thai and English language processing
- **🔍 Intelligent Search**: Vector-based semantic search with reranking
- **🤖 LLM Integration**: Ollama-based query enhancement and response generation
- **📊 Performance Monitoring**: Real-time system metrics and performance tracking
- **🛡️ Error Handling**: Robust error handling with circuit breaker patterns
- **💬 Conversation Context**: Maintains chat history for better query understanding
- **🌐 Web API**: FastAPI-based REST API with Swagger documentation (in `server/` folder)
- **📱 Web Interface**: Modern, responsive web UI for easy interaction (in `server/static/`)

## 🏗️ Architecture

```
src/
├── core/           # Core RAG components
│   ├── rag.py     # Main system orchestration
│   ├── embedder.py # Text embedding generation
│   ├── reranker.py # Result reranking
│   ├── query.py   # Query enhancement
│   └── generator.py # Response generation
├── utils/          # Utility modules
│   ├── config.py  # Configuration management
│   ├── logger.py  # Logging utilities
│   ├── error_handler.py # Error handling & retry logic
│   └── performance_monitor.py # Performance tracking
└── data/           # Data processing
    └── data_processor.py # Course data processing

server/              # Web API components
├── app.py          # FastAPI application
├── router.py       # API endpoints
├── models.py       # Pydantic models
└── static/         # Web interface
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama with Gemma3 model installed and running
- Required Python packages

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CE-RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama (in a separate terminal):**
   ```bash
   ollama serve
   ```

4. **Pull the required model:**
   ```bash
   ollama pull gemma3:4b-it-qat
   ```

### Run the demo

#### CLI Demo
```bash
python examples/demo.py
```

#### Web API & Interface
```bash
uvicorn server:app --host 0.0.0.0 --port 5500 --reload
```

Then open your browser to:
- 🌐 **Web Interface**: http://localhost:5500/
- 📚 **API Documentation**: http://localhost:5500/docs

## 🌐 Web API

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 5500 --reload
   ```

3. **Access the web interface:**
   - 🌐 **Web UI**: http://localhost:5500/
   - 📚 **API Docs**: http://localhost:5500/docs
   - 💚 **Health Check**: http://localhost:5500/health

### API Endpoints

- `GET /api/v1/health` - System health check
- `GET /api/v1/status` - Detailed system status
- `POST /api/v1/search` - Search for courses
- `POST /api/v1/generate` - Generate AI responses
- `GET /api/v1/performance` - Performance metrics
- `POST /api/v1/cache/clear` - Clear system cache
- `POST /api/v1/conversation/clear` - Clear conversation context

### Testing the API

You can test the API endpoints using:
- **Web Interface**: http://localhost:5500/ (recommended)
- **API Documentation**: http://localhost:5500/docs (interactive)
- **curl commands** or **Postman** for direct API testing

## 📖 Usage

### Basic Search
```python
from src.core.rag import RAGSystem

# Initialize system
rag = RAGSystem(use_reranker=True, use_query_enhancement=True)

# Load data
rag.load_and_process_data("data/raw/course_detail.json")

# Search for courses
results = rag.search("calculus courses", top_k=5)
```

### API Usage Example
```python
import requests

# Search for courses
response = requests.post("http://localhost:5500/api/v1/search", json={
    "query": "calculus courses",
    "top_k": 5,
    "language": "auto",
    "use_reranking": True
})

results = response.json()
print(f"Found {results['total_results']} courses")
```

### Interactive Demo
```bash
python examples/demo.py
```

Available commands:
- `help` - Show available commands
- `status` - System status and statistics
- `performance` - Performance metrics
- `clear` - Clear conversation context
- `quit` - Exit the demo

## ⚙️ Configuration

Configuration is managed through environment variables and `src/utils/config.py`:

```python
# API Configuration
API_HOST=0.0.0.0
API_PORT=5500
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

## 🔧 Development

### Code Style
- Use Black for code formatting
- Follow PEP 8 guidelines
- Add type hints for all functions
- Include comprehensive docstrings

### Testing
```bash
# Run tests
pytest

# Test API endpoints
python test_api.py

# Format code
black src/ examples/

# Lint code
flake8 src/ examples/
```

## 📊 Performance

The system includes comprehensive performance monitoring:
- Operation timing and memory usage
- System resource monitoring (CPU, memory, disk)
- Performance statistics and export capabilities
- Circuit breaker patterns for fault tolerance
- Real-time API performance metrics

## 🐛 Troubleshooting

### Common Issues
1. **Ollama Connection Failed**: Ensure Ollama server is running
2. **Model Loading Errors**: Check model availability and disk space
3. **Performance Issues**: Monitor system resources and adjust batch sizes
4. **API Connection Errors**: Check if the API server is running on the correct port

### Debug Mode
Enable debug logging by setting environment variables:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export API_RELOAD=true
```

### Debug Mode
Enable debug logging by setting log level in configuration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- KMITL Computer Engineering Department
- Open source community for RAG and ML libraries
- Ollama team for local LLM support
- FastAPI team for the excellent web framework

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Maintainer**: RAG System Team