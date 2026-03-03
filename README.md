# Computer Engineering RAG System

A **multilingual Retrieval-Augmented Generation (RAG)** system for Computer Engineering courses at King Mongkut's Institute of Technology Ladkrabang (KMITL). It provides course and professor search, query enhancement, and contextual answers using retrieved data, in Thai and English.

## Features

### Core Capabilities
- **Multilingual**: Thai and English for queries and responses
- **Hybrid search**: Combined vector (Qdrant) + BM25 keyword search, merged via Reciprocal Rank Fusion (RRF)
- **Optional reranking**: Cross-encoder reranking for improved result relevance
- **LLM integration**: Ollama for query classification, enhancement, metadata generation, and response generation
- **Query classification**: 5-category classifier (enhanced, pass, conversational, external, abusing) with follow-up detection
- **Abusive query rejection**: Detects and blocks prompt injection attempts, profanity, and system exploit attempts
- **Metadata-driven search**: Auto-generated metadata tags (focus area, career track, query intent) to improve retrieval accuracy
- **Session-based chat**: Per-user sessions with chat history stored in a database; responses use history when a session is provided
- **Chat history compression**: Optional summarization of older messages with a configurable interval to keep context within token limits
- **Data**: Course, professor, curriculum, and study plan information from project data files

### Technical
- **REST API**: FastAPI with OpenAPI docs at `/docs` and `/redoc`
- **Rate limiting**: Request throttling via SlowAPI
- **Caching**: Embedding and chunk caching
- **Streaming**: Server-Sent Events for streaming responses
- **Docker**: Dockerfile and docker-compose for deployment
- **Error handling**: Circuit breaker pattern, retry with exponential backoff, and comprehensive error classification
- **Monitoring**: Health, status, and performance endpoints; structured logging; CSV performance logs
- **Account management**: User account model with database persistence
- **Testing**: Evaluation scripts, concurrent test runner, and test datasets

## Architecture

```
CE-GPT/
├── src/                    # Core RAG system
│   ├── core/              # RAG and session components
│   │   ├── rag.py         # System orchestration (hybrid search, RRF merge)
│   │   ├── embedder.py    # Text embedding generation
│   │   ├── reranker.py    # Cross-encoder result reranking
│   │   ├── query.py       # Query classification, enhancement & metadata generation
│   │   ├── generator.py   # Response generation
│   │   ├── llm_client.py  # LLM client interface (sync & async)
│   │   ├── vector_store.py # Qdrant vector store
│   │   ├── session_manager.py   # Session lifecycle
│   │   ├── chat_history.py      # Per-session chat history
│   │   ├── history_compressor.py # Chat history summarization
│   │   └── account.py     # User account model & management
│   ├── preprocess/        # Data processing
│   │   ├── data_processor.py    # Unified data processor
│   │   ├── course_processor.py  # Course-specific processing
│   │   └── professor_processor.py # Professor-specific processing
│   └── utils/             # Utilities
│       ├── config.py      # Configuration
│       ├── logger.py      # Structured logging
│       ├── database.py    # DB connection and schema (SQLAlchemy)
│       ├── error_handler.py     # Circuit breaker, retry, error decorators
│       ├── performance_monitor.py # In-memory performance tracking
│       └── performance_logger.py  # CSV performance logging
├── server/                # API server
│   ├── app.py            # FastAPI application
│   ├── router.py         # API endpoints & rate limiting
│   └── models.py         # Pydantic request/response models
├── prompt/                # LLM prompt templates
│   ├── query_classifier.md    # 5-category query classification prompt
│   ├── query_enhancer.md      # Query term expansion prompt
│   ├── metadata_generator.md  # Metadata tag generation prompt
│   ├── system_prompt.md       # Response generation system prompt
│   └── archived/              # Archived/deprecated prompts
├── data/                  # Data storage
│   ├── raw/               # Raw source data (course, professor, curriculum, studyplan)
│   └── processed/         # Processed chunks
├── tests/                 # Test suite
│   ├── run_testcase.py    # Test case runner
│   ├── test_concurrent.py # Concurrent load testing
│   ├── evaluate.py        # Evaluation metrics
│   ├── dataset/           # Test datasets
│   └── result/            # Test results
├── cache/                 # Caching (embeddings, chunks)
├── logs/                  # Performance logs (CSV)
├── init_database.py       # DB initialization
├── migrate_chat_history.py      # Chat history migration script
├── migrate_compression_summary.py # Compression summary migration
├── qdrant.py              # Qdrant utility script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build
└── docker-compose.yml     # Docker Compose configuration
```

## Installation

### Prerequisites
- Python 3.11 or higher
- Ollama (e.g. Gemma model) for query classification, enhancement, and generation
- Qdrant (local or cloud) for vector search
- A database (SQLite or PostgreSQL); connection string via `DATABASE_URL` for sessions, chat history, and accounts
- 8GB+ RAM and sufficient disk space for models

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
   
   Copy `.env.example` to `.env`, then set values for your environment (especially `DATABASE_URL`, `QDRANT_*`, and `OLLAMA_*`).

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
   The server creates database tables on first run. To create or verify them manually: `python init_database.py`.

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

## Query Processing Pipeline

Every user query passes through a multi-stage pipeline:

1. **Classification** — The query is classified into one of five categories using the LLM:
   - `enhanced` — Needs semantic expansion (abbreviations, broad topics, qualitative queries)
   - `pass` — Database-ready (specific course codes, explicit titles, direct attribute queries)
   - `conversational` — Chat/social interaction, no search needed
   - `external` — Out-of-scope, non-academic topics
   - `abusing` — Prompt injection, profanity, or system exploit attempts → **rejected with warning**

2. **Follow-up detection** — If conversation context is available, the classifier detects follow-up questions (pronoun references, implicit context) and appends relevant course codes from history.

3. **Enhancement** (if classified as `enhanced`) — Query terms are semantically expanded in parallel with metadata generation.

4. **Metadata generation** — Tags (focus area, career track, research area) and query intent are generated to filter search results.

5. **Hybrid search** — Vector similarity search (Qdrant) combined with BM25 keyword search, merged via Reciprocal Rank Fusion.

6. **Reranking** (optional) — Cross-encoder reranking of search results for improved relevance.

7. **Response generation** — LLM generates a contextual response using retrieved chunks and chat history.

## API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs — request/response schemas, example payloads, and try-it-out
- **ReDoc**: http://localhost:8000/redoc

All generate endpoints require `user_id` in the request body; the API root is `GET /api/v1/`.

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
`POST /api/v1/generate` — Requires `user_id`. Optionally pass `session_id` to use and update chat history for that session. When sessions are enabled, a session is auto-created per user if none is provided.

```http
POST /api/v1/generate
Content-Type: application/json

{
  "query": "What are the prerequisites for calculus?",
  "user_id": "user-123",
  "session_id": "optional-session-uuid",
  "language": "en",
  "top_k": 5,
  "use_reranking": true,
  "include_sources": true,
  "stream": false
}
```

If the query is classified as `abusing`, the API returns a rejection response with a warning message instead of performing any search or generation.

#### Streaming Response
`POST /api/v1/generate/stream` — Same request shape; returns Server-Sent Events (SSE) with response chunks.

```http
POST /api/v1/generate/stream
Content-Type: application/json

{
  "query": "Explain machine learning concepts",
  "user_id": "user-123",
  "language": "en",
  "use_reranking": true
}
```

#### Sessions and Chat History
```http
POST   /api/v1/sessions              # Create session (user_id, optional metadata)
GET    /api/v1/sessions/{session_id}  # Get session
PUT    /api/v1/sessions/{session_id}  # Update session / extend TTL
DELETE /api/v1/sessions/{session_id}  # Delete session
GET    /api/v1/sessions/{session_id}/history   # Get chat history (limit, offset)
DELETE /api/v1/sessions/{session_id}/history   # Clear chat history for session
```

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

#### Cache and Context
```http
POST /api/v1/cache/clear         # Clear embedding/chunk cache
POST /api/v1/conversation/clear   # Clear in-memory conversation context
```

## Configuration

Copy `.env.example` to `.env` and adjust values. All options and defaults are defined in `src/utils/config.py` (environment variables override them).

## Performance

- **Monitoring**: `/api/v1/performance` and `/api/v1/status` for metrics and status; CSV logs in `logs/` for embedding search, reranking, query enhancement, and generation
- **Caching**: Embedding and chunk cache to reduce repeated work
- **Chat history**: Compression (summarize older messages at an interval) keeps context within the configured token window
- **Hybrid search**: BM25 + vector search improves retrieval accuracy over pure vector search

## Testing

The `tests/` directory contains evaluation and testing utilities:

- **`run_testcase.py`** — Run test cases from dataset files and evaluate responses
- **`test_concurrent.py`** — Concurrent load testing with multiple simultaneous users
- **`evaluate.py`** — Evaluation metrics for response quality

Test datasets are stored in `tests/dataset/` and results are written to `tests/result/`.

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

4. **API connection issues**
   - `curl http://localhost:8000/health`. On Windows: `netstat -ano | findstr :8000`; on Linux/Mac: `netstat -tulpn | grep :8000`

5. **Database / sessions**
   - Set `DATABASE_URL`. Run `init_database.py` to create tables. Sessions, chat history, and accounts require a working DB connection.

6. **Vector store empty**
   - Ensure data exists in `data/processed/`. RAG loads data on startup; verify the Qdrant collection has data.

7. **CORS**
   - Configure CORS in `server/app.py` (`allow_origins`) if a frontend needs to call the API.

### Debug Mode
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export API_RELOAD=true
uvicorn server.app:app --reload --log-level debug
```

## Data Processing

The system processes four types of data:
- **Course Data**: Course information from `data/raw/course_detail.json`
- **Professor Data**: Professor information from `data/raw/professor_detail.json`
- **Curriculum Data**: Degree requirements and graduation criteria
- **Study Plan Data**: Semester-by-semester course planning

Processed data is stored in `data/processed/` and automatically loaded into the vector store on system initialization.

## Performance Logging

Performance metrics are logged to CSV files in the `logs/` directory:
- `embedding_search_performance.csv`
- `reranking_performance.csv`
- `query_enhancement_performance.csv`
- `response_generation_performance.csv`
- `overall_rag_performance.csv`

---

**Version**: 1.1.0  
**Last Updated**: March 2026  
**Contact**: nickvivat@hotmail.com
