# Summarizer
test : python ./scripts/summarizer.py ./files/input.txt ./files/resume.txt
 
## Run with Docker Compose (Qdrant + RAG API)

This project provides a Docker Compose setup to run a local Qdrant instance and the FastAPI RAG API.

### Requirements
- Docker
- Docker Compose

### Start services
```bash
docker compose up -d --build
```

### Stop services
```bash
docker compose down
```

### Data persistence
Qdrant data is persisted in a local Docker volume named `qdrant_storage`.

### Endpoints
- REST API: http://localhost:6333
- gRPC: localhost:6334

### RAG API
- Base URL: http://localhost:8000
- Health/Test: `GET /hello`

```bash
curl http://localhost:8000/hello
```

### Embeddings
- Batch embeddings: `POST /embeddings`

Request body (JSON):

```json
{
  "texts": ["hello world", "bonjour le monde"],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "tokenizer": null,
  "normalize": true
}
```

Example:

```bash
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["hello world", "bonjour le monde"],
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "normalize": true
  }'
```

## FastAPI Hello World (Local Docker build)
If you prefer running only the API container manually:

```bash
docker build -t rag-api:latest .
docker run --rm -p 8000:8000 rag-api:latest
curl http://localhost:8000/hello
```

