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

## FastAPI Hello World (Local Docker build)
If you prefer running only the API container manually:

```bash
docker build -t rag-api:latest .
docker run --rm -p 8000:8000 rag-api:latest
curl http://localhost:8000/hello
```

