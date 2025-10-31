from fastapi import FastAPI

from app.routers.qdrant import router as qdrant_router
from app.routers.summarization import router as summarization_router
from app.routers.embeddings import router as embeddings_router
from app.routers.ollama import router as ollama_router
from app.routers.langchain import router as langchain_router

app = FastAPI(title="RAG API", version="0.1.0")



app.include_router(summarization_router, prefix="/summarize")
app.include_router(embeddings_router, prefix="/embeddings")
app.include_router(qdrant_router, prefix="/qdrant")
app.include_router(ollama_router, prefix="/ollama")
app.include_router(langchain_router, prefix="/langchain")

