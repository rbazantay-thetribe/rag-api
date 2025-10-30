from fastapi import FastAPI

from app.routers.qdrant import router as qdrant_router
from app.routers.summarization import router as summarization_router

app = FastAPI(title="RAG API", version="0.1.0")



app.include_router(summarization_router, prefix="/summarize")
app.include_router(qdrant_router, prefix="/qdrant")
