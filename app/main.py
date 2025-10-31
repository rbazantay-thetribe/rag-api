from fastapi import FastAPI

from app.routers.qdrant import router as qdrant_router
from app.routers.summarization import router as summarization_router
from app.routers.embeddings import router as embeddings_router
from app.routers.ollama import router as ollama_router
from app.routers.langchain import router as langchain_router
from app.services.summarizer import _get_t5_model, _is_t5_model

app = FastAPI(title="RAG API", version="0.1.0")



app.include_router(summarization_router, prefix="/summarize")
app.include_router(embeddings_router, prefix="/embeddings")
app.include_router(qdrant_router, prefix="/qdrant")
app.include_router(ollama_router, prefix="/ollama")
app.include_router(langchain_router, prefix="/langchain")


@app.on_event("startup")
async def warmup_models() -> None:
    # Preload default summarizer to download and cache model/tokenizer at startup
    try:
        default_model = "plguillou/t5-base-fr-sum-cnndm"
        if _is_t5_model(default_model):
            _get_t5_model(model=default_model)
        else:
            from app.services.summarizer import _get_summarizer
            _get_summarizer(model=default_model)
    except Exception:
        # Do not crash the app on warmup failure; requests may still trigger lazy load
        pass

