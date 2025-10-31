from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.langchain_rag import answer_question


router = APIRouter(tags=["langchain"], prefix="")


class RagAskRequest(BaseModel):
    collection_name: str = Field(..., description="Qdrant collection to search")
    question: str = Field(..., description="User question")
    k: int = Field(default=5, ge=1, le=50, description="Top K documents")

    model: str = Field(default="qwen2.5:3b", description="Ollama model name")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model id used for query encoding",
    )
    options: Optional[Dict[str, Any]] = Field(default=None, description="Ollama options")


class Reference(BaseModel):
    id: Any
    score: float
    payload: Dict[str, Any]


class RagAskResponse(BaseModel):
    answer: str
    context: str
    references: List[Reference]
    model: str
    embedding_model: str


@router.post(
    "/rag/ask",
    summary="RAG answer using LangChain with Qdrant + Ollama",
    response_model=RagAskResponse,
)
def rag_ask(req: RagAskRequest) -> RagAskResponse:
    try:
        result = answer_question(
            collection_name=req.collection_name,
            question=req.question,
            ollama_model=req.model,
            embedding_model=req.embedding_model,
            top_k=req.k,
            ollama_options=req.options,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG error: {exc}")

    refs = [Reference(id=r.get("id"), score=r.get("score"), payload=r.get("payload", {})) for r in result.get("references", [])]
    return RagAskResponse(
        answer=result.get("answer", ""),
        context=result.get("context", ""),
        references=refs,
        model=result.get("model", req.model),
        embedding_model=result.get("embedding_model", req.embedding_model),
    )


