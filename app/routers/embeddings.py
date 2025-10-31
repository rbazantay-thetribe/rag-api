from typing import List, Optional

from fastapi import APIRouter, Form
from pydantic import BaseModel, Field

from app.services.embedding import embed_texts


router = APIRouter(tags=["embeddings"], prefix="")


class EmbeddingsRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model id",
    )
    tokenizer: Optional[str] = Field(
        default=None,
        description="Optional tokenizer id (defaults to model)",
    )
    normalize: bool = Field(default=True, description="L2-normalize embeddings")


class EmbeddingsResponse(BaseModel):
    model: str
    count: int
    dim: int
    vectors: List[List[float]]


@router.post(
    "/",
    response_model=EmbeddingsResponse,
    summary="Embed a batch of texts",
    description="Compute sentence embeddings for a list of texts.",
)
def embed(req: EmbeddingsRequest) -> EmbeddingsResponse:
    result = embed_texts(
        texts=req.texts,
        model=req.model,
        tokenizer=req.tokenizer,
        normalize=req.normalize,
    )
    return EmbeddingsResponse(**result)


@router.post(
    "/form",
    response_model=EmbeddingsResponse,
    summary="Embed texts (form fields)",
    description=(
        "Same as /embeddings but accepts application/x-www-form-urlencoded form fields. "
        "Provide multiple 'texts' fields to embed several inputs."
    ),
)
def embed_form(
    texts: List[str] = Form(...),
    model: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    tokenizer: Optional[str] = Form(None),
    normalize: bool = Form(True),
) -> EmbeddingsResponse:
    result = embed_texts(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        normalize=normalize,
    )
    return EmbeddingsResponse(**result)




