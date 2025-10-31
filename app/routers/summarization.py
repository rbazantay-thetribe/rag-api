from typing import Optional, List

from fastapi import APIRouter, Form
from pydantic import BaseModel, Field

from app.services.summarizer import summarize_text


router = APIRouter(tags=["summarization"], prefix="")


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Raw text to summarize")
    model: str = Field(
        default="csebuetnlp/mT5_multilingual_XLSum",
        description="HuggingFace model id",
    )
    tokenizer: Optional[str] = Field(
        default=None,
        description="Optional tokenizer id (defaults to model)",
    )

    chunk_size: int = Field(default=1200, ge=1, description="Chunk size in characters")
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Chunk overlap in characters (must be < chunk_size)",
    )

    min_length: int = Field(default=40, ge=1)
    max_length: int = Field(default=120, ge=1)
    do_sample: bool = Field(default=False)
    return_parts: bool = Field(default=False)


class SummarizeResponse(BaseModel):
    model: str
    num_chunks: int
    summary: str
    parts: Optional[List[str]] = None


@router.post(
    "/",
    response_model=SummarizeResponse,
    summary="Summarize text",
    description="Summarize a long text using simple character chunking and a HF model.",
)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    if req.chunk_overlap >= req.chunk_size:
        return SummarizeResponse(
            model=req.model,
            num_chunks=0,
            summary="",
            parts=None,
        )

    result = summarize_text(
        text=req.text,
        model=req.model,
        tokenizer=req.tokenizer,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        min_length=req.min_length,
        max_length=req.max_length,
        do_sample=req.do_sample,
        return_parts=req.return_parts,
    )
    return SummarizeResponse(**result)


@router.post(
    "/form",
    response_model=SummarizeResponse,
    summary="Summarize text (form fields)",
    description=(
        "Same as /summarize but accepts application/x-www-form-urlencoded form fields, "
        "so Swagger shows individual inputs instead of a JSON body."
    ),
)
def summarize_form(
    text: str = Form(..., description="Raw text to summarize"),
    model: str = Form("csebuetnlp/mT5_multilingual_XLSum"),
    tokenizer: Optional[str] = Form(None),
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(100),
    min_length: int = Form(40),
    max_length: int = Form(120),
    do_sample: bool = Form(False),
    return_parts: bool = Form(False),
) -> SummarizeResponse:
    if chunk_overlap >= chunk_size:
        return SummarizeResponse(
            model=model,
            num_chunks=0,
            summary="",
            parts=None,
        )

    result = summarize_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_length=min_length,
        max_length=max_length,
        do_sample=do_sample,
        return_parts=return_parts,
    )
    return SummarizeResponse(**result)



