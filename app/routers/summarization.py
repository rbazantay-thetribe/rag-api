from typing import Optional, List

from fastapi import APIRouter, Form
from pydantic import BaseModel, Field

from app.services.summarizer import summarize_text


router = APIRouter(tags=["summarization"], prefix="")


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Raw text to summarize")
    model: str = Field(
        default="plguillou/t5-base-fr-sum-cnndm",
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
    batch_size: int = Field(default=8, ge=1, description="Batch size for summarization calls")
    num_beams: int = Field(default=4, ge=1, description="Beam size for decoding")
    no_repeat_ngram_size: int = Field(default=3, ge=0, description="No-repeat n-gram size")
    length_penalty: float = Field(default=1.0, description="Length penalty for decoding")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty for decoding")
    prompt_prefix: Optional[str] = Field(
        default="Résume fidèlement en français, sans inventer de faits.",
        description="Optional instruction prefix prepended to each chunk",
    )


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
        batch_size=req.batch_size,
        num_beams=req.num_beams,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        length_penalty=req.length_penalty,
        repetition_penalty=req.repetition_penalty,
        prompt_prefix=req.prompt_prefix,
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
    model: str = Form("plguillou/t5-base-fr-sum-cnndm"),
    tokenizer: Optional[str] = Form("plguillou/t5-base-fr-sum-cnndm"),
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(100),
    min_length: int = Form(40),
    max_length: int = Form(120),
    do_sample: bool = Form(False),
    return_parts: bool = Form(False),
    batch_size: int = Form(8),
    num_beams: int = Form(4),
    no_repeat_ngram_size: int = Form(3),
    length_penalty: float = Form(1.0),
    repetition_penalty: float = Form(1.1),
    prompt_prefix: Optional[str] = Form("Résume fidèlement en français, sans inventer de faits."),
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
        batch_size=batch_size,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        prompt_prefix=prompt_prefix,
    )
    return SummarizeResponse(**result)




