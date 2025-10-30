import warnings
from typing import Dict, List, Optional, Tuple

from transformers import pipeline
from transformers.utils import logging as hf_logging


# Keep third-party warnings quiet for cleaner server logs
warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL.*",
)
try:
    from urllib3.exceptions import NotOpenSSLWarning  # type: ignore

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers",
)
hf_logging.set_verbosity_error()


_PIPELINE_CACHE: Dict[str, object] = {}


def _get_summarizer(
    model: str = "csebuetnlp/mT5_multilingual_XLSum",
    tokenizer: Optional[str] = None,
) -> object:
    """
    Lazily create and cache a HF summarization pipeline.

    The tokenizer defaults to the same identifier as the model if not provided.
    """
    cache_key = f"{model}|{tokenizer or model}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    summarizer = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer or model,
        use_fast=False,
    )
    _PIPELINE_CACHE[cache_key] = summarizer
    return summarizer


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    Simple character-based chunking with optional overlap.

    This avoids tokenization overhead and is robust enough for many inputs.
    API consumers can tune sizes to match their model limits.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text or ""
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def _summarize_single(
    summarizer: object,
    chunk: str,
    min_length: int,
    max_length: int,
    do_sample: bool = False,
) -> str:
    result = summarizer(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
    )
    return result[0]["summary_text"]


def summarize_text(
    text: str,
    *,
    model: str = "csebuetnlp/mT5_multilingual_XLSum",
    tokenizer: Optional[str] = None,
    chunk_size: int = 1200,  # characters
    chunk_overlap: int = 100,  # characters
    min_length: int = 40,
    max_length: int = 120,
    do_sample: bool = False,
    return_parts: bool = False,
) -> Dict[str, object]:
    """
    Summarize an arbitrary-length text with simple character chunking.

    Parameters are geared for API usage and can be adjusted per request.

    Returns a dict suitable for JSON serialization:
    {
      "model": str,
      "num_chunks": int,
      "parts": List[str],            # included when return_parts=True
      "summary": str                 # concatenation of parts with separators
    }
    """
    summarizer = _get_summarizer(model=model, tokenizer=tokenizer)
    chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    part_summaries: List[str] = []
    for chunk in chunks or [""]:
        if not chunk:
            continue
        part = _summarize_single(
            summarizer=summarizer,
            chunk=chunk,
            min_length=min_length,
            max_length=max_length,
            do_sample=do_sample,
        )
        part_summaries.append(part)

    combined = "\n\n".join(part_summaries) if part_summaries else ""

    response: Dict[str, object] = {
        "model": model,
        "num_chunks": len(chunks),
        "summary": combined,
    }
    if return_parts:
        response["parts"] = part_summaries
    return response


__all__ = [
    "summarize_text",
]


