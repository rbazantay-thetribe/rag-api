import warnings
from typing import Dict, List, Optional, Tuple

from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers.utils import logging as hf_logging
import torch


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


_MODEL_CACHE: Dict[str, Tuple[T5Tokenizer, T5ForConditionalGeneration]] = {}
_PIPELINE_CACHE: Dict[str, object] = {}


def _is_t5_model(model_id: str) -> bool:
    """Check if model ID indicates a T5 model."""
    return "t5" in model_id.lower()


def _get_t5_model(
    model: str = "plguillou/t5-base-fr-sum-cnndm",
    tokenizer: Optional[str] = None,
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """
    Lazily create and cache T5 model and tokenizer.
    
    For T5 models, we use direct model access for better control.
    """
    tokenizer_id = tokenizer or model
    cache_key = f"{model}|{tokenizer_id}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    tok = T5Tokenizer.from_pretrained(tokenizer_id)
    mod = T5ForConditionalGeneration.from_pretrained(model)
    _MODEL_CACHE[cache_key] = (tok, mod)
    return tok, mod


def _get_summarizer(
    model: str = "plguillou/t5-base-fr-sum-cnndm",
    tokenizer: Optional[str] = None,
) -> object:
    """
    Lazily create and cache a HF summarization pipeline.

    For non-T5 models, falls back to pipeline.
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
    Sentence-aware chunking with optional character overlap.

    We split on sentence boundaries to avoid cutting mid-sentence,
    which reduces hallucinations for abstractive models.
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

    # Simple sentence splitting for FR/EN: ., !, ? followed by space/newline or end
    sentences: List[str] = []
    buff: List[str] = []
    for ch in text:
        buff.append(ch)
        if ch in ".!?":
            sentences.append("".join(buff).strip())
            buff = []
    if buff:
        sentences.append("".join(buff).strip())

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len + (1 if current_len else 0) <= chunk_size:
            if current_len:
                current.append(" ")
                current_len += 1
            current.append(sent)
            current_len += sent_len
        else:
            if current:
                chunks.append("".join(current))
            # start new chunk with current sentence
            current = [sent]
            current_len = sent_len
    if current:
        chunks.append("".join(current))

    # optional simple overlap in characters by copying tail of previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        with_overlap: List[str] = []
        prev_tail = ""
        for i, chnk in enumerate(chunks):
            if i == 0:
                with_overlap.append(chnk)
            else:
                prefix = prev_tail[-chunk_overlap:]
                with_overlap.append((prefix + chnk)[: chunk_size])
            prev_tail = chnk
        chunks = with_overlap

    return chunks


def _summarize_single_t5(
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    chunk: str,
    min_length: int,
    max_length: int,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.1,
) -> str:
    """Summarize using T5 with explicit 'résumer: ' prefix."""
    # T5 requires explicit prefix for summarization task
    input_text = f"résumer: {chunk}"
    
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # T5 input limit
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
            do_sample=False,
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def _summarize_single(
    summarizer: object,
    chunk: str,
    min_length: int,
    max_length: int,
    do_sample: bool = False,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.1,
) -> str:
    result = summarizer(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
    )
    return result[0]["summary_text"]


def _summarize_batch_t5(
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    chunks: List[str],
    min_length: int,
    max_length: int,
    batch_size: int = 4,  # T5 typically needs smaller batches
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Summarize multiple chunks using T5 with proper prefix.
    
    Process in smaller batches due to T5 memory constraints.
    """
    if not chunks:
        return []
    
    results = []
    # T5 requires "résumer: " prefix for each input
    prefixed_chunks = [f"résumer: {chunk}" for chunk in chunks]
    
    # Process in batches
    for i in range(0, len(prefixed_chunks), batch_size):
        batch = prefixed_chunks[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                early_stopping=True,
                do_sample=False,
            )
        
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_summaries)
    
    return results


def _summarize_batch(
    summarizer: object,
    chunks: List[str],
    min_length: int,
    max_length: int,
    do_sample: bool = False,
    batch_size: int = 8,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Summarize multiple chunks in batches for better performance.
    
    The pipeline's batch processing is more efficient than processing
    chunks one by one.
    """
    if not chunks:
        return []
    
    results = []
    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_results = summarizer(
            batch,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
        )
        results.extend([item["summary_text"] for item in batch_results])
    
    return results


def summarize_text(
    text: str,
    *,
    model: str = "plguillou/t5-base-fr-sum-cnndm",
    tokenizer: Optional[str] = None,
    chunk_size: int = 1200,  # characters
    chunk_overlap: int = 100,  # characters
    min_length: int = 40,
    max_length: int = 120,
    do_sample: bool = False,
    return_parts: bool = False,
    batch_size: int = 8,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.1,
    prompt_prefix: Optional[str] = None,
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
    chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]

    if not valid_chunks:
        combined = ""
        part_summaries = []
    elif _is_t5_model(model):
        # Use direct T5 model access with proper prefix
        tokenizer_obj, model_obj = _get_t5_model(model=model, tokenizer=tokenizer)
        
        # Adjust batch size for T5 (smaller due to memory)
        t5_batch_size = min(batch_size, 4)
        
        part_summaries = _summarize_batch_t5(
            tokenizer=tokenizer_obj,
            model=model_obj,
            chunks=valid_chunks,
            min_length=min_length,
            max_length=max_length,
            batch_size=t5_batch_size,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        combined = "\n\n".join(part_summaries) if part_summaries else ""
    else:
        # Fallback to pipeline for non-T5 models
        summarizer = _get_summarizer(model=model, tokenizer=tokenizer)
        
        # Filter out empty chunks and optionally prefix prompt
        if prompt_prefix:
            prefixed_chunks = [f"{prompt_prefix.strip()}\n\n{chunk}" for chunk in valid_chunks]
        else:
            prefixed_chunks = valid_chunks
        
        part_summaries = _summarize_batch(
            summarizer=summarizer,
            chunks=prefixed_chunks,
            min_length=min_length,
            max_length=max_length,
            do_sample=do_sample,
            batch_size=batch_size,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        combined = "\n\n".join(part_summaries) if part_summaries else ""

    response: Dict[str, object] = {
        "model": model,
        "num_chunks": len(chunks),
        "summary": combined,
    }
    if return_parts:
        response["parts"] = part_summaries if valid_chunks else []
    return response


__all__ = [
    "summarize_text",
]


