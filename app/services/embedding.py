import warnings
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging


# Suppress noisy third-party warnings for cleaner logs
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


def _get_feature_extractor(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokenizer: Optional[str] = None,
) -> object:
    """
    Lazily create and cache a HF feature-extraction pipeline.

    Defaults to a widely used sentence-transformers model. We do not depend on
    the sentence-transformers library; instead we mean-pool token embeddings.
    """
    cache_key = f"{model}|{tokenizer or model}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    extractor = pipeline(
        task="feature-extraction",
        model=model,
        tokenizer=tokenizer or model,
        return_tensors=False,
    )
    _PIPELINE_CACHE[cache_key] = extractor
    return extractor


def _mean_pool(features: List[List[List[float]]]) -> List[List[float]]:
    """
    Mean-pool token embeddings to produce sentence embeddings.

    Expects a nested list shaped as [batch, tokens, hidden]. Returns
    [batch, hidden].
    """
    embeddings: List[List[float]] = []
    for item in features:
        if not item:
            embeddings.append([])
            continue
        tensor = torch.tensor(item, dtype=torch.float32)
        # Collapse all dimensions except the last hidden dimension to handle
        # potential shapes like [tokens, hidden] or [1, tokens, hidden]
        if tensor.ndim <= 1:
            embeddings.append(tensor.tolist())
            continue
        reduce_dims = tuple(range(0, tensor.ndim - 1))
        sent = tensor.mean(dim=reduce_dims)  # -> [hidden]
        embeddings.append(sent.tolist())
    return embeddings


def embed_texts(
    texts: List[str],
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokenizer: Optional[str] = None,
    normalize: bool = True,
) -> Dict[str, object]:
    """
    Compute embeddings for a batch of texts.

    Returns a JSON-serializable dict:
    {
      "model": str,
      "count": int,
      "dim": int,
      "vectors": List[List[float]]
    }
    """
    extractor = _get_feature_extractor(model=model, tokenizer=tokenizer)
    texts = texts or []

    # HF pipeline supports batching internally; pass list of strings
    features = extractor(texts)
    vectors = _mean_pool(features)

    if normalize:
        normalized_vectors: List[List[float]] = []
        for vec in vectors:
            if not vec:
                normalized_vectors.append(vec)
                continue
            t = torch.tensor(vec, dtype=torch.float32)
            norm = torch.linalg.norm(t)
            if norm.item() == 0.0:
                normalized_vectors.append(vec)
            else:
                normalized_vectors.append((t / norm).tolist())
        vectors = normalized_vectors

    dim = len(vectors[0]) if vectors and vectors[0] else 0
    return {
        "model": model,
        "count": len(texts),
        "dim": dim,
        "vectors": vectors,
    }


__all__ = [
    "embed_texts",
]


