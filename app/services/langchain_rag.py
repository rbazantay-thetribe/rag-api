from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient

# LangChain core and community
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama import ChatOllama

from app.services.embedding import embed_texts


def _get_qdrant_client() -> QdrantClient:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    return QdrantClient(url=qdrant_url)


def _search_qdrant(
    collection_name: str,
    query: str,
    *,
    k: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Perform a similarity search in Qdrant and return a list of payload dicts.

    Expects payloads to contain a "text" field; falls back to joining key-values.
    """
    client = _get_qdrant_client()

    emb = embed_texts([query], model=embedding_model)
    vectors = emb.get("vectors", [])
    if not vectors or not vectors[0]:
        return []
    query_vector: List[float] = vectors[0]

    scored = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for p in scored:
        payload = getattr(p, "payload", {}) or {}
        # Normalize into a doc dict
        text = payload.get("text")
        if text is None:
            # fallback: stringify payload
            try:
                text = "\n".join(f"{k}: {v}" for k, v in payload.items())
            except Exception:
                text = str(payload)
        results.append({
            "id": p.id,
            "score": p.score,
            "text": text,
            "payload": payload,
        })
    return results


def _default_prompt() -> ChatPromptTemplate:
    system = (
        "You are a helpful and precise assistant. Use the provided context to answer the user question.\n"
        "- If the answer is not in the context, say you don't know.\n"
        "- Be concise and factual."
    )
    template = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ])
    return template


def _format_context(docs: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[{i}] {d.get('text', '')}")
    return "\n\n".join(parts)


def build_rag_chain(
    *,
    collection_name: str,
    ollama_model: str = "qwen2.5:3b",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    ollama_options: Optional[Dict[str, Any]] = None,
):
    """
    Build an LCEL RAG chain using Qdrant for retrieval and Ollama for generation.
    """
    retriever = RunnableLambda(
        lambda q: _search_qdrant(
            collection_name=collection_name,
            query=q,
            k=top_k,
            embedding_model=embedding_model,
        )
    )

    llm = ChatOllama(model=ollama_model, base_url=os.environ.get("OLLAMA_HOST", "http://ollama:11434"),
                     temperature=(ollama_options or {}).get("temperature"))

    prompt = _default_prompt()
    to_context = RunnableLambda(_format_context)

    chain = (
        RunnableParallel(
            question=lambda x: x,
            docs=retriever,
        )
        | RunnableLambda(lambda x: {"question": x["question"], "context": to_context.invoke(x["docs"])})
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def answer_question(
    *,
    collection_name: str,
    question: str,
    ollama_model: str = "qwen2.5:3b",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    ollama_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the RAG chain and return the answer along with references.
    """
    docs = _search_qdrant(
        collection_name=collection_name,
        query=question,
        k=top_k,
        embedding_model=embedding_model,
    )
    context_text = _format_context(docs)

    chain = build_rag_chain(
        collection_name=collection_name,
        ollama_model=ollama_model,
        embedding_model=embedding_model,
        top_k=top_k,
        ollama_options=ollama_options,
    )
    answer = chain.invoke(question)
    return {
        "answer": answer,
        "context": context_text,
        "references": [{"id": d.get("id"), "score": d.get("score"), "payload": d.get("payload")} for d in docs],
        "model": ollama_model,
        "embedding_model": embedding_model,
    }


__all__ = [
    "build_rag_chain",
    "answer_question",
]


