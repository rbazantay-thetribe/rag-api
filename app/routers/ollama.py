from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.ollama_client import (
    list_models as svc_list_models,
    pull_model as svc_pull_model,
    generate_text as svc_generate_text,
    chat_completion as svc_chat_completion,
    embedding as svc_embedding,
)


router = APIRouter(tags=["ollama"], prefix="")


class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]


@router.get(
    "/models",
    summary="List Ollama models",
    response_model=ModelListResponse,
)
def list_models() -> ModelListResponse:
    return ModelListResponse(models=svc_list_models())


class PullModelRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'llama3.1:8b' or 'mistral' ")
    stream: bool = Field(default=False, description="Stream pull progress server-side")


class PullModelResponse(BaseModel):
    status: str = Field(default="ok")
    detail: Dict[str, Any] = {}


@router.post(
    "/models/pull",
    summary="Pull a model",
    response_model=PullModelResponse,
)
def pull_model(req: PullModelRequest) -> PullModelResponse:
    try:
        detail = svc_pull_model(model=req.model, stream=req.stream)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama pull error: {exc}")
    return PullModelResponse(status="ok", detail=detail)


class GenerateRequest(BaseModel):
    model: str = Field(default="qwen2.5:3b", description="Model name, e.g. 'llama3.1:8b' or 'mistral' ")
    prompt: str = Field(default="Hello, world!", description="Prompt to generate text")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Ollama generation options")


class GenerateResponse(BaseModel):
    text: str


@router.post(
    "/generate",
    summary="Text generation with Ollama models",
    response_model=GenerateResponse,
)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        text = svc_generate_text(model=req.model, prompt=req.prompt, options=req.options)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama generate error: {exc}")
    return GenerateResponse(text=text)


class ChatMessage(BaseModel):
    role: str = Field(..., description="one of 'system' | 'user' | 'assistant'")
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    options: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    text: str


@router.post(
    "/chat",
    summary="Chat completion with Ollama models",
    response_model=ChatResponse,
)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        # Convert pydantic models to dict list
        messages: List[Dict[str, str]] = [m.model_dump() for m in req.messages]
        text = svc_chat_completion(model=req.model, messages=messages, options=req.options)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama chat error: {exc}")
    return ChatResponse(text=text)


class EmbeddingRequest(BaseModel):
    model: str
    input: str = Field(..., description="Input text to embed")


class EmbeddingResponse(BaseModel):
    model: str
    vector: List[float]


@router.post(
    "/embeddings",
    summary="Get embeddings from Ollama models",
    response_model=EmbeddingResponse,
)
def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    try:
        vec = svc_embedding(model=req.model, input_text=req.input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama embeddings error: {exc}")
    return EmbeddingResponse(model=req.model, vector=vec)


