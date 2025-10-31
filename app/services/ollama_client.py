import os
from typing import Any, Dict, List, Optional

from ollama import Client


def get_ollama_client() -> Client:
    host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    return Client(host=host)


def list_models() -> List[Dict[str, Any]]:
    client = get_ollama_client()
    resp = client.list()
    # resp is typically { 'models': [ { 'name': '...', 'size': ..., ... }, ... ] }
    return resp.get("models", [])


def pull_model(model: str, stream: bool = False) -> Dict[str, Any]:
    client = get_ollama_client()
    if stream:
        # Return the last progress snapshot as a summary when streaming
        last: Dict[str, Any] = {}
        for chunk in client.pull(model=model, stream=True):
            last = chunk
        return last or {"status": "done", "model": model}
    return client.pull(model=model, stream=False)


def generate_text(model: str = "qwen2.5:3b", prompt: str = "Hello, world!", options: Optional[Dict[str, Any]] = None) -> str:
    client = get_ollama_client()
    resp = client.generate(model=model, prompt=prompt, options=options or {})
    return resp.get("response", "")


def chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    options: Optional[Dict[str, Any]] = None,
) -> str:
    client = get_ollama_client()
    resp = client.chat(model=model, messages=messages, options=options or {})
    message = resp.get("message", {})
    return message.get("content", "")


def embedding(model: str, input_text: str) -> List[float]:
    client = get_ollama_client()
    resp = client.embeddings(model=model, prompt=input_text)
    return resp.get("embedding", [])


