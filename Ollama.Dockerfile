FROM ollama/ollama:latest

ARG OLLAMA_DEFAULT_MODEL=qwen2.5:3b

# Install curl (ensure availability during build)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Warm the image with the desired model at build time
RUN (ollama serve & pid=$!; \
    echo "Waiting for Ollama to be ready..."; \
    until curl -sSf http://localhost:11434/api/tags >/dev/null; do sleep 1; done; \
    echo "Pulling model: ${OLLAMA_DEFAULT_MODEL}"; \
    ollama pull ${OLLAMA_DEFAULT_MODEL}; \
    kill $pid || true)


