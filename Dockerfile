FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    HF_HOME=/models/hf

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app

# Create and own model cache directory
RUN mkdir -p /models/hf

# Pre-download summarization model weights into the image cache
# Using the default model configured in app.services.summarizer
RUN python - << 'PY'
from transformers import pipeline
model_id = "plguillou/t5-base-fr-sum-cnndm"
pipe = pipeline(
    task="summarization",
    model=model_id,
    tokenizer=model_id,
    use_fast=False,
)
_ = pipe("Texte de test pour initialiser le cache.")
print("[Warmup] Hugging Face model cached")
PY

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


