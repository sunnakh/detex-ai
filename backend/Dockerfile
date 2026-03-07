# ── Backend – detex.ai FastAPI (SVM + Jina embeddings) ───────────────────────
FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ML + API deps (torch CPU-only to keep image size reasonable)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "scikit-learn==1.8.0" \
    joblib \
    transformers \
    peft \
    xgboost \
    "torch>=2.0.0" \
    python-multipart \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download the embedding model at build time so startup is instant
ENV HF_HOME=/cache/huggingface
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v5-text-small', trust_remote_code=True); \
    AutoModel.from_pretrained('jinaai/jina-embeddings-v5-text-small', trust_remote_code=True)"

# Copy application code (context is project root per docker-compose)
COPY backend/app/ ./app/

# Copy the trained model artifact
COPY backend/artifacts/jina-v5-finetuned/best_clf.joblib ./artifacts/jina-v5-finetuned/best_clf.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
