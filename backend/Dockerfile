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
    scikit-learn \
    joblib \
    sentence-transformers \
    "torch>=2.0.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY app/ ./app/

# Copy the trained model artifact (relative to backend/ context)
COPY artifacts/jina-v5-finetuned/best_clf.joblib ./artifacts/jina-v5-finetuned/best_clf.joblib

# HuggingFace cache (embedding model weights downloaded on first run)
ENV HF_HOME=/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
