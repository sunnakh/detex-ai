# ── Stage 1: base image ──────────────────────────────────────────────────────
# PyTorch 2.4+ required by newer Hugging Face transformers.
# Use -devel variant so nvcc is available if any extension needs compilation.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer independently
# of any source-file changes.
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore \
        # Core ML
        torch>=2.0.0 \
        transformers>=4.40.0 \
        sentence-transformers>=3.0.0 \
        peft>=0.10.0 \
        accelerate>=0.29.0 \
        safetensors>=0.4.0 \
        # Data
        datasets>=2.18.0 \
        huggingface-hub>=0.22.0 \
        # Utilities
        python-dotenv>=1.0.0 \
        psutil>=5.9.0

# ── Copy source ───────────────────────────────────────────────────────────────
# Copy the fine-tune scripts into /app/ml/fine-tune so that the relative
# parent-directory lookup for .env (parents[2]) resolves correctly:
#   /app/ml/fine-tune/train.py  →  parents[2] = /app  →  /app/.env
COPY ml/fine-tune/ /app/ml/fine-tune/

# Working directory for all RUN/CMD invocations
WORKDIR /app/ml/fine-tune

# ── Environment ───────────────────────────────────────────────────────────────
# Reduce CUDA memory fragmentation (same as the in-script default)
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
# Silence the HF symlinks warning inside Docker
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Redirect HF cache to a well-known path that can be volume-mounted
ENV HF_HOME=/app/.cache/huggingface
# Disable tokenizer parallelism warnings in subprocess context
ENV TOKENIZERS_PARALLELISM=false

# ── Volumes ───────────────────────────────────────────────────────────────────
# Mount these at runtime so data and checkpoints survive container restarts:
#   -v $(pwd)/ml/fine-tune/.data:/app/ml/fine-tune/.data  ← processed dataset
#   -v $(pwd)/checkpoints:/app/checkpoints                ← training checkpoints
#   -v ~/.cache/huggingface:/app/.cache                   ← HF model cache
VOLUME ["/app/ml/fine-tune/.data", "/app/checkpoints", "/app/.cache"]

# ── Entry point ───────────────────────────────────────────────────────────────
# Default: run training only (assumes dataset already exists via volume mount).
# To run the full pipeline: docker run ... bash -c "python data_pipeline.py && python train.py"
CMD ["python", "train.py"]
