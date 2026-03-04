# ── Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 ────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System deps ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy fine-tune source ─────────────────────────────────────────────────
COPY ml/fine-tune/ .

# ── HuggingFace cache dir (mount a volume here to avoid re-downloading) ───
ENV HF_HOME=/app/.cache/huggingface

# ── Output artifacts dir ──────────────────────────────────────────────────
RUN mkdir -p /app/artifacts /app/checkpoints

# ── Run: data pipeline → training ────────────────────────────────────────
CMD ["sh", "-c", "python data_pipeline.py && python train.py"]
