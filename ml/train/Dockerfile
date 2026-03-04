# ── Stage 1: base image ──────────────────────────────────────────────────────
# PyTorch 2.6+ with CUDA 12.4 — required for RTX 5090 (Blackwell, sm_120).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer independently
# of any source-file changes.
WORKDIR /app

RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore \
        # Embedding
        sentence-transformers>=3.0.0 \
        transformers>=4.40.0 \
        safetensors>=0.4.0 \
        huggingface-hub>=0.22.0 \
        # Datasets
        datasets>=2.18.0 \
        # Classifiers
        scikit-learn>=1.4.0 \
        xgboost>=2.0.0 \
        lightgbm>=4.3.0 \
        joblib>=1.3.0 \
        # Plotting
        matplotlib>=3.8.0 \
        # Utilities
        numpy>=1.26.0 \
        python-dotenv>=1.0.0

# ── Copy source ───────────────────────────────────────────────────────────────
# Copy the train scripts into /app/ml/train so that the relative
# parent-directory lookup for config._ROOT (parents[2]) resolves correctly:
#   /app/ml/train/config.py  →  ../../  →  /app  →  /app/checkpoints/
COPY ml/train/ /app/ml/train/

# Working directory for all RUN/CMD invocations
WORKDIR /app/ml/train

# ── Environment ───────────────────────────────────────────────────────────────
# Silence the HF symlinks warning inside Docker
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Redirect HF cache to a well-known path that can be volume-mounted
ENV HF_HOME=/app/.cache/huggingface
# Disable tokenizer parallelism warnings in subprocess context
ENV TOKENIZERS_PARALLELISM=false
# Suppress LightGBM OpenMP warnings
ENV OMP_NUM_THREADS=4

# ── Volumes ───────────────────────────────────────────────────────────────────
# Mount these at runtime so artifacts survive container restarts:
#   -v $(pwd)/checkpoints:/app/checkpoints           ← fine-tuned Jina model (read)
#   -v $(pwd)/ml/train/artifacts:/app/ml/train/artifacts  ← embeddings + classifiers (write)
#   -v ~/.cache/huggingface:/app/.cache              ← HF model cache
VOLUME ["/app/checkpoints", "/app/ml/train/artifacts", "/app/.cache"]

# ── Entry point ───────────────────────────────────────────────────────────────
# Runs the full classifier pipeline in order:
#   1. data_train.py           — download datasets, embed with fine-tuned Jina, save .npy
#   2. train_classifiers.py    — train LogReg / SVM / XGBoost / LightGBM, save .joblib
#   3. evaluating.py           — evaluate all classifiers, save metrics + plots
#   4. calibration_ensemble.py — calibrate, build weighted ensemble, save weights
#   5. select_model.py         — compare all candidates, promote best to production
#   6. threshold_tuning.py     — sweep thresholds on production model, save optimal_threshold.json
#
# To run a single step:
#   docker run ... python select_model.py
CMD ["bash", "-c", \
    "python data_train.py && \
     python train_classifiers.py && \
     python evaluating.py && \
     python calibration_ensemble.py && \
     python select_model.py && \
     python threshold_tuning.py"]
