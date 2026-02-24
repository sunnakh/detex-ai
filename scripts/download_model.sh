#!/bin/bash

echo "Downloading jina-embeddings-v5-text-small..."

HF_ENDPOINT=https://hf-mirror.com uv run huggingface-cli download \
    jinaai/jina-embeddings-v5-text-small \
    --local-dir ./models/jina-model \
    --local-dir-use-symlinks False

echo "Model saved to ./models/jina-model"