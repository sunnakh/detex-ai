# config.py — single source of truth for all hyperparameters and paths

import os

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_ID = "jinaai/jina-embeddings-v5-text-small"

# ── Paths ──────────────────────────────────────────────────────────────────
# Resolve paths relative to the project root (two levels up from this file),
# so they work regardless of which directory the script is run from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL_DIR = os.path.join(_ROOT, "checkpoints", "jina-v5-ai-detection-final")

# ── Text constraints ───────────────────────────────────────────────────────
MIN_TEXT_LEN = 150
MAX_TEXT_LEN = 1500

# ── Data ───────────────────────────────────────────────────────────────────
TEST_SIZE = 0.05

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
