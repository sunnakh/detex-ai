# config.py — single source of truth for all hyperparameters and paths
import os

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
TRAIN_ADAPTER = "ai_detection"

# ── Paths ──────────────────────────────────────────────────────────────────
# Resolve paths relative to the project root (two levels up from this file),
# so they are saved alongside backend/frontend.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints", "jina-v5-ai-detection")
FINAL_DIR = os.path.join(_ROOT, "checkpoints", "jina-v5-ai-detection-final")
DATASET_PATH = "../..data/processed/fine-tune"  # saved by part_01, loaded by part_03

# ── Data caps ──────────────────────────────────────────────────────────────
HC3_CAP = 100_000
MAGE_CAP = 100_000
RAID_CAP = 50_000
PILE_CAP = 100_000
HARD_NEG_CAP = 100_000  # 60% of 50K base
HARD_NEG_RATIO = 0.30  # final dataset: 50K base + 15K hard neg = ~65K

# ── Text constraints ───────────────────────────────────────────────────────
MIN_TEXT_LEN = 150
MAX_TEXT_LEN = 1500

# ── LoRA ───────────────────────────────────────────────────────────────────
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ── Training ───────────────────────────────────────────────────────────────
# Target: ~6 hrs on RTX 5090 (32GB VRAM)
EPOCHS = 1
BATCH_SIZE = 32     # 8× larger batches — saturates the 5090 properly
GRAD_ACCUM = 1      # effective batch = 32 (no accumulation needed)
MAX_SEQ_LENGTH = 256  # caps tokenizer — Jina v5 defaults to 8192 which causes OOM
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100  # ~10% of ~1,000 total steps
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 50
TEST_SIZE = 0.05

# ── Matryoshka dims ────────────────────────────────────────────────────────
MATRYOSHKA_DIMS = [1024, 512, 256]

# ── Inference ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.05  # margin below which confidence is "low"
SEED = 42
