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
DATASET_PATH = ".data/processed/fine-tune"  # saved by part_01, loaded by part_03

# ── Data caps ──────────────────────────────────────────────────────────────
HC3_CAP = 500
MAGE_CAP = 500
RAID_CAP = 500
PILE_CAP = 500
HARD_NEG_CAP = 600  # 60% of 50K base
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
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 4  # effective batch = 16
MAX_SEQ_LENGTH = 256  # caps tokenizer — Jina v5 defaults to 8192 which causes OOM
LEARNING_RATE = 2e-4
WARMUP_STEPS = (
    312  # ~10% of ~3,125 total steps (50K samples, 1 epoch, batch 4, grad_accum 4)
)
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
