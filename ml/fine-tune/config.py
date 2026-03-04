# config.py — single source of truth for all hyperparameters and paths

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
TRAIN_ADAPTER = "ai_detection"

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints/jina-v5-ai-detection"
FINAL_DIR = "./checkpoints/jina-v5-ai-detection-final"
DATASET_PATH = ".data/processed/fine-tune"  # saved by part_01, loaded by part_03

# ── Data caps ──────────────────────────────────────────────────────────────
HC3_CAP = 100_000
MAGE_CAP = 100_000
RAID_CAP = 50_000
PILE_CAP = 100_000
HARD_NEG_CAP = 100_000
HARD_NEG_RATIO = 0.30

# ── Text constraints ───────────────────────────────────────────────────────
MIN_TEXT_LEN = 150
MAX_TEXT_LEN = 1500

# ── LoRA ───────────────────────────────────────────────────────────────────
LORA_R = 16          # reduced from 32 — halves LoRA activation memory
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]  # fewer targets = less OOM pressure

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16  # effective batch = 16
MAX_SEQ_LENGTH = 256  # reduced from 512 — attention is O(n²), biggest VRAM win
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 50
TEST_SIZE = 0.05

# ── Matryoshka dims ────────────────────────────────────────────────────────
MATRYOSHKA_DIMS = [512, 256]  # dropped 1024 dim — saves forward pass memory

# ── Inference ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.05  # margin below which confidence is "low"
SEED = 42
