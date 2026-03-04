import os
import gc
from pathlib import Path

# Reduce CUDA memory fragmentation (recommended in OOM error message)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import config

from datasets import DatasetDict, load_from_disk
from huggingface_hub import login
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from model import load_model
from peft import PeftModel

# ── HuggingFace auth ──────────────────────────────────────────────────────
login(os.getenv("HF_TOKEN"))

print("Load dataset from disk: ...")

if not os.path.exists(config.DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{config.DATASET_PATH}'.\n"
        "Run part_01_data_pipeline.py first."
    )

split: DatasetDict = load_from_disk(config.DATASET_PATH)
train_ds = split["train"]
eval_ds = split["test"]

if len(train_ds) == 0:
    raise RuntimeError("Train split is empty. Re-run part_01_data_pipeline.py.")

print(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

# loading model

model = load_model()

# Enforce sequence length cap — without this the tokenizer uses the model's
# default max (8192 for Jina v5), which destroys VRAM at training time.
model.max_seq_length = config.MAX_SEQ_LENGTH

torch.cuda.empty_cache()
gc.collect()

# ── 3. Loss ───────────────────────────────────────────────────────────────
# MNR treats all other samples in the batch as negatives — very sample-efficient.
# Matryoshka wraps MNR so the model trains at all truncation levels simultaneously.
base_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=config.MATRYOSHKA_DIMS)
print(f"Loss: MatryoshkaLoss over dims {config.MATRYOSHKA_DIMS}")

# ── 4. Evaluator ──────────────────────────────────────────────────────────
# IR evaluator measures MAP, NDCG, MRR — more meaningful than raw eval_loss
# for an embedding model. Anchor → should retrieve its own positive.
# to_dict() reads all columns in one pass — much faster than row-by-row indexing
eval_dict = eval_ds.to_dict()
eval_keys = [str(i) for i in range(len(eval_ds))]
eval_queries = dict(zip(eval_keys, eval_dict["anchor"]))
eval_corpus = dict(zip(eval_keys, eval_dict["positive"]))
eval_relevant = {k: {k} for k in eval_keys}

evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant,
    name="ai-detection-eval",
)

# ── 5. Training arguments ─────────────────────────────────────────────────
Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(config.FINAL_DIR).mkdir(parents=True, exist_ok=True)

args = SentenceTransformerTrainingArguments(
    output_dir=config.CHECKPOINT_DIR,
    seed=config.SEED,
    num_train_epochs=config.EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRAD_ACCUM,
    learning_rate=config.LEARNING_RATE,
    warmup_ratio=config.WARMUP_RATIO,
    weight_decay=config.WEIGHT_DECAY,
    max_grad_norm=config.MAX_GRAD_NORM,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=0,  # avoids pickling error with patched forward
    eval_strategy="steps",
    eval_steps=config.EVAL_STEPS,
    save_strategy="steps",
    save_steps=config.SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="ai-detection-eval_ndcg@10",
    greater_is_better=True,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    logging_steps=config.LOGGING_STEPS,
)

# ── 6. Train ──────────────────────────────────────────────────────────────
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    loss=loss,
    evaluator=evaluator,
)

print("Starting training...")
try:
    trainer.train()
except Exception:
    print("Training interrupted — saving checkpoint...")
    trainer.save_state()
    raise

# ── 7. Save LoRA weights ─────────────────────────────────────────────────────
print("Saving LoRA weights...")
transformer_module = model._first_module()
backbone = getattr(transformer_module, "auto_model", None) or getattr(
    transformer_module, "model"
)
backbone.save_pretrained(config.CHECKPOINT_DIR)
print(f"LoRA adapter saved to {config.CHECKPOINT_DIR}")
