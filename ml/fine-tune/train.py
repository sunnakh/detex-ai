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


class LoRATrainer(SentenceTransformerTrainer):
    """Overrides _save to manually extract and save LoRA adapter weights.

    JinaEmbeddingsV5Model is wrapped by PeftMixedModel (for non-standard
    architectures), which explicitly raises NotImplementedError on
    save_pretrained(). We bypass this by extracting only the LoRA weight
    keys from the state dict and writing them in the standard PEFT adapter
    format (adapter_model.safetensors + adapter_config.json).
    """

    def _save(self, output_dir: str | None = None, state_dict=None):
        import json
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        transformer_module = self.model._first_module()
        backbone = getattr(transformer_module, "auto_model", None) or getattr(
            transformer_module, "model"
        )

        # Extract only LoRA adapter weights from the full state dict
        lora_sd = {
            k: v.cpu()
            for k, v in backbone.state_dict().items()
            if "lora_" in k
        }

        # Save weights — safetensors preferred, fallback to .bin
        try:
            from safetensors.torch import save_file
            save_file(lora_sd, os.path.join(output_dir, "adapter_model.safetensors"))
        except ImportError:
            torch.save(lora_sd, os.path.join(output_dir, "adapter_model.bin"))

        # Save adapter config as JSON (same format as PEFT's own save)
        peft_cfg = backbone.peft_config[config.TRAIN_ADAPTER]
        cfg_dict = peft_cfg.to_dict()
        for k, v in cfg_dict.items():
            if hasattr(v, "value"):        # Enum → its primitive value
                cfg_dict[k] = v.value
            elif isinstance(v, set):       # set → list (e.g. target_modules)
                cfg_dict[k] = sorted(v)
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        # Save tokenizer alongside weights so the checkpoint is self-contained
        tokenizer = getattr(transformer_module, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

# ── HuggingFace auth ──────────────────────────────────────────────────────
_hf_token = os.getenv("HF_TOKEN")
if _hf_token:
    login(token=_hf_token, add_to_git_credential=False)
else:
    print("⚠  HF_TOKEN not set — skipping login (model weights are public)")

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
    warmup_steps=config.WARMUP_STEPS,
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
    metric_for_best_model="ai-detection-eval_cosine_ndcg@10",
    greater_is_better=True,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    logging_steps=config.LOGGING_STEPS,
)

# ── 6. Train ──────────────────────────────────────────────────────────────
trainer = LoRATrainer(
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
    raise

# ── 7. Save final LoRA weights ───────────────────────────────────────────────
# Reuse LoRATrainer._save() — backbone.save_pretrained() raises NotImplementedError
# for JinaEmbeddingsV5Model (PeftMixedModel limitation).
print("Saving final LoRA adapter...")
trainer._save(config.FINAL_DIR)
print(f"LoRA adapter saved to {config.FINAL_DIR}")
