import os
import types
import torch
import torch.nn.functional as F
import gc
from pathlib import Path
from typing import Dict, Optional

# Project root is one level above this file (ml/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_from_disk


def train_jina(
    dataset_path=str(PROJECT_ROOT / "data/processed/fine_tune_triplets"),
    output_dir=str(PROJECT_ROOT / "artifacts/jina-v5-finetuned"),
    epochs=3,
    batch_size=32,
    lr=2e-4,
    unfreeze_last_n_layers=4,
):
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    print("Loading Jina Model v5 Small...")
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v5-text-small",
        trust_remote_code=True,
        model_kwargs={
            "dtype": torch.float32,  # float32 for MPS compatibility
            "default_task": "retrieval",
        },
    )

    # ── Patch Jina v5's forward to allow gradient computation ─────────────────
    # Jina v5's custom Transformer.forward() hard-codes self.model.eval() and
    # torch.no_grad() on every call, making backprop impossible. We replace it
    # with a version that only suppresses gradients during inference.
    def _jina_forward_with_grad(
        self,
        features: Dict[str, torch.Tensor],
        task: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if task is None:
            task = self.default_task
        self.model.set_adapter(task)
        device = self.model.device

        def _compute(batch):
            outputs = self.model(**batch)
            hidden = outputs.last_hidden_state
            mask = batch.get("attention_mask")
            if mask is None:
                pooled = hidden[:, -1]
            else:
                seq_len = mask.sum(dim=1) - 1
                pooled = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device), seq_len
                ]
            if truncate_dim is not None:
                pooled = pooled[:, :truncate_dim]
            return F.normalize(pooled, p=2, dim=-1)

        batch = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}
        if self.training:
            self.model.train()
            features["sentence_embedding"] = _compute(batch)
        else:
            self.model.eval()
            with torch.no_grad():
                features["sentence_embedding"] = _compute(batch)
        return features

    model[0].forward = types.MethodType(_jina_forward_with_grad, model[0])

    # ── Freeze all parameters, then selectively unfreeze ──────────────────────
    # Jina v5 already has built-in multi-task LoRA adapters baked into its
    # architecture. Applying PEFT on top creates nested LoRA which breaks the
    # computation graph. Instead, we unfreeze only the retrieval LoRA adapter
    # weights (lora_A / lora_B) in the last N transformer layers. This is
    # parameter-efficient (~2M params vs 74M for full layers) and keeps
    # activation memory manageable on MPS.
    for param in model.parameters():
        param.requires_grad = False

    # Find the transformer layers inside the Jina backbone
    jina_module = model[0].model
    backbone = None
    for attr in ["model", "encoder", "transformer", "backbone"]:
        if hasattr(jina_module, attr):
            candidate = getattr(jina_module, attr)
            if hasattr(candidate, "layers"):
                backbone = candidate
                break

    if backbone is None or not hasattr(backbone, "layers"):
        raise RuntimeError(
            "Could not locate transformer layers inside JinaEmbeddingsV5Model. "
            "Inspect model[0].model to find the correct attribute."
        )

    layers = backbone.layers
    n_layers = len(layers)
    unfreeze_from = max(0, n_layers - unfreeze_last_n_layers)
    print(
        f"Model has {n_layers} transformer layers. "
        f"Unfreezing retrieval LoRA in last {unfreeze_last_n_layers} (layers {unfreeze_from}–{n_layers - 1})."
    )

    # Unfreeze ONLY the retrieval LoRA adapter params in those layers
    # Parameter names look like: <layer>.self_attn.q_proj.lora_A.retrieval.weight
    for name, param in model.named_parameters():
        # Check the layer index from the name
        for lora_part in ["lora_A.retrieval", "lora_B.retrieval"]:
            if lora_part in name:
                # Extract layer index: "...layers.<idx>..."
                parts = name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    layer_idx = -1
                if layer_idx >= unfreeze_from:
                    param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    model.train()

    # Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Loss
    loss = MultipleNegativesRankingLoss(model)

    # Evaluator
    eval_queries = {str(i): eval_dataset[i]["anchor"] for i in range(len(eval_dataset))}
    eval_corpus = {
        str(i): eval_dataset[i]["positive"] for i in range(len(eval_dataset))
    }
    eval_relevant = {str(i): {str(i)} for i in range(len(eval_dataset))}
    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant,
        name="ai-detection-eval",
    )

    # Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # effective batch = batch_size * 4
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True if device == "cuda" else False,
        gradient_checkpointing=False,  # keep off; custom forward patch handles MPS memory
        dataloader_pin_memory=False,  # required for MPS; harmless on CUDA
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    print("Starting Training...")
    trainer.train()

    # Save final model
    print("Saving fine-tuned model...")
    model.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Model saved to {os.path.join(output_dir, 'final')}")


if __name__ == "__main__":
    train_jina(epochs=1, batch_size=4)
