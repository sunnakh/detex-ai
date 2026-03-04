import types
import torch
import torch.nn.functional as F
import config

from peft import LoraConfig
from sentence_transformers import SentenceTransformer


def load_model() -> SentenceTransformer:
    """Full pipeline: load base → patch forward → apply LoRA → freeze base weights"""

    def _backbone(module):
        """Compat helper: sentence-transformers ≥2.7 uses auto_model, older uses model."""
        return getattr(module, "auto_model", None) or getattr(module, "model")

    # ──  Load base model ────────────────────────────────────────────────
    print("Loading Jina v5 model: ...")
    model = SentenceTransformer(
        config.MODEL_ID,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16, "default_task": "classification"},
    )

    # ──  Patch forward ──────────────────────────────────────────────────
    def _training_forward(self, features, task=None, truncate_dim=None):
        bb = _backbone(self)
        bb.set_adapter([config.TRAIN_ADAPTER])
        device = next(bb.parameters()).device

        batch = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}

        outputs = bb(**batch)
        hidden = outputs.last_hidden_state
        mask = batch.get("attention_mask")

        if mask is None:
            pooled = hidden[:, -1]
        else:
            sequence_length = mask.sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), sequence_length
            ]

        if truncate_dim is not None:
            pooled = pooled[:, :truncate_dim]

        features["sentence_embedding"] = F.normalize(pooled, p=2, dim=1)
        return features

    transformer = model._first_module()
    transformer.forward = types.MethodType(_training_forward, transformer)
    print("Forward patch applied.")

    # ──  Apply LoRA adapter ─────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGETS,
        lora_dropout=config.LORA_DROPOUT,
    )

    jina = _backbone(transformer)
    jina.add_adapter(config.TRAIN_ADAPTER, lora_cfg)
    jina.set_adapter(config.TRAIN_ADAPTER)
    print(f"LoRA adapter '{config.TRAIN_ADAPTER}' added.")

    # ──  Freeze base weights, keep LoRA trainable ───────────────────────
    for param in model.parameters():
        param.requires_grad = False
    for name, param in jina.named_parameters():
        if config.TRAIN_ADAPTER in name and "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ──  Cap sequence length to reduce memory usage ──────────────────────
    model.max_seq_length = config.MAX_SEQ_LENGTH
    print(f"max_seq_length set to {config.MAX_SEQ_LENGTH}")

    return model


if __name__ == "__main__":
    model = load_model()
    print("Model ready for training.")
