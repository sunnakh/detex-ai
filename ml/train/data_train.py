# part_05_classifier_data.py
# Loads artem9k/ai-text-detection-pile → builds labeled dataset
# → embeds with fine-tuned Jina → saves
# OUTPUT: ./artifacts/clf_X_train.npy, clf_X_test.npy, clf_y_train.npy, clf_y_test.npy
#
# Dataset (public, no token needed):
#   artem9k/ai-text-detection-pile — text / source ("human"|"ai") — ~1.4M rows

import os
import random

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

import config

random.seed(config.SEED)
np.random.seed(config.SEED)

SEP = "- " * 50

DATASET_NAME = "artem9k/ai-text-detection-pile"
# Cap per class — raise or lower to trade off training time vs coverage
HUMAN_CAP = 100_000
AI_CAP = 100_000


def add_sample(texts: list, labels: list, text: str, label: int) -> None:
    """label: 0 = human, 1 = AI"""
    text = text.strip()
    if len(text) >= config.MIN_TEXT_LEN:
        texts.append(text[: config.MAX_TEXT_LEN])
        labels.append(label)


def build_dataset() -> tuple[list, list]:
    texts: list[str] = []
    labels: list[int] = []

    # ---------------------------------------------------------------------------
    # AI-Text-Detection-Pile — GPT2/GPT3/ChatGPT/GPTJ + Reddit/WebText human text
    # Fields: text (str), source ("human" | "ai")
    # ---------------------------------------------------------------------------

    print(SEP)
    print(f"Loading {DATASET_NAME}: ...")

    pile = load_dataset(DATASET_NAME, split="train")
    pile_human, pile_ai = [], []

    for x in pile:
        text = (x.get("text") or "").strip()
        if len(text) < config.MIN_TEXT_LEN:
            continue
        if (x.get("source") or "").lower() == "human":
            pile_human.append(text)
        else:
            pile_ai.append(text)

    random.shuffle(pile_human)
    random.shuffle(pile_ai)

    for t in pile_human[:HUMAN_CAP]:
        add_sample(texts, labels, t, 0)
    for t in pile_ai[:AI_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"Added {len(pile_human[:HUMAN_CAP]):,} human + {len(pile_ai[:AI_CAP]):,} AI"
        f" | Running total: {len(texts):,}"
    )

    print(SEP)

    # ── Balance check ─────────────────────────────────────────────────────
    total_human = sum(1 for l in labels if l == 0)
    total_ai = sum(1 for l in labels if l == 1)
    print(f"Class balance — Human: {total_human:,} | AI: {total_ai:,}")
    if len(labels) > 0 and abs(total_human - total_ai) / len(labels) > 0.1:
        print(
            "⚠  Class imbalance > 10% detected — consider class_weight='balanced' in classifiers"
        )

    return texts, labels


if __name__ == "__main__":
    texts, labels = build_dataset()

    # ── Embed with fine-tuned Jina ─────────────────────────────────────────
    # Most expensive step — embeddings are saved so classifiers can be
    # retrained/tuned without re-embedding every time.
    print(SEP)
    print(f"Embedding {len(texts):,} samples with fine-tuned Jina...")

    # Load the fine-tuned model correctly:
    # FINAL_DIR contains only LoRA adapter weights saved by LoRATrainer._save().
    # SentenceTransformer() cannot load a raw LoRA adapter dir directly.
    import sys, os as _os, importlib, importlib.util

    _finetune_dir = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "fine-tune")
    )

    adapter_path = _os.path.join(config._ROOT, "checkpoints", "jina-v5-ai-detection-final")
    adapter_exists = _os.path.isdir(adapter_path) and _os.path.isfile(
        _os.path.join(adapter_path, "adapter_config.json")
    )

    if adapter_exists:
        print(f"  Loading fine-tuned model from {adapter_path} ...")

        # Load ml/fine-tune/config.py under a temp name so model.py's
        # `import config` picks up LORA_R and friends correctly.
        _ft_cfg_spec = importlib.util.spec_from_file_location(
            "config", _os.path.join(_finetune_dir, "config.py")
        )
        _ft_config = importlib.util.module_from_spec(_ft_cfg_spec)
        _ft_cfg_spec.loader.exec_module(_ft_config)

        # Temporarily replace the "config" in sys.modules so model.py uses it
        _orig_config = sys.modules.get("config")
        sys.modules["config"] = _ft_config

        _ft_model_spec = importlib.util.spec_from_file_location(
            "finetune_model", _os.path.join(_finetune_dir, "model.py")
        )
        _ft_model_mod = importlib.util.module_from_spec(_ft_model_spec)
        _ft_model_spec.loader.exec_module(_ft_model_mod)
        _load_model = _ft_model_mod.load_model

        # Restore the train config
        if _orig_config is not None:
            sys.modules["config"] = _orig_config

        import torch
        from safetensors.torch import load_file as _load_sf

        embedder = _load_model()
        transformer = embedder._first_module()
        backbone = getattr(transformer, "auto_model", None) or getattr(transformer, "model")
        sf_path = _os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = _os.path.join(adapter_path, "adapter_model.bin")
        saved_sd = _load_sf(sf_path, device="cpu") if _os.path.isfile(sf_path) else \
                   torch.load(bin_path, map_location="cpu")
        backbone.load_state_dict(saved_sd, strict=False)
        backbone.set_adapter(_ft_config.TRAIN_ADAPTER)
        print("  ✓ Fine-tuned LoRA weights applied.")
    else:
        print(f"⚠  No fine-tuned adapter found at {adapter_path}.")
        print(f"   Falling back to base model: {config.MODEL_ID}")
        from sentence_transformers import SentenceTransformer as _ST
        embedder = _ST(config.MODEL_ID, trust_remote_code=True)
    X = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    y = np.array(labels)

    # ── Train / test split ─────────────────────────────────────────────────
    # Stratified to preserve class balance in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y,
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs("./artifacts", exist_ok=True)

    np.save("./artifacts/clf_X_train.npy", X_train)
    np.save("./artifacts/clf_X_test.npy", X_test)
    np.save("./artifacts/clf_y_train.npy", y_train)
    np.save("./artifacts/clf_y_test.npy", y_test)
    print("Embeddings saved to ./artifacts/")
