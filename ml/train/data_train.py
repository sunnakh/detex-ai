# part_05_classifier_data.py
# Loads M4, TuringBench, OUTFOX → builds labeled dataset → embeds with fine-tuned Jina → saves
# OUTPUT: ./artifacts/clf_X_train.npy, clf_X_test.npy, clf_y_train.npy, clf_y_test.npy

import os
import random

import numpy as np
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

import config

random.seed(config.SEED)
np.random.seed(config.SEED)

SEP = "- " * 50


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
    # M4 — 6 generators × 4 domains — best diversity for classifier training
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading M4 dataset: ...")

    M4_CAP = 80_000
    m4 = load_dataset("NLP4TAS/M4", split="train")
    m4_human, m4_ai = [], []

    for x in m4:
        text = x.get("text", "").strip()
        if len(text) < config.MIN_TEXT_LEN:
            continue
        if x["label"] == 0:
            m4_human.append(text)
        else:
            m4_ai.append(text)

    random.shuffle(m4_human)
    random.shuffle(m4_ai)

    for t in m4_human[:M4_CAP]:
        add_sample(texts, labels, t, 0)
    for t in m4_ai[:M4_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"M4 added {len(m4_human[:M4_CAP])} human + {len(m4_ai[:M4_CAP])} AI | Running total: {len(texts)}"
    )

    # ---------------------------------------------------------------------------
    # TuringBench — 19 LLMs — critical for cross-model robustness
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading TuringBench dataset: ...")

    TURING_CAP = 60_000
    turing = load_dataset("turingbench/TuringBench", split="train")
    turing_human, turing_ai = [], []

    for x in turing:
        text = x.get("Generation", "").strip()
        if len(text) < config.MIN_TEXT_LEN:
            continue
        if x["label"] == "human":
            turing_human.append(text)
        else:
            turing_ai.append(text)

    random.shuffle(turing_human)
    random.shuffle(turing_ai)

    for t in turing_human[:TURING_CAP]:
        add_sample(texts, labels, t, 0)
    for t in turing_ai[:TURING_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"TuringBench added {len(turing_human[:TURING_CAP])} human + {len(turing_ai[:TURING_CAP])} AI | Running total: {len(texts)}"
    )

    # ---------------------------------------------------------------------------
    # OUTFOX — adversarial: humans fooling detectors, AI mimicking humans
    # Without this, classifiers fail on paraphrased/edited AI text in production
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading OUTFOX dataset: ...")

    OUTFOX_CAP = 30_000
    outfox = load_dataset("ryokan76/outfox", split="train")
    outfox_human, outfox_ai = [], []

    for x in outfox:
        text = x.get("text", "").strip()
        if len(text) < config.MIN_TEXT_LEN:
            continue
        if x["label"] == 0:
            outfox_human.append(text)
        else:
            outfox_ai.append(text)

    random.shuffle(outfox_human)
    random.shuffle(outfox_ai)

    for t in outfox_human[:OUTFOX_CAP]:
        add_sample(texts, labels, t, 0)
    for t in outfox_ai[:OUTFOX_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"OUTFOX added {len(outfox_human[:OUTFOX_CAP])} human + {len(outfox_ai[:OUTFOX_CAP])} AI | Running total: {len(texts)}"
    )

    print(SEP)

    # ── Balance check ─────────────────────────────────────────────────────
    total_human = sum(1 for l in labels if l == 0)
    total_ai = sum(1 for l in labels if l == 1)
    print(f"Class balance — Human: {total_human:,} | AI: {total_ai:,}")
    if abs(total_human - total_ai) / len(labels) > 0.1:
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

    embedder = SentenceTransformer(config.FINAL_DIR, trust_remote_code=True)
    X = embedder.encode(
        texts,
        batch_size=64,
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
