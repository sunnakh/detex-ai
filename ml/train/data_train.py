# part_05_classifier_data.py
# Loads AI-Detection-Pile, TuringBench, Human-AI-Generated-Text → builds labeled dataset
# → embeds with fine-tuned Jina → saves
# OUTPUT: ./artifacts/clf_X_train.npy, clf_X_test.npy, clf_y_train.npy, clf_y_test.npy
#
# Datasets (all public, no token needed):
#   artem9k/ai-text-detection-pile  — text / source ("human"|"ai")           ~1.4M rows
#   Hello-SimpleAI/HC3              — human_answers / chatgpt_answers lists   ~24k rows
#   dmitva/human_ai_generated_text  — human_text / ai_text pairs              ~1M rows

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
    print("Loading AI-Text-Detection-Pile dataset: ...")

    PILE_CAP = 80_000
    pile = load_dataset("artem9k/ai-text-detection-pile", split="train")
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

    for t in pile_human[:PILE_CAP]:
        add_sample(texts, labels, t, 0)
    for t in pile_ai[:PILE_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"AI-Detection-Pile added {len(pile_human[:PILE_CAP])} human + {len(pile_ai[:PILE_CAP])} AI | Running total: {len(texts)}"
    )

    # ---------------------------------------------------------------------------
    # HC3 — Human ChatGPT Comparison Corpus — 24k QA pairs across 6 domains
    # Fields: human_answers (list[str]), chatgpt_answers (list[str])
    # TuringBench was removed: its loading script is no longer supported by
    # newer datasets versions (RuntimeError: Dataset scripts are no longer supported)
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading HC3 dataset: ...")

    HC3_CAP = 60_000  # per class — dataset is ~24k rows so this is uncapped
    hc3 = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
    hc3_human, hc3_ai = [], []

    for x in hc3:
        for ans in x.get("human_answers") or []:
            t = (ans or "").strip()
            if len(t) >= config.MIN_TEXT_LEN:
                hc3_human.append(t)
        for ans in x.get("chatgpt_answers") or []:
            t = (ans or "").strip()
            if len(t) >= config.MIN_TEXT_LEN:
                hc3_ai.append(t)

    random.shuffle(hc3_human)
    random.shuffle(hc3_ai)

    for t in hc3_human[:HC3_CAP]:
        add_sample(texts, labels, t, 0)
    for t in hc3_ai[:HC3_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"HC3 added {len(hc3_human[:HC3_CAP])} human + {len(hc3_ai[:HC3_CAP])} AI | Running total: {len(texts)}"
    )

    # ---------------------------------------------------------------------------
    # Human-AI-Generated-Text — paired rows: each row has human_text + ai_text
    # Fields: human_text (str), ai_text (str) — 1M paired rows
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading Human-AI-Generated-Text dataset: ...")

    HAGT_CAP = 30_000  # per class — 30k human + 30k AI
    hagt = load_dataset("dmitva/human_ai_generated_text", split="train")
    hagt_human, hagt_ai = [], []

    for x in hagt:
        h = (x.get("human_text") or "").strip()
        a = (x.get("ai_text") or "").strip()
        if len(h) >= config.MIN_TEXT_LEN:
            hagt_human.append(h)
        if len(a) >= config.MIN_TEXT_LEN:
            hagt_ai.append(a)

    random.shuffle(hagt_human)
    random.shuffle(hagt_ai)

    for t in hagt_human[:HAGT_CAP]:
        add_sample(texts, labels, t, 0)
    for t in hagt_ai[:HAGT_CAP]:
        add_sample(texts, labels, t, 1)

    print(
        f"Human-AI-Text added {len(hagt_human[:HAGT_CAP])} human + {len(hagt_ai[:HAGT_CAP])} AI | Running total: {len(texts)}"
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
