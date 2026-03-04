import gc
import json
import os
import random
import config
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files, login
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

# ── HuggingFace auth ──────────────────────────────────────────────────────
# Set HF_TOKEN env var, or replace the string below with your token
_hf_token = os.environ.get("HF_TOKEN", "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")
login(token=_hf_token, add_to_git_credential=False)

random.seed(config.SEED)


def build_dataset() -> Dataset:
    anchors, positives, negatives = [], [], []

    def add_triplet(human: str, ai: str) -> None:
        if len(human) > 150 and len(ai) > 150:
            anchors.append(f"Query: {human[:1500]}")
            positives.append(f"Document: {human[:1500]}")
            negatives.append(f"Document: {ai[:1500]}")

    def quality_filter(example) -> bool:
        # anchor == positive is intentional (same human text as query and document)
        p = example["positive"].replace("Document: ", "")
        n = example["negative"].replace("Document: ", "")
        if (
            p[:100] == n[:100]
        ):  # positive == negative (human text == AI text — bad pair)
            return False
        return True

    SEP = "- " * 50

    # ---------------------------------------------------------------------------
    # dataset pipeline — importing dataset from HC3 for fine tuning
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading HC3 dataset: ...")

    HC3_num = config.HC3_CAP
    repo_files = list(list_repo_files("Hello-SimpleAI/HC3", repo_type="dataset"))
    jsonl_files = [f for f in repo_files if f.endswith(".jsonl") and "all" in f]

    hc3_pairs = []
    for fname in jsonl_files:
        path = hf_hub_download(
            repo_id="Hello-SimpleAI/HC3", filename=fname, repo_type="dataset"
        )
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if not row.get("human_answers") or not row.get("chatgpt_answers"):
                    continue
                for h in row["human_answers"]:
                    for a in row["chatgpt_answers"]:
                        if len(h) > 150 and len(a) > 150:
                            hc3_pairs.append((h, a))

    random.shuffle(hc3_pairs)
    hc3_pairs = hc3_pairs[:HC3_num]
    for h, a in hc3_pairs:
        add_triplet(h, a)

    print(f"HC3 added {len(hc3_pairs)} pairs | Running total: {len(anchors)}")

    # ---------------------------------------------------------------------------
    # importing data from Mage dataset for fine tuning
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading Mage dataset: ...")

    MAGE_no = config.MAGE_CAP
    mage_dataset = load_dataset(path="yaful/MAGE", split="train")

    human_mage = [
        x["text"] for x in mage_dataset if x["label"] == 0 and len(x["text"]) > 150
    ]
    ai_mage = [
        x["text"] for x in mage_dataset if x["label"] == 1 and len(x["text"]) > 150
    ]

    random.shuffle(human_mage)
    random.shuffle(ai_mage)

    mage_count = min(MAGE_no, len(human_mage))
    for i in range(mage_count):
        add_triplet(human=human_mage[i], ai=ai_mage[i % len(ai_mage)])

    print(f"Mage added: {mage_count} pairs | Running total: {len(anchors)}")

    # ---------------------------------------------------------------------------
    # importing data from RAID dataset for fine tuning
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading RAID dataset: ...")

    RAID_no = config.RAID_CAP
    raid_dataset = load_dataset(path="liamdugan/raid", split="train", streaming=True)
    human_raid, ai_raid = [], []

    for x in raid_dataset:
        gen = x["generation"]
        if len(gen) <= 150:
            continue
        if x["model"] == "human":
            if len(human_raid) < RAID_no:
                human_raid.append(gen)
        else:
            if len(ai_raid) < RAID_no * 5:
                ai_raid.append(gen)
        if len(human_raid) >= RAID_no and len(ai_raid) >= RAID_no * 5:
            break

    random.shuffle(human_raid)
    random.shuffle(ai_raid)

    raid_count = len(human_raid)
    for i in range(raid_count):
        add_triplet(human=human_raid[i], ai=ai_raid[i % len(ai_raid)])

    print(f"RAID added: {raid_count} pairs | Running total: {len(anchors)}")

    # ---------------------------------------------------------------------------
    # importing data from AI Detection Pile dataset for fine tuning
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading AI Detection Pile dataset: ...")

    PILE_no = config.PILE_CAP
    pile_dataset = load_dataset(path="artem9k/ai-text-detection-pile", split="train")
    human_pile = [
        x["text"]
        for x in pile_dataset
        if x["source"] == "human" and len(x["text"]) > 150
    ]
    ai_pile = [
        x["text"]
        for x in pile_dataset
        if x["source"] != "human" and len(x["text"]) > 150
    ]

    random.shuffle(human_pile)
    random.shuffle(ai_pile)

    pile_count = min(PILE_no, len(human_pile))
    for i in range(pile_count):
        add_triplet(human=human_pile[i], ai=ai_pile[i % len(ai_pile)])

    print(f"Pile added {pile_count} pairs | Running total: {len(anchors)}")
    print(SEP)

    # ---------------------------------------------------------------------------
    # build and filter dataset
    # ---------------------------------------------------------------------------

    raw_dataset = Dataset.from_dict(
        {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
        }
    ).shuffle(seed=42)

    raw_dataset = raw_dataset.filter(quality_filter, num_proc=4)
    print(f"After quality filter: {len(raw_dataset)}")

    return raw_dataset


if __name__ == "__main__":
    raw_dataset = build_dataset()

    # ── Hard negative mining ───────────────────────────────────────────────
    print("=" * 60)
    print("Mining hard negatives...")

    miner = SentenceTransformer(
        config.MODEL_ID,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16, "default_task": "classification"},
    )

    # Build deduplicated AI corpus embeddings in chunks
    ai_corpus = list(set(raw_dataset["negative"]))
    ai_embeddings = []
    for start in range(0, len(ai_corpus), 10_000):
        chunk = ai_corpus[start : start + 10_000]
        emb = miner.encode(
            chunk,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            task="classification",
        )
        ai_embeddings.append(emb)
        del emb
        gc.collect()
    ai_embeddings = torch.cat(ai_embeddings, dim=0)

    # Mine hard negatives in chunks of 50k
    hard_anchors, hard_positives, hard_negatives = [], [], []
    for start in range(0, min(len(raw_dataset), config.HARD_NEG_CAP), 50_000):
        chunk = raw_dataset.select(range(start, min(start + 50_000, len(raw_dataset))))
        anchor_embs = miner.encode(
            chunk["anchor"],
            batch_size=32,
            convert_to_tensor=True,
            device="cuda",
            show_progress_bar=True,
            task="classification",
        )
        results = semantic_search(anchor_embs, ai_embeddings, top_k=5)
        for i, hits in enumerate(results):
            hard_neg = ai_corpus[hits[0]["corpus_id"]]
            if hard_neg[:100] == chunk["positive"][i][:100]:
                hard_neg = ai_corpus[hits[1]["corpus_id"]]
            hard_anchors.append(chunk["anchor"][i])
            hard_positives.append(chunk["positive"][i])
            hard_negatives.append(hard_neg)

    hard_neg_dataset = Dataset.from_dict(
        {
            "anchor": hard_anchors,
            "positive": hard_positives,
            "negative": hard_negatives,
        }
    )

    # config.HARD_NEG_RATIO of the final dataset will be hard negatives
    n_hard = int(len(raw_dataset) * config.HARD_NEG_RATIO)
    final_dataset = concatenate_datasets(
        [
            raw_dataset,
            hard_neg_dataset.select(range(min(n_hard, len(hard_neg_dataset)))),
        ]
    ).shuffle(seed=config.SEED)

    split = final_dataset.train_test_split(test_size=config.TEST_SIZE, seed=config.SEED)
    print(f"Final — Train: {len(split['train']):,} | Eval: {len(split['test']):,}")

    # ── Save ──────────────────────────────────────────────────────────────
    split.save_to_disk(config.DATASET_PATH)
    print(f"Dataset saved to {config.DATASET_PATH}")
