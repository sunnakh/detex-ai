# fine_tune.py — converted from jina-fine-tuning.ipynb

# import finetuner
from transformers import Trainer, TrainingArguments
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets

# ---------------------------------------------------------------------------
anchors, positives, negatives = [], [], []


def add_triplet(human, ai):
    if len(human) > 150 and len(ai) > 150:
        anchors.append(f"Query: {human[:1500]}")
        positives.append(f"Document: {human[:1500]}")
        negatives.append(f"Document: {ai[:1500]}")


# ---------------------------------------------------------------------------
# dataset pipeline — importing dataset from HC3 for fine tuning
# ---------------------------------------------------------------------------
from huggingface_hub import hf_hub_download, list_repo_files
import json, random

print("Loading HC3...")

# HC3 is stored as JSONL — find the 'all' file and parse it directly
# (dataset scripts are blocked in newer versions of `datasets`)
repo_files = list(list_repo_files("Hello-SimpleAI/HC3", repo_type="dataset"))
jsonl_files = [f for f in repo_files if f.endswith(".jsonl") and "all" in f]
print(f"Found HC3 files: {jsonl_files}")

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
hc3_pairs = hc3_pairs[:100000]
for h, a in hc3_pairs:
    add_triplet(h, a)

print(f"HC3 added: {len(hc3_pairs)} pairs | Running total: {len(anchors)}")

# ---------------------------------------------------------------------------
# importing data from Mage dataset for fine tuning
# ---------------------------------------------------------------------------
print("Loading MAGE...")
mage = load_dataset("yaful/MAGE", split="train")

human_mage = [x["text"] for x in mage if x["label"] == 0 and len(x["text"]) > 150]
ai_mage = [x["text"] for x in mage if x["label"] == 1 and len(x["text"]) > 150]
random.shuffle(human_mage)
random.shuffle(ai_mage)

mage_count = min(100000, len(human_mage))
for i in range(mage_count):
    add_triplet(human_mage[i], ai_mage[i % len(ai_mage)])

print(f"MAGE added: {mage_count} pairs | Running total: {len(anchors)}")

# ---------------------------------------------------------------------------
print("Loading RAID...")
raid = load_dataset("liamdugan/raid", split="train", streaming=True)

RAID_CAP = 50000
human_raid, ai_raid = [], []

for x in raid:
    gen = x["generation"]
    if len(gen) <= 150:
        continue
    if x["model"] == "human":
        if len(human_raid) < RAID_CAP:
            human_raid.append(gen)
    else:
        if len(ai_raid) < RAID_CAP * 5:
            ai_raid.append(gen)
    if len(human_raid) >= RAID_CAP and len(ai_raid) >= RAID_CAP * 5:
        break

random.shuffle(human_raid)
random.shuffle(ai_raid)

raid_count = len(human_raid)
for i in range(raid_count):
    add_triplet(human_raid[i], ai_raid[i % len(ai_raid)])

print(f" RAID added: {raid_count} pairs | Running total: {len(anchors)}")

# ---------------------------------------------------------------------------
print("Loading AI Detection Pile...")
pile = load_dataset("artem9k/ai-text-detection-pile", split="train")

human_pile = [
    x["text"] for x in pile if x["source"] == "human" and len(x["text"]) > 150
]
ai_pile = [x["text"] for x in pile if x["source"] != "human" and len(x["text"]) > 150]
random.shuffle(human_pile)
random.shuffle(ai_pile)

pile_count = min(100000, len(human_pile))
for i in range(pile_count):
    add_triplet(human_pile[i], ai_pile[i % len(ai_pile)])

print(f" Pile added: {pile_count} pairs | Running total: {len(anchors)}")

# ---------------------------------------------------------------------------
from datasets import Dataset


def quality_filter(example):
    a = example["anchor"].replace("Query: ", "")
    p = example["positive"].replace("Document: ", "")
    n = example["negative"].replace("Document: ", "")
    if len(a) < 150 or len(p) < 150 or len(n) < 150:
        return False
    # anchor == positive is intentional (same human text as query and document)
    if p[:100] == n[:100]:  # positive == negative (human text == AI text — bad pair)
        return False
    return True


raw_dataset = Dataset.from_dict(
    {
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    }
).shuffle(seed=42)

raw_dataset = raw_dataset.filter(quality_filter, num_proc=4)
print(f"After quality filter: {len(raw_dataset)}")

# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import torch
import gc

print("Mining hard negatives...")
miner = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    model_kwargs={"dtype": torch.bfloat16, "default_task": "classification"},
)
# Deduplicated AI corpus
ai_corpus = list(set(raw_dataset["negative"]))
# Reduce batch size and process in smaller chunks to avoid memory issues
ai_embeddings = []
chunk_size = 10000  # Process 10k samples at a time
for start in range(0, len(ai_corpus), chunk_size):
    chunk = ai_corpus[start : start + chunk_size]
    embeddings = miner.encode(
        chunk,
        batch_size=32,  # Reduced batch size
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        task="classification",
    )
    ai_embeddings.append(embeddings)
    del embeddings
    gc.collect()  # Free up memory
ai_embeddings = torch.cat(ai_embeddings, dim=0)
# Mine in chunks of 50k
chunk_size = 50000
hard_anchors, hard_positives, hard_negatives = [], [], []
for start in range(0, min(len(raw_dataset), 100000), chunk_size):
    chunk = raw_dataset.select(range(start, min(start + chunk_size, len(raw_dataset))))
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
# 70% random negatives + 30% hard negatives
n_hard = int(len(raw_dataset) * 0.30)
final_dataset = concatenate_datasets(
    [raw_dataset, hard_neg_dataset.select(range(min(n_hard, len(hard_neg_dataset))))]
).shuffle(seed=42)
split = final_dataset.train_test_split(test_size=0.05, seed=42)
print(f"Final — Train: {len(split['train'])} | Eval: {len(split['test'])}")

# ---------------------------------------------------------------------------
from peft import LoraConfig
import torch.nn.functional as F
import types

print("Loading model...")
model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    model_kwargs={"dtype": torch.bfloat16, "default_task": "classification"},
)

# ── Patch forward ─────────────────────────────────────────────────────────
# The Jina v5 custom_st.py Transformer.forward() wraps the ENTIRE forward
# in torch.no_grad() and calls self.model.eval() every pass — this makes
# training impossible. We replace it with a gradient-friendly version.
TRAIN_ADAPTER = "ai_detection"


def _training_forward(self, features, task=None, truncate_dim=None):
    self.model.set_adapter([TRAIN_ADAPTER])
    device = next(self.model.parameters()).device
    batch = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}
    outputs = self.model(**batch)
    hidden = outputs.last_hidden_state
    mask = batch.get("attention_mask")
    if mask is None:
        pooled = hidden[:, -1]
    else:
        sequence_lengths = mask.sum(dim=1) - 1
        pooled = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            sequence_lengths,
        ]
    if truncate_dim is not None:
        pooled = pooled[:, :truncate_dim]
    embeddings = F.normalize(pooled, p=2, dim=-1)
    features["sentence_embedding"] = embeddings
    return features


model[0].forward = types.MethodType(_training_forward, model[0])

# ── Add custom LoRA adapter ──────────────────────────────────────────────
# JinaEmbeddingsV5Model IS a PeftMixedModel — it already has task adapters.
# We add our own adapter with the desired rank instead of nesting get_peft_model.
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
jina = model[0].model  # JinaEmbeddingsV5Model (PeftMixedModel)
jina.add_adapter(TRAIN_ADAPTER, lora_config)
jina.set_adapter([TRAIN_ADAPTER])

# Freeze everything, then unfreeze only the new adapter's LoRA params
for param in model.parameters():
    param.requires_grad = False
for name, param in jina.named_parameters():
    if TRAIN_ADAPTER in name and "lora_" in name:
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ---------------------------------------------------------------------------
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss

base_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(
    model, base_loss, matryoshka_dims=[1024, 512, 256]  # preserve all truncation levels
)

# ---------------------------------------------------------------------------
from sentence_transformers.evaluation import InformationRetrievalEvaluator

eval_data = split["test"]

eval_queries = {str(i): eval_data[i]["anchor"] for i in range(len(eval_data))}
eval_corpus = {str(i): eval_data[i]["positive"] for i in range(len(eval_data))}
eval_relevant = {str(i): {str(i)} for i in range(len(eval_data))}

evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant,
    name="ai-detection-eval",
)

# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    output_dir="./jina-v5-ai-detection",
    # Core
    num_train_epochs=3,
    per_device_train_batch_size=8,  # reduced from 64 — MPS only has ~30GB
    gradient_accumulation_steps=16,  # effective batch = 128 (same as before)
    learning_rate=2e-4,  # LoRA tolerates higher LR than full fine-tune
    # Stability
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Speed & memory
    bf16=True,
    gradient_checkpointing=False,  # Incompatible with manual PEFT injection
    dataloader_num_workers=0,  # Disable multiprocessing to avoid pickling error
    # Evaluation & saving
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=3,
    logging_steps=50,
)

# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformerTrainer

if len(split["train"]) == 0:
    raise RuntimeError(
        "split['train'] is empty. Re-run all cells from the top in order:\n"
        "  1. anchors/positives/negatives init\n"
        "  2. HC3, MAGE, RAID, AI Detection Pile loading cells\n"
        "  3. Quality filter cell\n"
        "  4. Hard negative mining cell\n"
        "Then re-run the LoRA, loss, evaluator, and args cells before training."
    )
print(f"Train: {len(split['train']):,} | Eval: {len(split['test']):,}")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# ---------------------------------------------------------------------------
print("Merging LoRA weights...")
model[0].model = model[0].model.merge_and_unload()
model.save_pretrained("./jina-v5-ai-detection-final")
print("Model saved.")

# ---------------------------------------------------------------------------
# Inference — test with example text
# ---------------------------------------------------------------------------
# # ── Inference demo ───────────────────────────────────────────────────────────
# # Change TEST_TEXT to any passage you want to evaluate.
# TEST_TEXT = """
# The quick brown fox jumps over the lazy dog. I remember writing this sentence
# in school when I was learning to type. It felt oddly satisfying that every
# letter of the alphabet appeared in it.
# """

# # Reference anchors: a handful of known-human and known-AI passages.
# HUMAN_REFS = [
#     "Query: " + "I woke up at 6am and couldn't fall back asleep. Spent an hour just staring at the ceiling, thinking about nothing in particular.",
#     "Query: " + "Honestly, I didn't expect the movie to be that good. By the end I was crying and I'm not even sure why.",
#     "Query: " + "We drove through three states in one day. My back was killing me but the scenery made it worth it.",
# ]
# AI_REFS = [
#     "Query: " + "Artificial intelligence has revolutionized numerous industries by enabling machines to perform tasks that previously required human intelligence, such as image recognition and natural language processing.",
#     "Query: " + "The importance of maintaining a balanced diet cannot be overstated. Consuming a variety of nutrient-rich foods ensures optimal bodily function and promotes long-term health outcomes.",
#     "Query: " + "In conclusion, the implementation of sustainable practices within corporate frameworks is paramount to addressing the global climate crisis while ensuring continued economic viability.",
# ]

# from sentence_transformers import SentenceTransformer
# import torch
# import torch.nn.functional as F

# # Load the fine-tuned model (after merge cell) or fall back to the training model.
# try:
#     eval_model = SentenceTransformer("./jina-v5-ai-detection-final", trust_remote_code=True)
#     print("Loaded fine-tuned model from ./jina-v5-ai-detection-final")
# except Exception:
#     eval_model = model  # use the in-memory model if save hasn't happened yet
#     print("Using in-memory model (fine-tuned weights not saved yet)")

# query = f"Query: {TEST_TEXT.strip()[:1500]}"

# all_texts = [query] + HUMAN_REFS + AI_REFS
# embeddings = eval_model.encode(all_texts, convert_to_tensor=True, normalize_embeddings=True)

# q_emb = embeddings[0]
# human_embs = embeddings[1 : 1 + len(HUMAN_REFS)]
# ai_embs = embeddings[1 + len(HUMAN_REFS) :]

# human_score = F.cosine_similarity(q_emb.unsqueeze(0), human_embs).mean().item()
# ai_score    = F.cosine_similarity(q_emb.unsqueeze(0), ai_embs).mean().item()

# print(f"\n{'='*50}")
# print(f"  Avg similarity → human refs : {human_score:.4f}")
# print(f"  Avg similarity → AI refs    : {ai_score:.4f}")
# print(f"{'='*50}")
# verdict = "HUMAN" if human_score > ai_score else "AI-GENERATED"
# margin  = abs(human_score - ai_score)
# confidence = "high" if margin > 0.05 else "low"
# print(f"  Verdict : {verdict}  (confidence: {confidence}, margin={margin:.4f})")
# print(f"{'='*50}\n")
