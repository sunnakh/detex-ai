import json
import random
import torch
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download, list_repo_files

def prepare_triplets(max_samples=200000):
    anchors, positives, negatives = [], [], []

    def add_triplet(human, ai):
        if len(human) > 150 and len(ai) > 150:
            anchors.append(f"Query: {human[:1500]}")
            positives.append(f"Document: {human[:1500]}")
            negatives.append(f"Document: {ai[:1500]}")

    print("Loading HC3...")
    try:
        repo_files = list(list_repo_files("Hello-SimpleAI/HC3", repo_type="dataset"))
        jsonl_files = [f for f in repo_files if f.endswith(".jsonl") and "all" in f]
        for fname in jsonl_files:
            path = hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename=fname, repo_type="dataset")
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    if not row.get("human_answers") or not row.get("chatgpt_answers"):
                        continue
                    for h in row["human_answers"]:
                        for a in row["chatgpt_answers"]:
                            if len(h) > 150 and len(a) > 150:
                                add_triplet(h, a)
    except Exception as e:
        print(f"Error loading HC3: {e}")

    print(f"HC3 added. Total triplets: {len(anchors)}")

    # Add other datasets if needed, following the same pattern...
    # For now, let's stick to a robust HC3 base and a subset of others to keep it manageable

    print("Loading MAGE...")
    try:
        mage = load_dataset("yaful/MAGE", split="train")
        human_mage = [x["text"] for x in mage if x["label"] == 0 and len(x["text"]) > 150]
        ai_mage = [x["text"] for x in mage if x["label"] == 1 and len(x["text"]) > 150]
        random.shuffle(human_mage)
        random.shuffle(ai_mage)
        for h, a in zip(human_mage[:50000], ai_mage[:50000]):
            add_triplet(h, a)
    except Exception as e:
        print(f"Error loading MAGE: {e}")

    print(f"MAGE added. Total triplets: {len(anchors)}")

    def quality_filter(example):
        p = example["positive"].replace("Document: ", "")
        n = example["negative"].replace("Document: ", "")
        return p[:100] != n[:100]

    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    }).shuffle(seed=42)

    dataset = dataset.filter(quality_filter)
    
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        
    return dataset

if __name__ == "__main__":
    ds = prepare_triplets(max_samples=50000) # Smaller sample for local testing
    save_path = Path("data/processed/fine_tune_triplets")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")
