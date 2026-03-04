# Loads fine-tuned model → embeds input text → scores vs human/AI refs → prints verdict


import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import config

# Reuse the exact same model + LoRA architecture setup that train.py uses.
# This guarantees the backbone/adapter structure matches the saved weights.
from model import load_model as _build_model_arch

# ── Reference anchors ─────────────────────────────────────────────────────
# A small set of known-human and known-AI passages used as comparison anchors.
# Extend these with more diverse examples to improve verdict reliability.
HUMAN_REFS = [
    "Query: I woke up at 6am and couldn't fall back asleep. Spent an hour just staring at the ceiling, thinking about nothing in particular.",
    "Query: Honestly, I didn't expect the movie to be that good. By the end I was crying and I'm not even sure why.",
    "Query: We drove through three states in one day. My back was killing me but the scenery made it worth it.",
]

AI_REFS = [
    "Query: Artificial intelligence has revolutionized numerous industries by enabling machines to perform tasks that previously required human intelligence, such as image recognition and natural language processing.",
    "Query: The importance of maintaining a balanced diet cannot be overstated. Consuming a variety of nutrient-rich foods ensures optimal bodily function and promotes long-term health outcomes.",
    "Query: In conclusion, the implementation of sustainable practices within corporate frameworks is paramount to addressing the global climate crisis while ensuring continued economic viability.",
]


def _has_lora_adapter(path: str) -> bool:
    """True when the directory contains a LoRA adapter saved by train.py."""
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "adapter_config.json")
    )


def _load_saved_weights(path: str) -> dict:
    """Load adapter weights — tries safetensors first, falls back to .bin."""
    sf_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    if os.path.isfile(sf_path):
        from safetensors.torch import load_file

        return load_file(sf_path, device="cpu")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(
        f"No adapter weights found in {path} "
        "(expected adapter_model.safetensors or adapter_model.bin)"
    )


def load_model() -> SentenceTransformer:
    """
    Load the fine-tuned model.

    train.py saves raw backbone state-dict keys (lora_* keys) via LoRATrainer._save().
    PeftModel.from_pretrained() cannot reliably consume those on a plain backbone.

    Correct approach:
      1. Call model.py's load_model() — builds the IDENTICAL LoRA architecture
         (same adapter name, same targets, same forward patch) as training.
      2. Extract the backbone that already has the LoRA layers wired in.
      3. Overwrite the random LoRA weights with the trained ones via load_state_dict.

    Falls back to the base pre-trained model if no checkpoint is found.
    """
    if _has_lora_adapter(config.FINAL_DIR):
        print(f"Loading fine-tuned model from {config.FINAL_DIR} ...")
        try:
            # Step 1 — build architecture (base + LoRA + forward patch)
            model = _build_model_arch()

            # Step 2 — get the backbone that has LoRA layers already applied
            transformer = model._first_module()
            backbone = getattr(transformer, "auto_model", None) or getattr(
                transformer, "model"
            )

            # Step 3 — load saved weights and apply them to the backbone
            saved_sd = _load_saved_weights(config.FINAL_DIR)
            missing, unexpected = backbone.load_state_dict(saved_sd, strict=False)

            lora_missing = [k for k in missing if "lora_" in k]
            if lora_missing:
                print(
                    f"  ⚠ {len(lora_missing)} LoRA keys not loaded: {lora_missing[:3]} ..."
                )
            else:
                print(f"  ✓ All LoRA weights loaded ({len(saved_sd)} tensors).")

            if unexpected:
                print(f"  ⚠ {len(unexpected)} unexpected keys ignored.")

            # Keep adapter active for inference
            backbone.set_adapter(config.TRAIN_ADAPTER)
            print("  ✓ Fine-tuned model ready.\n")
            return model

        except Exception as exc:
            print(f"  ✗ Failed to load fine-tuned model: {exc}")
            print("  Falling back to base pre-trained model.\n")

    else:
        print(
            f"No fine-tuned checkpoint found at '{config.FINAL_DIR}'.\n"
            f"  Run train.py first to produce a checkpoint.\n"
            f"  Falling back to base pre-trained model: {config.MODEL_ID}\n"
        )

    # Fallback — plain base model, no LoRA
    print(f"  Loading base model: {config.MODEL_ID} ...")
    return SentenceTransformer(
        config.MODEL_ID,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16, "default_task": "text-matching"},
    )


def embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def predict(model: SentenceTransformer, text: str) -> dict:
    """
    Embeds the input text and computes its avg cosine similarity
    to human and AI reference anchors, then returns a verdict dict.
    """
    query = f"Query: {text.strip()[:config.MAX_TEXT_LEN]}"

    all_texts = [query] + HUMAN_REFS + AI_REFS
    embeddings = embed(model, all_texts)

    q_emb = embeddings[0]
    human_embs = embeddings[1 : 1 + len(HUMAN_REFS)]
    ai_embs = embeddings[1 + len(HUMAN_REFS) :]

    human_score = F.cosine_similarity(q_emb.unsqueeze(0), human_embs).mean().item()
    ai_score = F.cosine_similarity(q_emb.unsqueeze(0), ai_embs).mean().item()
    margin = abs(human_score - ai_score)

    # Normalise the two scores so they sum to 100 %
    total = human_score + ai_score
    human_pct = (human_score / total * 100) if total > 0 else 50.0
    ai_pct = 100.0 - human_pct

    verdict = "HUMAN" if human_score > ai_score else "AI-GENERATED"
    confidence = "high" if margin > config.CONFIDENCE_THRESHOLD else "low"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "margin": round(margin, 4),
        "human_score": round(human_score, 4),
        "ai_score": round(ai_score, 4),
        "human_pct": round(human_pct, 1),
        "ai_pct": round(ai_pct, 1),
    }


def print_result(result: dict):
    sep = "=" * 50
    verdict = result["verdict"]
    conf = result["confidence"].upper()
    human_pct = result["human_pct"]
    ai_pct = result["ai_pct"]

    # Progress-bar style indicator (40 chars wide)
    bar_width = 40
    human_fill = round(bar_width * human_pct / 100)
    ai_fill = bar_width - human_fill
    bar = "H" * human_fill + "A" * ai_fill

    print(f"\n{sep}")
    print(f"  Human        : {human_pct:5.1f}%")
    print(f"  AI-Generated : {ai_pct:5.1f}%")
    print(f"  [{bar}]")
    print(f"{sep}")
    print(f"  Verdict      : {verdict}  [{conf} confidence]")
    print(f"{sep}\n")


def get_multiline_input() -> str:
    """
    Lets the user paste or type multi-line text.
    They type END on a new line to finish.
    """
    print("\nPaste or type your text below.")
    print("When done, type  END  on a new line and press Enter:\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


# ── Interactive loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model()
    print("\nModel loaded. Ready to detect.\n")

    while True:
        print("─ " * 50)
        print("Options:  [1] Check text   [2] Quit")
        choice = input(">> ").strip()

        if choice == "2" or choice.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break

        if choice == "1":
            text = get_multiline_input()

            if len(text.strip()) < config.MIN_TEXT_LEN:
                print(
                    f"\nText too short. Minimum {config.MIN_TEXT_LEN} characters required.\n"
                )
                continue

            print("\nAnalyzing...")
            result = predict(model, text)
            print_result(result)
        else:
            print("Invalid option. Enter 1 or 2.")
