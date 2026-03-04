# Loads fine-tuned model → embeds input text → scores vs human/AI refs → prints verdict


import os
import psutil
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import config

# Reuse the exact same model + LoRA architecture setup that train.py uses.
# This guarantees the backbone/adapter structure matches the saved weights.
from model import load_model as _build_model_arch

# ── Reference anchors (identical to pre-trained/detect.py for fair comparison) ─
HUMAN_ANCHORS = [
    "honestly i had no idea what to do so i just winged it and somehow it worked out",
    "ugh i woke up late again and missed the first half of the meeting, classic me",
    "my grandma made her famous soup and i swear it cures everything, no cap",
    "been staring at this bug for 3 hours and it turned out to be a missing semicolon lmao",
    "the sunset tonight was wild, took like 20 photos and none of them do it justice",
    "not sure if i should quit or keep going, just feeling stuck lately",
    "we stayed up till 3am talking about literally everything and nothing",
    "the coffee machine broke AGAIN and i am not okay",
    "i hate how i always second-guess myself right before submitting something",
    "drove past my old school today and got hit with a wave of nostalgia out of nowhere",
]

AI_ANCHORS = [
    "Artificial intelligence has revolutionized numerous industries by enabling machines to perform tasks that previously required human intelligence.",
    "The importance of maintaining a balanced diet cannot be overstated. Consuming a variety of nutrient-rich foods ensures optimal bodily function.",
    "In conclusion, the implementation of sustainable practices within corporate frameworks is paramount to addressing the global climate crisis.",
    "This comprehensive analysis examines the multifaceted implications of technological advancements on contemporary societal structures.",
    "To summarize, the evidence strongly suggests that further research is warranted to fully elucidate the underlying mechanisms.",
    "It is worth noting that this methodology offers significant advantages over traditional approaches in terms of both efficiency and scalability.",
    "The results of this study demonstrate a statistically significant correlation between the two variables under investigation.",
    "Furthermore, the integration of machine learning algorithms has proven instrumental in optimizing operational workflows across diverse sectors.",
    "This paper presents a novel framework for addressing the challenges associated with large-scale data processing in distributed environments.",
    "In light of the aforementioned considerations, it becomes evident that a holistic approach is required to effectively mitigate these risks.",
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
    return torch.tensor(
        model.encode(texts, normalize_embeddings=True),
        dtype=torch.float32,
    )


def predict(model: SentenceTransformer, text: str) -> dict:
    """
    Identical logic to pre-trained/detect.py — only the model differs.
    Same anchors, same normalization, same confidence thresholds.
    """
    import time
    t0 = time.perf_counter()

    all_texts = [text.strip()] + HUMAN_ANCHORS + AI_ANCHORS
    embeddings = embed(model, all_texts)

    q_emb      = embeddings[0]
    human_embs = embeddings[1 : 1 + len(HUMAN_ANCHORS)]
    ai_embs    = embeddings[1 + len(HUMAN_ANCHORS) :]

    human_score = F.cosine_similarity(q_emb.unsqueeze(0), human_embs).mean().item()
    ai_score    = F.cosine_similarity(q_emb.unsqueeze(0), ai_embs).mean().item()
    elapsed_ms  = (time.perf_counter() - t0) * 1000

    # Normalize so ai_pct + human_pct = 100%
    total     = human_score + ai_score
    human_pct = (human_score / total * 100) if total > 0 else 50.0
    ai_pct    = 100.0 - human_pct
    margin_pct = abs(ai_pct - human_pct)

    if margin_pct < 5:
        verdict    = "UNCERTAIN"
        confidence = "low"
    elif human_score > ai_score:
        verdict    = "HUMAN-WRITTEN"
        confidence = "high" if margin_pct > 20 else "medium"
    else:
        verdict    = "AI-GENERATED"
        confidence = "high" if margin_pct > 20 else "medium"

    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)

    return {
        "verdict":    verdict,
        "confidence": confidence,
        "human_pct":  round(human_pct,  2),
        "ai_pct":     round(ai_pct,     2),
        "margin_pct": round(margin_pct, 2),
        "elapsed_ms": round(elapsed_ms, 1),
        "word_count": len(text.split()),
        "char_count": len(text),
        "ram_gb":     round(ram_gb, 2)
    }


# ── Pretty print (identical to pre-trained/detect.py) ────────────────────────
VERDICT_COLORS = {
    "AI-GENERATED":  "\033[91m",  # red
    "HUMAN-WRITTEN": "\033[92m",  # green
    "UNCERTAIN":     "\033[93m",  # yellow
}
RESET = "\033[0m"
BOLD  = "\033[1m"
CONFIDENCE_ICONS = {"high": "●●●", "medium": "●●○", "low": "●○○"}


def print_result(r: dict):
    color     = VERDICT_COLORS.get(r["verdict"], "")
    icon      = CONFIDENCE_ICONS.get(r["confidence"], "○○○")
    bar_ai    = "█" * int(r["ai_pct"]    / 100 * 30)
    bar_human = "█" * int(r["human_pct"] / 100 * 30)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"  {BOLD}Verdict   :{RESET}  {color}{BOLD}{r['verdict']}{RESET}")
    print(f"  Confidence:  {icon}  ({r['confidence']})")
    print("╠══════════════════════════════════════════════════╣")
    print(f"  AI score    {r['ai_pct']:>6.2f}%  │{bar_ai:<30}│")
    print(f"  Human score {r['human_pct']:>6.2f}%  │{bar_human:<30}│")
    print(f"  Margin      {r['margin_pct']:>6.2f}%")
    print("╠══════════════════════════════════════════════════╣")
    print(
        f"  Words: {r['word_count']}   Chars: {r['char_count']}   Time: {r['elapsed_ms']} ms"
    )
    print(f"  RAM Usage: {r['ram_gb']} GB")
    print("╚══════════════════════════════════════════════════╝")
    print()


# ── Input helper (identical to pre-trained/detect.py) ────────────────────────
def get_text_input() -> str:
    print()
    print("  Paste or type your text, then press Enter twice to analyze:\n")
    lines = []
    while True:
        try:
            line = input("  │ ")
        except EOFError:
            break
        if line.strip() == "" and lines and lines[-1].strip() == "":
            break
        lines.append(line)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


# ── Main REPL ─────────────────────────────────────────────────────────────────
BANNER = r"""
  ██████╗ ███████╗████████╗███████╗██╗  ██╗
  ██╔══██╗██╔════╝╚══██╔══╝██╔════╝╚██╗██╔╝
  ██║  ██║█████╗     ██║   █████╗   ╚███╔╝
  ██║  ██║██╔══╝     ██║   ██╔══╝   ██╔██╗
  ██████╔╝███████╗   ██║   ███████╗██╔╝ ██╗
  ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
  AI Text Detector  ·  Jina v5 Small (fine-tuned)
"""

if __name__ == "__main__":
    import sys
    print(BANNER)
    model = load_model()
    print("Model loaded. Ready to detect.\n")

    print("  Type or paste text, press Enter twice to analyze.")
    print("  Type  q  or  quit  to exit.\n")

    while True:
        print("─" * 52)
        text = get_text_input().strip()

        if text.lower() in ("q", "quit", "exit"):
            print("\n  👋 Bye!\n")
            sys.exit(0)

        if not text:
            continue

        if len(text) < config.MIN_TEXT_LEN:
            print(f"\n  ⚠  Text too short (min {config.MIN_TEXT_LEN} chars). Try again.\n")
            continue

        print("\n  Analyzing…")
        result = predict(model, text)
        print_result(result)
