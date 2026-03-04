"""
detect.py — Pretrained Jina v5 Small: AI vs Human Text Detector
================================================================
Uses jinaai/jina-embeddings-v5-text-small (NO fine-tuning needed).
Embeds your input + reference anchors, then votes via cosine similarity.

Run:
    python detect.py
"""

import sys
import time
import os
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
MAX_SEQ_LEN = 512  # safe limit for jina-small
MIN_CHARS = 80  # reject very short inputs
MARGIN_THRESH = 0.005  # below this → "uncertain"

# ── Reference anchors ─────────────────────────────────────────────────────────
# A diverse pool of known-human and known-AI passages.
# More anchors = more robust voting. Feel free to extend these.

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


# ── Device setup ──────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(device: torch.device):
    print(f"\n[•] Loading {MODEL_ID} …")
    print("    (first run will download ~500 MB, cached afterwards)\n")
    # bfloat16 on GPU, float32 on CPU (bfloat16 unsupported on CPU)
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"[✓] Model ready on {device}\n")
    return model


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed(texts: list[str], model) -> torch.Tensor:
    """Use Jina's official encode() with task='classification'.
    Returns a float32 tensor of shape (N, dim), L2-normalized.
    """
    with torch.no_grad():
        embeddings = model.encode(
            texts=texts,
            task="classification",
            max_length=MAX_SEQ_LEN,
        )
    # encode() returns numpy array — convert to tensor for cosine_similarity
    return torch.tensor(embeddings, dtype=torch.float32)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str, model) -> dict:
    t0 = time.perf_counter()

    # task="classification" tells Jina v5 to apply the correct internal prefix
    all_texts = [text.strip()] + HUMAN_ANCHORS + AI_ANCHORS
    embeddings = embed(all_texts, model)

    q_emb = embeddings[0]  # (dim,)
    human_embs = embeddings[1 : 1 + len(HUMAN_ANCHORS)]  # (N, dim)
    ai_embs = embeddings[1 + len(HUMAN_ANCHORS) :]  # (M, dim)

    human_score = F.cosine_similarity(q_emb.unsqueeze(0), human_embs).mean().item()
    ai_score = F.cosine_similarity(q_emb.unsqueeze(0), ai_embs).mean().item()
    margin = abs(human_score - ai_score)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Normalize so ai_pct + human_pct = 100% (proper probability split)
    total = ai_score + human_score
    ai_pct = (ai_score / total) * 100
    human_pct = (human_score / total) * 100
    margin_pct = abs(ai_pct - human_pct)

    if margin_pct < 5:  # less than 5% gap → uncertain
        verdict = "UNCERTAIN"
        confidence = "low"
    elif human_score > ai_score:
        verdict = "HUMAN-WRITTEN"
        confidence = "high" if margin_pct > 20 else "medium"
    else:
        verdict = "AI-GENERATED"
        confidence = "high" if margin_pct > 20 else "medium"

    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "human_pct": round(human_pct, 2),
        "ai_pct": round(ai_pct, 2),
        "margin_pct": round(margin_pct, 2),
        "elapsed_ms": round(elapsed_ms, 1),
        "word_count": len(text.split()),
        "char_count": len(text),
        "ram_gb": round(ram_gb, 2)
    }


# ── Pretty print ──────────────────────────────────────────────────────────────
VERDICT_COLORS = {
    "AI-GENERATED": "\033[91m",  # red
    "HUMAN-WRITTEN": "\033[92m",  # green
    "UNCERTAIN": "\033[93m",  # yellow
}
RESET = "\033[0m"
BOLD = "\033[1m"

CONFIDENCE_ICONS = {"high": "●●●", "medium": "●●○", "low": "●○○"}


def print_result(r: dict):
    color = VERDICT_COLORS.get(r["verdict"], "")
    icon = CONFIDENCE_ICONS.get(r["confidence"], "○○○")
    bar_ai = "█" * int(r["ai_pct"] / 100 * 30)
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


# ── Input helpers ─────────────────────────────────────────────────────────────
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
            # Two consecutive blank lines = done
            break
        lines.append(line)
    # Strip trailing blank lines
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
  AI Text Detector  ·  Jina v5 Small (pretrained)
"""


def main():
    print(BANNER)
    device = get_device()
    model = load_model(device)

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

        if len(text) < MIN_CHARS:
            print(f"\n  ⚠  Text too short (min {MIN_CHARS} chars). Try again.\n")
            continue

        print("\n  Analyzing…")
        result = predict(text, model)
        print_result(result)


if __name__ == "__main__":
    main()
