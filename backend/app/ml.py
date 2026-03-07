import time
import joblib
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import CLF_PATH, EMBEDDING_MODEL

# Global model holders
clf = None
model = None
tokenizer = None
device = None

def load_models():
    """Loads the classifier and embedding models into global memory."""
    global clf, model, tokenizer, device
    print(f"[startup] Loading classifier from {CLF_PATH}")
    clf = joblib.load(CLF_PATH)
    print("[startup] Loading embedding model (may download on first run)…")

    # CUDA > MPS > CPU (Docker/Linux will use CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, trust_remote_code=True, dtype=dtype
    ).to(device)
    model.eval()

    print(f"[startup] All models ready on {device}.")

def unload_models():
    """Clears models from memory."""
    global clf, model, tokenizer
    clf = None
    model = None
    tokenizer = None


def embed_and_classify(text: str):
    """Shared embedding + classification logic. Returns (ai_score, human_score, elapsed_ms)."""
    encoded = tokenizer(
        [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        output = model(**encoded)

    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
    embedding = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

    proba = clf.predict_proba(embedding)[0]
    elapsed = (time.perf_counter() - t0) * 1000
    return float(proba[1]), float(proba[0]), elapsed

def is_ready() -> bool:
    return clf is not None and model is not None
