"""
detex.ai — FastAPI Backend
Using jinaai/jina-embeddings-v5-text-small + SVM (best_clf.joblib) for detection.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# paths
_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "jina-v5-finetuned"
)
CLF_PATH = os.path.join(_ARTIFACTS_DIR, "best_clf.joblib")
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"

# Global model holders ───────────────────────────────────────────────────────
clf = None
model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    print(f"[startup] All models ready on {device}.")
    yield
    clf = None
    model = None
    tokenizer = None


# app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="detex.ai API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Schemas ────────────────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class DetectResponse(BaseModel):
    label: str
    confidence: float
    ai_score: float
    human_score: float
    word_count: int
    char_count: int
    analysis_time_ms: float


#  Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    ready = clf is not None and model is not None
    return {"status": "ok" if ready else "loading", "model_ready": ready}


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if clf is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please try again in a moment",
        )

    t0 = time.perf_counter()

    # exact same embedding logic as training notebook
    encoded = tokenizer(
        [req.text], padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        output = model(**encoded)

    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
    embedding = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

    proba = clf.predict_proba(embedding)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    # classes_[0] = human (0), classes_[1] = AI (1) by convention
    ai_score = float(proba[1])
    human_score = float(proba[0])

    return DetectResponse(
        label="AI-generated" if ai_score > human_score else "Human-written",
        confidence=round(max(ai_score, human_score), 4),
        ai_score=round(ai_score, 4),
        human_score=round(human_score, 4),
        word_count=len(req.text.split()),
        char_count=len(req.text),
        analysis_time_ms=round(elapsed, 2),
    )
