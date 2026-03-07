"""
detex.ai — FastAPI Backend
Using jinaai/jina-embeddings-v5-text-small + SVM (best_clf.joblib) for detection.
"""

import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

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
        EMBEDDING_MODEL, trust_remote_code=True, dtype=dtype
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

class HumanizeRequest(BaseModel):
    text: str

class HumanizeResponse(BaseModel):
    humanized_text: str


class DetectResponse(BaseModel):
    label: str
    confidence: float
    ai_score: float
    human_score: float
    word_count: int
    char_count: int
    analysis_time_ms: float


class DetectFileResponse(DetectResponse):
    filename: str
    extracted_chars: int


# ── File upload limits / allowed types ────────────────────────────────────────
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from .txt / .pdf / .docx bytes."""
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Unsupported extension: {ext}")


def _embed_and_classify(text: str):
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


# ── Gemini Setup ─────────────────────────────────────────────────────────────
# We initialize the client inside the route or globally if key is available.
# Since the user provided the key, we'll set it in the environment.
os.environ["GEMINI_API_KEY"] = "AIzaSyBjmsD0SCI-w_X36r_aIG6WI90wye-y8bM"
gemini_client = genai.Client()

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

    ai_score, human_score, elapsed = _embed_and_classify(req.text)

    return DetectResponse(
        label="AI-generated" if ai_score > human_score else "Human-written",
        confidence=round(max(ai_score, human_score), 4),
        ai_score=round(ai_score, 4),
        human_score=round(human_score, 4),
        word_count=len(req.text.split()),
        char_count=len(req.text),
        analysis_time_ms=round(elapsed, 2),
    )


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(req: HumanizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    prompt = f"""
You are an expert copywriter tasked with rewriting the following text to bypass AI detectors.
Your goal is to make the text sound completely natural, conversational, and human-written.

Guidelines:
1. Increase burstiness (mix very short sentences with longer, complex ones).
2. Increase perplexity (use varied vocabulary, analogies, and slightly informal phrasing where appropriate).
3. Do NOT use common AI transition words (e.g., "Furthermore", "In conclusion", "Moreover", "Additionally", "It is important to note").
4. Keep the exact same core meaning and formatting (e.g., if it's an email, keep it an email).
5. Output ONLY the rewritten text, nothing else. No introductions or explanations.

Text to rewrite:
{req.text}
"""
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return HumanizeResponse(humanized_text=response.text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/detect-file", response_model=DetectFileResponse)
async def detect_file(file: UploadFile = File(...)):
    if clf is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please try again in a moment",
        )

    # ── Validate extension ───────────────────────────────────────────────────
    filename = file.filename or ""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .txt, .pdf, .docx",
        )

    # ── Read and validate size ───────────────────────────────────────────────
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) // 1024 // 1024} MB). Max is 10 MB.",
        )

    # ── Extract text ─────────────────────────────────────────────────────────
    try:
        text = _extract_text(filename, data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {exc}")

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the file. Is it empty or scanned?",
        )

    # ── Inference (same pipeline as /detect) ─────────────────────────────────
    ai_score, human_score, elapsed = _embed_and_classify(text)

    return DetectFileResponse(
        label="AI-generated" if ai_score > human_score else "Human-written",
        confidence=round(max(ai_score, human_score), 4),
        ai_score=round(ai_score, 4),
        human_score=round(human_score, 4),
        word_count=len(text.split()),
        char_count=len(text),
        analysis_time_ms=round(elapsed, 2),
        filename=filename,
        extracted_chars=len(text),
    )
