import os
from fastapi import APIRouter, File, HTTPException, UploadFile
from google import genai

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_BYTES, GEMINI_API_KEY
from app.schemas import (
    DetectRequest,
    DetectResponse,
    DetectFileResponse,
    HumanizeRequest,
    HumanizeResponse,
)
from app.ml import embed_and_classify, is_ready
from app.utils import extract_text

router = APIRouter()

# ── Gemini Setup ─────────────────────────────────────────────────────────────
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    ready = is_ready()
    return {"status": "ok" if ready else "loading", "model_ready": ready}


@router.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please try again in a moment",
        )

    ai_score, human_score, elapsed = embed_and_classify(req.text)

    return DetectResponse(
        label="AI-generated" if ai_score > human_score else "Human-written",
        confidence=round(max(ai_score, human_score), 4),
        ai_score=round(ai_score, 4),
        human_score=round(human_score, 4),
        word_count=len(req.text.split()),
        char_count=len(req.text),
        analysis_time_ms=round(elapsed, 2),
    )


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(req: HumanizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not gemini_client:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API key is not configured in the environment."
        )
        
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


@router.post("/detect-file", response_model=DetectFileResponse)
async def detect_file(file: UploadFile = File(...)):
    if not is_ready():
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
        text = extract_text(filename, data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {exc}")

    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the file. Is it empty or scanned?",
        )

    # ── Inference (same pipeline as /detect) ─────────────────────────────────
    ai_score, human_score, elapsed = embed_and_classify(text)

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
