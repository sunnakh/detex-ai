"""
detex.ai — FastAPI Backend
Using jinaai/jina-embeddings-v5-text-small + SVM (best_clf.joblib) for detection.
Refactored into modular architecture.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ml import load_models, unload_models
from app.api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    load_models()
    yield
    # Unload on shutdown
    unload_models()


# app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="detex.ai API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
