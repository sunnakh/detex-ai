import joblib
from pathlib import Path
from ml.embeddings import get_embeddings

BEST_MODEL_DIR = Path("artifacts") / "jina-v5-finetuned"

_clf = None


def _load_clf():
    """Lazily load the calibrated production classifier."""
    global _clf
    if _clf is None:
        model_path = BEST_MODEL_DIR / "best_clf.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No production model found at {model_path}. Run train.py then evaluate.py first."
            )
        _clf = joblib.load(model_path)
        print(f"Loaded production model from {model_path}")
    return _clf


def predict_text(text: str, clf=None) -> dict:
    """Predict if text is AI or Human generated.

    Args:
        text: Input text to classify.
        clf: Optional classifier override. If None, loads the saved production model.

    Returns:
        dict with label, ai_prob, and human_prob.
    """
    if clf is None:
        clf = _load_clf()

    emb = get_embeddings([text])
    emb_np = emb.cpu().numpy()
    probs = clf.predict_proba(emb_np)[0]

    return {
        "label": "AI" if probs[1] > 0.5 else "Human",
        "ai_prob": round(float(probs[1]) * 100, 2),
        "human_prob": round(float(probs[0]) * 100, 2),
    }
