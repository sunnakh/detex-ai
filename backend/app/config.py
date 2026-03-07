import os

# paths
_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "jina-v5-finetuned"
)
CLF_PATH = os.path.join(_ARTIFACTS_DIR, "best_clf.joblib")
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"

# File limits
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# Gemini Environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
