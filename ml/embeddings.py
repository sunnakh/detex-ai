import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv"
MAX_PER_CLASS = 200_000

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Use float16 on CUDA for faster inference + lower VRAM
dtype = torch.float16 if device.type == "cuda" else torch.float32

_tokenizer = None
_model = None


def load_embedder():
    """Lazily load the tokenizer and model into module-level singletons."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading embedder '{MODEL_NAME}' on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        _model.eval()
        print(
            f"  Embedder loaded ({sum(p.numel() for p in _model.parameters()) / 1e6:.1f}M params)"
        )


def load_data(
    max_per_class: int = MAX_PER_CLASS,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Load train.csv, balance classes, and return (X_train, X_test, y_train, y_test)."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"], errors="ignore")
    df = df.dropna(subset=["text", "source"])
    df["text"] = df["text"].astype(str)

    # Binary label: human=0, AI=1
    df["label"] = df["source"].apply(lambda x: 0 if str(x).lower() == "human" else 1)

    # Balanced sample (capped per class)
    n = min(max_per_class, df["label"].value_counts().min())
    human_df = df[df["label"] == 0].sample(n=n, random_state=random_state)
    ai_df = df[df["label"] == 1].sample(n=n, random_state=random_state)
    balanced_df = (
        pd.concat([human_df, ai_df])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    texts = balanced_df["text"].tolist()
    labels = balanced_df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"Total balanced samples : {len(texts)}")
    print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")
    print(f"Train -> Human: {y_train.count(0)}  AI: {y_train.count(1)}")
    print(f"Test  -> Human: {y_test.count(0)}   AI: {y_test.count(1)}")

    return X_train, X_test, y_train, y_test


def get_embeddings(texts: list[str], batch_size: int = 32) -> torch.Tensor:
    """Embed a list of texts. Calls load_embedder() if not already loaded."""
    load_embedder()
    _model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        encoded = _tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            output = _model(**encoded)

        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0)
