import joblib
import json
import time
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from ml.embeddings import load_embedder, get_embeddings, load_data, device

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def build_candidates():
    """Build candidate classifiers matching the notebook configuration."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced"),
            method="sigmoid",
            cv=5,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,
            tree_method="hist",
            device=device.type,
            n_jobs=1 if device.type != "cpu" else -1,
            random_state=42,
        ),
    }


def train():
    """Full training pipeline: load data -> embed -> train all candidates -> save."""
    # 1. Load embedder
    load_embedder()

    # 2. Load + split data (balanced, 200k per class)
    X_train_texts, X_test_texts, y_train, y_test = load_data()

    # 3. Embed
    print("\nEmbedding train set...")
    t0 = time.time()
    X_train = get_embeddings(X_train_texts).numpy()
    print(f"  Train embeddings: {X_train.shape}")

    print("Embedding test set...")
    X_test = get_embeddings(X_test_texts).numpy()
    print(f"  Test  embeddings: {X_test.shape}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # 4. Train all candidates and save as joblib
    candidates = build_candidates()
    trained_models = {}

    for name, clf_model in candidates.items():
        print(f"\nTraining {name}...")
        start = time.time()
        clf_model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.2f}s")

        trained_models[name] = clf_model

        out_path = ARTIFACTS_DIR / f"{name}.joblib"
        joblib.dump(clf_model, out_path)
        print(f"  Saved -> {out_path}")

    # 5. Save training metadata
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "device": device.type,
        "embedding_model": "jinaai/jina-embeddings-v5-text-small",
        "train_samples": len(X_train_texts),
        "test_samples": len(X_test_texts),
        "models": list(candidates.keys()),
    }
    metadata_path = ARTIFACTS_DIR / "train_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved -> {metadata_path}")

    print(
        "\nAll models trained and saved! Run evaluate.py to compare and select the best."
    )


if __name__ == "__main__":
    train()
