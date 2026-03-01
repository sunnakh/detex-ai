import joblib
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from ml.embeddings import load_embedder, get_embeddings, load_data

ARTIFACTS_DIR = Path("artifacts")
BEST_MODEL_DIR = ARTIFACTS_DIR / "jina-v5-finetuned"
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate():
    """Evaluate all trained models, pick best by F1, calibrate, and save."""
    
    # loading embedder
    load_embedder()

    # loading data splits
    X_train_texts, X_test_texts, y_train, y_test = load_data()

    # embedding test set
    print("Embedding test set... ")
    X_test = get_embeddings(X_test_texts).numpy()
    print(f"  Test embeddings: {X_test.shape}")

    # all trained candidate models
    candidate_files = [p for p in ARTIFACTS_DIR.glob("*.joblib")]
    if not candidate_files:
        raise FileNotFoundError(
            "No candidate models found in artifacts/. Run train.py first."
        )

    results = {}
    loaded_models = {}

    for path in sorted(candidate_files):
        name = path.stem
        clf = joblib.load(path)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "ROC AUC": round(roc_auc_score(y_test, y_prob), 4),
        }
        loaded_models[name] = clf

    # comparison table
    results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(results_df.to_string())
    print("=" * 60)

    # picking the best by F1
    best_name = results_df["F1"].idxmax()
    best_clf_raw = loaded_models[best_name]

    print(f"\nBest model: {best_name}")
    print(f"  F1: {results[best_name]['F1']:.4f}")

    #  calibrating best model with isotonic regression
    
    print(f"\nCalibrating {best_name}...")
    X_train = get_embeddings(X_train_texts).numpy()
    best_clf = CalibratedClassifierCV(best_clf_raw, method="isotonic", cv=5)
    best_clf.fit(X_train, y_train)

    # 8. Evaluate calibrated model
    y_pred_cal = best_clf.predict(X_test)
    y_prob_cal = best_clf.predict_proba(X_test)[:, 1]

    print(f"\nCalibrated model performance: ")
    cal_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred_cal), 4),
        "precision": round(precision_score(y_test, y_pred_cal), 4),
        "recall": round(recall_score(y_test, y_pred_cal), 4),
        "f1": round(f1_score(y_test, y_pred_cal), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob_cal), 4),
    }
    for k, v in cal_metrics.items():
        print(f"  {k:<12}: {v}")

    print("\nClassification Report: ")
    print(classification_report(y_test, y_pred_cal, target_names=["human", "ai"]))

    cm = confusion_matrix(y_test, y_pred_cal)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: human", "Actual: ai"],
        columns=["Predicted: human", "Predicted: ai"],
    )
    print("Confusion Matrix: ")
    print(cm_df.to_string())
    print()

    # saving calibrated best model as production model
    model_path = BEST_MODEL_DIR / "best_clf.joblib"
    joblib.dump(best_clf, model_path)
    print(f"Best model saved -> {model_path}")

    # 10. saving metadata
    metadata = {
        "model_name": best_name,
        "saved_at": datetime.utcnow().isoformat(),
        "metrics": cal_metrics,
        "calibration": "isotonic (cv=5)",
        "embedding_model": "jinaai/jina-embeddings-v5-text-small",
        "train_samples": len(X_train_texts),
        "test_samples": len(X_test_texts),
    }
    metadata_path = BEST_MODEL_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved -> {metadata_path}")

    return cal_metrics


if __name__ == "__main__":
    evaluate()
