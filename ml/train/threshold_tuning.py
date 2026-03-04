# part_10_threshold_tuning.py
# Loads the SELECTED production model → sweeps threshold → finds optimal cutoff
# Saves optimal_threshold.json for direct use by FastAPI backend
# REQUIRES: part_09 (model_selection.json) + part_08 (calibrated .joblib models)

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import config

SEP = "- " * 50

CLASSIFIER_NAMES = ["logistic_regression", "svm", "xgboost", "lightgbm"]


if __name__ == "__main__":
    os.makedirs("./artifacts/eval", exist_ok=True)

    # ---------------------------------------------------------------------------
    # load model_selection.json from part_09 to find the production model
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading model selection: ...")

    selection_path = "./artifacts/model_selection.json"
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            "Missing model_selection.json — run select_model.py first."
        )

    with open(selection_path) as f:
        selection = json.load(f)

    winner = selection["winner"]
    print(f"  Production model  : {winner}")
    print(f"  Ranked by         : {selection['ranked_by']}")
    print(f"  ROC-AUC           : {selection['metrics'].get('roc_auc', 0):.4f}")
    print(f"  F1                : {selection['metrics'].get('f1', 0):.4f}")

    X_test = np.load("./artifacts/clf_X_test.npy")
    y_test = np.load("./artifacts/clf_y_test.npy")

    # ---------------------------------------------------------------------------
    # build production probabilities: single model or ensemble
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading production model: ...")

    if winner == "ensemble":
        weights_path = selection.get(
            "ensemble_weights_path", "./artifacts/ensemble_weights.json"
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing {weights_path} — run part_08 first.")
        with open(weights_path) as f:
            weights = json.load(f)

        ensemble_proba = np.zeros(len(y_test))
        for name in CLASSIFIER_NAMES:
            cal_path = f"./artifacts/models/{name}_calibrated.joblib"
            raw_path = f"./artifacts/models/{name}.joblib"
            if os.path.exists(cal_path):
                path = cal_path
            elif os.path.exists(raw_path):
                print(f"  ⚠  {name}: calibrated model not found, using raw")
                path = raw_path
            else:
                raise FileNotFoundError(
                    f"No model found for {name}. Run part_06 and part_08 first."
                )
            clf = joblib.load(path)
            proba = clf.predict_proba(X_test)[:, 1]
            ensemble_proba += proba * weights[name]
            print(f"  {name}: weight={weights[name]:.4f}")
        print(f"  Ensemble assembled from {len(CLASSIFIER_NAMES)} calibrated models")

    else:
        # single production model selected by select_model.py
        prod_path = (
            selection.get("production_model_path")
            or f"./artifacts/models/{winner}_calibrated.joblib"
        )
        if not os.path.exists(prod_path):
            raise FileNotFoundError(f"Production model not found at {prod_path}")
        clf = joblib.load(prod_path)
        ensemble_proba = clf.predict_proba(X_test)[:, 1]
        print(f"  Loaded: {prod_path}")

    # ---------------------------------------------------------------------------
    # threshold sweep
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Sweeping thresholds: ...")

    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores, precisions, recalls, accuracies = [], [], [], []

    for t in thresholds:
        preds = (ensemble_proba >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds, zero_division=0))
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        accuracies.append(accuracy_score(y_test, preds))

    f1_scores = np.array(f1_scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    accuracies = np.array(accuracies)

    # ---------------------------------------------------------------------------
    # find optimal thresholds and print results
    # ---------------------------------------------------------------------------

    best_f1_idx = f1_scores.argmax()
    best_prec_idx = precisions.argmax()
    best_rec_idx = recalls.argmax()

    best_threshold = round(float(thresholds[best_f1_idx]), 2)

    print(SEP)
    print(f"Optimal threshold (max F1)       : {best_threshold}")
    print(f"  F1        : {f1_scores[best_f1_idx]:.4f}")
    print(f"  Precision : {precisions[best_f1_idx]:.4f}")
    print(f"  Recall    : {recalls[best_f1_idx]:.4f}")
    print(f"  Accuracy  : {accuracies[best_f1_idx]:.4f}")
    print(SEP)
    print(
        f"Max Precision threshold          : {thresholds[best_prec_idx]:.2f}"
        f"  →  P={precisions[best_prec_idx]:.4f}  R={recalls[best_prec_idx]:.4f}  F1={f1_scores[best_prec_idx]:.4f}"
    )
    print(
        f"Max Recall threshold             : {thresholds[best_rec_idx]:.2f}"
        f"  →  P={precisions[best_rec_idx]:.4f}  R={recalls[best_rec_idx]:.4f}  F1={f1_scores[best_rec_idx]:.4f}"
    )

    # ---------------------------------------------------------------------------
    # plot threshold vs metrics
    # ---------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, label="F1", linewidth=2)
    ax.plot(thresholds, precisions, label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", linewidth=2)
    ax.plot(thresholds, accuracies, label="Accuracy", linewidth=2, linestyle="--")
    ax.axvline(
        best_threshold,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"Best F1 threshold = {best_threshold}",
    )
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{winner} — Threshold vs Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("./artifacts/eval/threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # save threshold config — loaded directly by the FastAPI backend
    # ---------------------------------------------------------------------------

    result = {
        "production_model": winner,
        "optimal_threshold": best_threshold,
        "at_optimal_f1": {
            "f1": round(float(f1_scores[best_f1_idx]), 4),
            "precision": round(float(precisions[best_f1_idx]), 4),
            "recall": round(float(recalls[best_f1_idx]), 4),
            "accuracy": round(float(accuracies[best_f1_idx]), 4),
        },
        "max_precision_threshold": round(float(thresholds[best_prec_idx]), 2),
        "max_recall_threshold": round(float(thresholds[best_rec_idx]), 2),
        "production_model_path": selection.get("production_model_path"),
        "ensemble_weights_path": (
            selection.get("ensemble_weights_path") if winner == "ensemble" else None
        ),
        "component_models": (
            {n: f"./artifacts/models/{n}_calibrated.joblib" for n in CLASSIFIER_NAMES}
            if winner == "ensemble"
            else None
        ),
        "note": (
            "Use optimal_threshold for balanced production. "
            "Raise to max_precision_threshold to reduce false positives. "
            "Lower to max_recall_threshold to catch all AI text."
        ),
    }

    with open("./artifacts/optimal_threshold.json", "w") as f:
        json.dump(result, f, indent=2)

    print(SEP)
    print("Threshold tuning complete | ML pipeline ready")
    print("  ./artifacts/optimal_threshold.json  ← load this in FastAPI backend")
    print("  ./artifacts/eval/threshold_sweep.png")
