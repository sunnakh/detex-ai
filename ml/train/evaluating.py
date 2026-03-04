# part_07_evaluate_classifiers.py
# Loads all .joblib classifiers → runs full evaluation → saves metrics + plots
# Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
# REQUIRES: part_05 (embeddings) + part_06 (trained .joblib classifiers)

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config

SEP = "- " * 50

CLASSIFIER_NAMES = ["logistic_regression", "svm", "xgboost", "lightgbm"]


if __name__ == "__main__":
    os.makedirs("./artifacts/eval", exist_ok=True)

    # ---------------------------------------------------------------------------
    # load test embeddings produced by part_05
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading test embeddings: ...")

    X_test = np.load("./artifacts/clf_X_test.npy")
    y_test = np.load("./artifacts/clf_y_test.npy")
    print(f"Test set: {X_test.shape}")

    # ---------------------------------------------------------------------------
    # evaluate each classifier
    # ---------------------------------------------------------------------------

    all_results = {}
    all_probas = {}

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
    ax_cal.plot(
        [0, 1], [0, 1], linestyle="--", label="Perfectly calibrated", color="gray"
    )

    for name in CLASSIFIER_NAMES:
        path = f"./artifacts/models/{name}.joblib"
        if not os.path.exists(path):
            print(f"⚠  Skipping {name} — not found at {path}. Run part_06 first.")
            continue

        print(SEP)
        print(f"Evaluating {name}: ...")
        clf = joblib.load(path)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        all_probas[name] = y_proba

        # ── Core metrics ───────────────────────────────────────────────────
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        }
        all_results[name] = metrics

        print(f"  Accuracy  : {metrics['accuracy']}")
        print(f"  Precision : {metrics['precision']}")
        print(f"  Recall    : {metrics['recall']}")
        print(f"  F1        : {metrics['f1']}")
        print(f"  ROC-AUC   : {metrics['roc_auc']}")
        print(f"  PR-AUC    : {metrics['pr_auc']}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

        # ── Confusion matrix ───────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_xticklabels(["Human", "AI"])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_yticklabels(["Human", "AI"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix — {name}")
        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.colorbar(im, ax=ax_cm)
        plt.tight_layout()
        fig_cm.savefig(f"./artifacts/eval/confusion_matrix_{name}.png", dpi=150)
        plt.close(fig_cm)

        # ── ROC curve ──────────────────────────────────────────────────────
        RocCurveDisplay.from_predictions(
            y_test,
            y_proba,
            name=f"{name} (AUC={metrics['roc_auc']})",
            ax=ax_roc,
        )

        # ── PR curve ───────────────────────────────────────────────────────
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba,
            name=f"{name} (AP={metrics['pr_auc']})",
            ax=ax_pr,
        )

        # ── Calibration curve ──────────────────────────────────────────────
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker="o", label=name)

    # ---------------------------------------------------------------------------
    # save combined plots
    # ---------------------------------------------------------------------------

    ax_roc.set_title("ROC Curves — All Classifiers")
    ax_roc.legend(loc="lower right")
    fig_roc.savefig("./artifacts/eval/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    ax_pr.set_title("Precision-Recall Curves — All Classifiers")
    ax_pr.legend(loc="upper right")
    fig_pr.savefig("./artifacts/eval/pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pr)

    ax_cal.set_title("Calibration Curves — All Classifiers (pre-calibration)")
    ax_cal.legend(loc="upper left")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Fraction of positives")
    fig_cal.savefig(
        "./artifacts/eval/calibration_curves.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_cal)

    # ---------------------------------------------------------------------------
    # summary table + save results JSON
    # ---------------------------------------------------------------------------

    print(SEP)
    print(f"{'Model':<25} {'Acc':>6} {'F1':>6} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 57)
    for name, m in all_results.items():
        print(
            f"{name:<25} {m['accuracy']:>6} {m['f1']:>6} {m['roc_auc']:>8} {m['pr_auc']:>8}"
        )

    with open("./artifacts/eval/classifier_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(SEP)
    print(f"Evaluation complete | {len(all_results)} classifiers")
    print("  ./artifacts/eval/classifier_results.json")
    print("  ./artifacts/eval/roc_curves.png")
    print("  ./artifacts/eval/pr_curves.png")
    print("  ./artifacts/eval/calibration_curves.png")
    print("  ./artifacts/eval/confusion_matrix_<name>.png  (x4)")
