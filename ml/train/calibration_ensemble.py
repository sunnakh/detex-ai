# part_08_calibration_ensemble.py
# Calibrates each classifier → builds weighted ensemble → evaluates → saves
# OUTPUT: calibrated .joblib models + ensemble_weights.json
# REQUIRES: part_06 (.joblib classifiers) + part_05 (embeddings)

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

import config

SEP = "- " * 50

CLASSIFIER_NAMES = ["logistic_regression", "svm", "xgboost", "lightgbm"]

# SVM was already wrapped in CalibratedClassifierCV in part_06 — skip re-calibration
SKIP_CALIBRATION = {"svm"}


if __name__ == "__main__":
    os.makedirs("./artifacts/models", exist_ok=True)
    os.makedirs("./artifacts/eval", exist_ok=True)

    # ---------------------------------------------------------------------------
    # load embeddings produced by part_05
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading embeddings: ...")

    X_train = np.load("./artifacts/clf_X_train.npy")
    y_train = np.load("./artifacts/clf_y_train.npy")
    X_test = np.load("./artifacts/clf_X_test.npy")
    y_test = np.load("./artifacts/clf_y_test.npy")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ---------------------------------------------------------------------------
    # step 1: calibrate each classifier and save as _calibrated.joblib
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Calibrating classifiers: ...")

    calibrated_probas = {}

    for name in CLASSIFIER_NAMES:
        raw_path = f"./artifacts/models/{name}.joblib"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing {raw_path} — run part_06 first.")

        clf = joblib.load(raw_path)

        if name in SKIP_CALIBRATION:
            print(f"  {name}: skipping (already calibrated in part_06)")
            calibrated_probas[name] = clf.predict_proba(X_test)[:, 1]
            # Save under calibrated name for consistent loading in backend
            joblib.dump(clf, f"./artifacts/models/{name}_calibrated.joblib", compress=3)
            continue

        # Isotonic regression — better than Platt for datasets > 1k samples
        cal_clf = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
        cal_clf.fit(X_train, y_train)
        calibrated_probas[name] = cal_clf.predict_proba(X_test)[:, 1]

        save_path = f"./artifacts/models/{name}_calibrated.joblib"
        joblib.dump(cal_clf, save_path, compress=3)
        print(f"  {name}: calibrated → saved to {save_path}")

    # ---------------------------------------------------------------------------
    # step 2: derive ensemble weights from ROC-AUC
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Deriving ensemble weights from ROC-AUC: ...")

    raw_aucs = {
        name: roc_auc_score(y_test, proba) for name, proba in calibrated_probas.items()
    }
    total_auc = sum(raw_aucs.values())
    weights = {name: round(auc / total_auc, 4) for name, auc in raw_aucs.items()}

    for name, w in weights.items():
        print(f"  {name:<25}: weight={w:.4f}  (ROC-AUC={raw_aucs[name]:.4f})")

    # ---------------------------------------------------------------------------
    # step 3: ensemble prediction
    # ---------------------------------------------------------------------------

    ensemble_proba = sum(
        calibrated_probas[name] * weights[name] for name in CLASSIFIER_NAMES
    )
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    # ---------------------------------------------------------------------------
    # step 4: compare ensemble vs individual classifiers
    # ---------------------------------------------------------------------------

    print(SEP)
    print(f"{'Model':<25} {'Acc':>6} {'F1':>6} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 57)

    eval_summary = {}
    for name in CLASSIFIER_NAMES:
        proba = calibrated_probas[name]
        pred = (proba >= 0.5).astype(int)
        m = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
        }
        eval_summary[name] = m
        print(
            f"{name:<25} {m['accuracy']:>6} {m['f1']:>6} {m['roc_auc']:>8} {m['pr_auc']:>8}"
        )

    ensemble_metrics = {
        "accuracy": round(accuracy_score(y_test, ensemble_pred), 4),
        "f1": round(f1_score(y_test, ensemble_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, ensemble_proba), 4),
        "pr_auc": round(average_precision_score(y_test, ensemble_proba), 4),
    }
    eval_summary["ensemble"] = ensemble_metrics
    print("-" * 57)
    print(
        f"{'ENSEMBLE':<25} {ensemble_metrics['accuracy']:>6} {ensemble_metrics['f1']:>6} "
        f"{ensemble_metrics['roc_auc']:>8} {ensemble_metrics['pr_auc']:>8}"
    )
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=["Human", "AI"]))

    # ---------------------------------------------------------------------------
    # step 5: calibration before vs after plot
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Plotting calibration before vs after: ...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(CLASSIFIER_NAMES):
        ax = axes[idx]
        raw_clf = joblib.load(f"./artifacts/models/{name}.joblib")
        raw_proba = raw_clf.predict_proba(X_test)[:, 1]
        cal_proba = calibrated_probas[name]

        pt_r, pp_r = calibration_curve(y_test, raw_proba, n_bins=10)
        pt_c, pp_c = calibration_curve(y_test, cal_proba, n_bins=10)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        ax.plot(pp_r, pt_r, marker="o", label="Before calibration")
        ax.plot(pp_c, pt_c, marker="s", label="After calibration")
        ax.set_title(name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend()

    plt.suptitle("Calibration: Before vs After", fontsize=14)
    plt.tight_layout()
    fig.savefig(
        "./artifacts/eval/calibration_before_after.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # save weights and results
    # ---------------------------------------------------------------------------

    with open("./artifacts/ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    with open("./artifacts/eval/ensemble_results.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(SEP)
    print(f"Calibration & ensemble complete | {len(CLASSIFIER_NAMES)} classifiers")
    print("  ./artifacts/models/*_calibrated.joblib  (one per classifier)")
    print("  ./artifacts/ensemble_weights.json")
    print("  ./artifacts/eval/ensemble_results.json")
    print("  ./artifacts/eval/calibration_before_after.png")


