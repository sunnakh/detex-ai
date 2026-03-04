# part_09_select_model.py
# Reads ensemble_results.json → ranks all candidates by ROC-AUC → picks winner
# winner is promoted to production: model_selection.json written to artifacts/
# OUTPUT: artifacts/model_selection.json
# REQUIRES: part_08 (calibration_ensemble.py)

import json
import os
import shutil

SEP = "- " * 50

CLASSIFIER_NAMES = ["logistic_regression", "svm", "xgboost", "lightgbm"]

# Primary ranking metric — change to "f1" or "pr_auc" to prefer those instead
RANK_BY = "roc_auc"


def _model_path(name: str) -> str | None:
    """Return the calibrated .joblib path for `name`, or None for the ensemble."""
    if name == "ensemble":
        return None
    cal = f"./artifacts/models/{name}_calibrated.joblib"
    raw = f"./artifacts/models/{name}.joblib"
    if os.path.exists(cal):
        return cal
    if os.path.exists(raw):
        return raw
    return None


def select_best(results: dict[str, dict]) -> tuple[str, dict]:
    """Return (winner_name, metrics_dict) ranked by RANK_BY descending."""
    ranked = sorted(results.items(), key=lambda kv: kv[1].get(RANK_BY, 0), reverse=True)
    return ranked[0]


if __name__ == "__main__":
    os.makedirs("./artifacts", exist_ok=True)

    # ---------------------------------------------------------------------------
    # load evaluation results from part_08
    # ---------------------------------------------------------------------------

    results_path = "./artifacts/eval/ensemble_results.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Missing {results_path} — run calibration_ensemble.py first."
        )

    print(SEP)
    print("Loading evaluation results: ...")

    with open(results_path) as f:
        results = json.load(f)

    # ---------------------------------------------------------------------------
    # print comparison table
    # ---------------------------------------------------------------------------

    print(SEP)
    print(f"{'Model':<28} {'Accuracy':>8} {'F1':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 64)

    all_names = CLASSIFIER_NAMES + ["ensemble"]
    for name in all_names:
        if name not in results:
            print(f"  ⚠  {name}: not in results — skipping")
            continue
        m = results[name]
        marker = (
            "  ← best"
            if name == max(results, key=lambda k: results[k].get(RANK_BY, 0))
            else ""
        )
        print(
            f"{name:<28} {m.get('accuracy', 0):>8.4f} {m.get('f1', 0):>8.4f}"
            f" {m.get('roc_auc', 0):>8.4f} {m.get('pr_auc', 0):>8.4f}{marker}"
        )

    # ---------------------------------------------------------------------------
    # pick winner
    # ---------------------------------------------------------------------------

    print(SEP)
    winner_name, winner_metrics = select_best(results)
    winner_path = _model_path(winner_name)

    print(f"Winner (ranked by {RANK_BY}): {winner_name}")
    print(f"  Accuracy  : {winner_metrics.get('accuracy', 0):.4f}")
    print(f"  F1        : {winner_metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC   : {winner_metrics.get('roc_auc', 0):.4f}")
    print(f"  PR-AUC    : {winner_metrics.get('pr_auc', 0):.4f}")

    # ---------------------------------------------------------------------------
    # copy winner to production_model.joblib (single classifiers only)
    # the ensemble has no single .joblib — backend assembles it from weights
    # ---------------------------------------------------------------------------

    production_model_path = None

    if winner_name != "ensemble" and winner_path:
        production_model_path = "./artifacts/production_model.joblib"
        shutil.copy2(winner_path, production_model_path)
        print(f"\n  Copied to: {production_model_path}")
    else:
        print(
            "\n  Winner is the ensemble — FastAPI backend uses ensemble_weights.json"
            "\n  to assemble the production model at request time."
        )

    # ---------------------------------------------------------------------------
    # save model_selection.json — loaded by threshold_tuning.py + FastAPI backend
    # ---------------------------------------------------------------------------

    selection = {
        "winner": winner_name,
        "ranked_by": RANK_BY,
        "metrics": winner_metrics,
        "production_model_path": production_model_path,
        "ensemble_weights_path": (
            "./artifacts/ensemble_weights.json" if winner_name == "ensemble" else None
        ),
        "component_models": (
            {n: f"./artifacts/models/{n}_calibrated.joblib" for n in CLASSIFIER_NAMES}
            if winner_name == "ensemble"
            else None
        ),
        "all_results": results,
        "note": (
            f"'{winner_name}' selected by highest {RANK_BY}. "
            "To change the ranking metric, edit RANK_BY in select_model.py."
        ),
    }

    out_path = "./artifacts/model_selection.json"
    with open(out_path, "w") as f:
        json.dump(selection, f, indent=2)

    print(SEP)
    print("Model selection complete")
    print(f"  {out_path}")
    if production_model_path:
        print(f"  {production_model_path}")
