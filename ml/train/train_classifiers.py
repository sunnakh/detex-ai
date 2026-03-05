# part_06_train_classifiers.py
# Loads embedded data → trains LogReg, SVM, XGBoost, LightGBM → saves all as .joblib
# REQUIRES: part_05 to have saved the embedding arrays

import os

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import config

SEP = "- " * 50


def build_classifiers(scale_pos: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=config.SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=config.SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }


if __name__ == "__main__":
    os.makedirs("./artifacts/models", exist_ok=True)

    # ---------------------------------------------------------------------------
    # load embeddings produced by part_05
    # ---------------------------------------------------------------------------

    print(SEP)
    print("Loading embeddings: ...")

    for fname in ["clf_X_train.npy", "clf_y_train.npy"]:
        if not os.path.exists(f"./artifacts/{fname}"):
            raise FileNotFoundError(f"Missing ./artifacts/{fname} — run part_05 first.")

    X_train = np.load("./artifacts/clf_X_train.npy")
    y_train = np.load("./artifacts/clf_y_train.npy")
    print(f"Train set: {X_train.shape} | Labels: {y_train.shape}")

    # ── Class imbalance weight ─────────────────────────────────────────────
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos
    print(
        f"Class balance — Human: {n_neg:,} | AI: {n_pos:,} | scale_pos_weight: {scale_pos:.2f}"
    )

    # ---------------------------------------------------------------------------
    # train each classifier and save as .joblib
    # ---------------------------------------------------------------------------

    classifiers = build_classifiers(scale_pos)
    saved_paths = {}

    for name, clf in classifiers.items():
        print(SEP)
        print(f"Training {name}: ...")
        clf.fit(X_train, y_train)

        path = f"./artifacts/models/{name}.joblib"
        joblib.dump(
            clf, path, compress=3
        )  # compress=3: good balance of size vs load speed
        saved_paths[name] = path
        print(f"{name} saved → {path}")

    # ---------------------------------------------------------------------------
    # summary
    # ---------------------------------------------------------------------------

    print(SEP)
    print(f"All classifiers trained and saved | {len(saved_paths)} models\n")
    for name, path in saved_paths.items():
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {name:<25}: {path}  ({size_mb:.1f} MB)")
