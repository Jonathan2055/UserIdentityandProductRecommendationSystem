"""Shared training / decision logic for the two identity models (face and voice).

Both models answer the same question - "which team member is this, and are we sure?" -
so they use the same recipe:

* Candidate models: Logistic Regression (with feature scaling) and Random Forest.
* Honest validation: StratifiedGroupKFold grouped by the ORIGINAL file. All augmented
  copies of one photo/recording are always in the same fold, so the model is never
  tested on a rotated copy of a photo it trained on (that would inflate the scores).
* Metrics: accuracy, macro F1, log loss (+ confusion matrix), on out-of-fold predictions.
* Open-set rejection (needed for the "unauthorized attempt"):
    1. confidence threshold - the top probability must be >= threshold
    2. novelty detector     - LocalOutlierFactor flags inputs that look like nobody
                              in the training data (a stranger)
  A closed-set classifier always names *some* member, so without these two checks a
  stranger would always be "recognised" as someone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_THRESHOLD, RANDOM_STATE

META_COLS = {"member", "expression", "phrase", "augmentation", "source_image", "source_file", "face_detected"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def candidate_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE),
    }


def novelty_detector(n_rows: int):
    return make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=max(2, min(10, n_rows - 1)), novelty=True))


def _n_splits(y: pd.Series, groups: pd.Series) -> int:
    """As many folds as the member with the fewest original files allows (2..5)."""
    per_class = pd.DataFrame({"y": y, "g": groups}).groupby("y")["g"].nunique()
    return int(max(2, min(5, per_class.min())))


def cross_validate_model(model, X, y, groups, classes):
    """Out-of-fold probabilities (n_rows x n_classes) + novelty flags for every row."""
    cv = StratifiedGroupKFold(n_splits=_n_splits(y, groups), shuffle=True, random_state=RANDOM_STATE)
    proba = np.zeros((len(y), len(classes)))
    is_novel = np.zeros(len(y), dtype=bool)
    for tr, te in cv.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        cols = [classes.index(c) for c in m.classes_]          # a fold can miss a class
        proba[np.ix_(te, cols)] = m.predict_proba(X.iloc[te])
        nov = novelty_detector(len(tr)).fit(X.iloc[tr])
        is_novel[te] = nov.predict(X.iloc[te]) == -1
    return proba, is_novel


def scores(y_true, proba, classes) -> dict:
    pred = np.array(classes)[proba.argmax(1)]
    p = np.clip(proba, 1e-6, 1)
    p = p / p.sum(1, keepdims=True)
    return {"accuracy": accuracy_score(y_true, pred),
            "f1_macro": f1_score(y_true, pred, average="macro"),
            "log_loss": log_loss(y_true, p, labels=classes)}


def choose_threshold(y_true, proba, classes) -> float:
    """Lowest confidence still seen on 90% of CORRECT out-of-fold guesses.

    i.e. we accept ~90% of genuine attempts and reject anything less confident.
    Clipped to [0.40, 0.85]: never below 0.40 (with 4 members, 0.25 is a pure guess),
    never so high that real members get locked out.
    """
    pred = np.array(classes)[proba.argmax(1)]
    correct_conf = proba.max(1)[pred == np.asarray(y_true)]
    if len(correct_conf) == 0:
        return DEFAULT_THRESHOLD
    return float(np.clip(np.quantile(correct_conf, 0.10), 0.40, 0.85))


def train_identity_model(df: pd.DataFrame, group_col: str, verbose: bool = True) -> dict:
    feats = feature_columns(df)
    X, y, groups = df[feats], df["member"], df[group_col]
    classes = sorted(y.unique())
    if len(classes) < 2:
        raise ValueError("Need data from at least 2 members to train an identity model.")

    # 1) compare candidates with grouped cross-validation
    results, oof = [], {}
    for name, model in candidate_models().items():
        proba, novel = cross_validate_model(model, X, y, groups, classes)
        oof[name] = (proba, novel)
        results.append({"model": name, **scores(y, proba, classes)})
    table = pd.DataFrame(results).sort_values(["f1_macro", "log_loss"], ascending=[False, True])
    best = table.iloc[0]["model"]
    proba, novel = oof[best]

    # 2) decision rule, measured on the same out-of-fold predictions
    threshold = choose_threshold(y, proba, classes)
    pred = np.array(classes)[proba.argmax(1)]
    accepted = (proba.max(1) >= threshold) & ~novel
    genuine_accept = float(np.mean(accepted & (pred == y.values)))
    wrong_member_accept = float(np.mean(accepted & (pred != y.values)))

    # 3) refit on everything for the final saved model
    final = clone(candidate_models()[best]).fit(X, y)
    train_scores = scores(y, final.predict_proba(X), list(final.classes_))
    nov = novelty_detector(len(X)).fit(X)

    bundle = {
        "model": final, "novelty": nov, "features": feats, "classes": list(final.classes_),
        "threshold": threshold, "model_name": best,
        "cv_table": table.reset_index(drop=True),
        "cv_scores": table.iloc[0].drop("model").to_dict(),
        "train_scores": train_scores,
        "oof_proba": proba, "oof_true": y.values, "oof_novel": novel,
        "confusion": confusion_matrix(y, pred, labels=classes),
        "genuine_accept_rate": genuine_accept,
        "wrong_member_accept_rate": wrong_member_accept,
        "n_rows": len(df), "n_sources": groups.nunique(), "n_folds": _n_splits(y, groups),
    }
    if verbose:
        print_summary(bundle)
    return bundle


def print_summary(b: dict) -> None:
    print(f"Rows: {b['n_rows']} from {b['n_sources']} original files | {b['n_folds']}-fold grouped CV")
    print(b["cv_table"].round(3).to_string(index=False))
    print(f"\nChosen: {b['model_name']}")
    print(f"  CV    -> acc {b['cv_scores']['accuracy']:.2f} | F1 {b['cv_scores']['f1_macro']:.2f} | loss {b['cv_scores']['log_loss']:.2f}")
    print(f"  Train -> acc {b['train_scores']['accuracy']:.2f} | F1 {b['train_scores']['f1_macro']:.2f} | loss {b['train_scores']['log_loss']:.2f}")
    print(f"  Confidence threshold: {b['threshold']:.2f}")
    print(f"  CV genuine attempts accepted: {b['genuine_accept_rate']:.0%} | accepted as the WRONG member: {b['wrong_member_accept_rate']:.0%}")


def save_bundle(bundle: dict, path) -> None:
    """Save only what the app needs (not the big CV arrays)."""
    import joblib
    keep = ["model", "novelty", "features", "classes", "threshold", "model_name",
            "cv_scores", "train_scores", "genuine_accept_rate", "wrong_member_accept_rate"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: bundle[k] for k in keep}, path)


def identify(bundle: dict, X_row: pd.DataFrame) -> dict:
    """Decision for ONE input. Returns who it is (or None) and why."""
    X_row = X_row[bundle["features"]]
    proba = bundle["model"].predict_proba(X_row)[0]
    best = int(proba.argmax())
    member, conf = bundle["classes"][best], float(proba[best])
    novel = bool(bundle["novelty"].predict(X_row)[0] == -1)

    if novel:
        reason = "input does not look like any enrolled team member (novelty check)"
    elif conf < bundle["threshold"]:
        reason = f"confidence {conf:.2f} is below the threshold {bundle['threshold']:.2f}"
    else:
        reason = "ok"
    return {"accepted": reason == "ok", "member": member if reason == "ok" else None,
            "best_guess": member, "confidence": conf, "novel": novel, "reason": reason,
            "probabilities": dict(zip(bundle["classes"], proba.round(3)))}
