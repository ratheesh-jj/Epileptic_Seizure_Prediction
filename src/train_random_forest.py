"""
Baseline A: Random Forest Classifier for Seizure Prediction on Extracted Features.
Trained on patient-independent feature splits with balanced class weighting.
Saves trained model to models/random_forest.joblib and outputs metrics to outputs/rf_evaluation.json.
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"
DEFAULT_MODELS = ROOT / "models"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Computes full set of binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    has_both = len(np.unique(y_true)) > 1
    auroc = float(roc_auc_score(y_true, y_prob)) if has_both else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both else None

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "sensitivity_recall": round(rec, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4) if auroc is not None else "N/A",
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A",
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Finds decision threshold maximizing balanced accuracy on validation set."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_thresh = 0.5
    best_bal_acc = -1.0
    for t in np.linspace(0.10, 0.90, 81):
        pred = (y_prob >= t).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_bal_acc:
            best_bal_acc = score
            best_thresh = float(round(t, 3))
    return best_thresh


def train_random_forest(
    features_npz: Path = DEFAULT_OUT / "features.npz",
    meta_csv: Path = DEFAULT_OUT / "features_metadata.csv",
    split_manifest: Path = DEFAULT_OUT / "split_manifest.json",
    models_dir: Path = DEFAULT_MODELS,
    out_dir: Path = DEFAULT_OUT,
    n_estimators: int = 100,
    max_depth: int = 15,
    seed: int = 42,
):
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_npz.exists() or not meta_csv.exists():
        raise FileNotFoundError(
            f"Extracted features not found at {features_npz}. Run extract_features.py first."
        )
    if not split_manifest.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {split_manifest}. Run split_dataset.py first."
        )

    with np.load(features_npz) as d:
        X = d["X"]
        y = d["y"]

    df_meta = pd.read_csv(meta_csv)
    with open(split_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_pats = set(p.lower() for p in manifest["train"]["patients"])
    val_pats = set(p.lower() for p in manifest["validation"]["patients"])
    test_pats = set(p.lower() for p in manifest["test"]["patients"])

    pat_col = df_meta["patient_id"].str.lower()
    train_mask = pat_col.isin(train_pats).to_numpy()
    val_mask = pat_col.isin(val_pats).to_numpy()
    test_mask = pat_col.isin(test_pats).to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("=" * 60)
    print("BASELINE A: OPTIMIZED RANDOM FOREST & GRADIENT BOOSTING")
    print("=" * 60)
    print(f"Features dimension: {X.shape[1]}")
    print(f"Train set: {len(y_train)} windows (Preictal: {(y_train==1).sum()}, Interictal: {(y_train==0).sum()})")
    print(f"Val set:   {len(y_val)} windows (Preictal: {(y_val==1).sum()}, Interictal: {(y_val==0).sum()})")
    print(f"Test set:  {len(y_test)} windows (Preictal: {(y_test==1).sum()}, Interictal: {(y_test==0).sum()})")

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both interictal and preictal samples.")

    # Build Pipeline with RobustScaler to prevent scale leakage across patients
    rf_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=4,
        )),
    ])

    print("\nFitting Random Forest Pipeline (RobustScaler + Balanced RF)...", flush=True)
    rf_pipeline.fit(X_train, y_train)

    # Evaluate RF on Validation Set to find optimal decision threshold
    val_prob = rf_pipeline.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    opt_threshold = find_optimal_threshold(y_val, val_prob)
    print(f"Optimal validation threshold tuned for Balanced Accuracy: {opt_threshold:.3f}")

    val_pred_default = (val_prob >= 0.5).astype(int) if len(X_val) else np.array([])
    val_metrics_default = compute_metrics(y_val, val_pred_default, val_prob) if len(X_val) else {}

    val_pred_opt = (val_prob >= opt_threshold).astype(int) if len(X_val) else np.array([])
    val_metrics_opt = compute_metrics(y_val, val_pred_opt, val_prob) if len(X_val) else {}

    # Evaluate RF on Held-out Test Set
    test_prob = rf_pipeline.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
    test_pred_default = (test_prob >= 0.5).astype(int) if len(X_test) else np.array([])
    test_metrics_default = compute_metrics(y_test, test_pred_default, test_prob) if len(X_test) else {}

    test_pred_opt = (test_prob >= opt_threshold).astype(int) if len(X_test) else np.array([])
    test_metrics_opt = compute_metrics(y_test, test_pred_opt, test_prob) if len(X_test) else {}

    # Benchmark: HistGradientBoostingClassifier
    print("\nFitting HistGradientBoostingClassifier Benchmark...")
    hgb_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("classifier", HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            max_iter=100,
        )),
    ])
    hgb_pipeline.fit(X_train, y_train)
    hgb_val_prob = hgb_pipeline.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    hgb_opt_thresh = find_optimal_threshold(y_val, hgb_val_prob)
    hgb_test_prob = hgb_pipeline.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
    hgb_test_pred = (hgb_test_prob >= hgb_opt_thresh).astype(int) if len(X_test) else np.array([])
    hgb_test_metrics = compute_metrics(y_test, hgb_test_pred, hgb_test_prob) if len(X_test) else {}

    # Save RF Pipeline model
    model_path = models_dir / "random_forest.joblib"
    joblib.dump(rf_pipeline, model_path)

    results = {
        "model": "RandomForest",
        "train_patients": sorted(list(train_pats)),
        "val_patients": sorted(list(val_pats)),
        "test_patients": sorted(list(test_pats)),
        "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth, "seed": seed},
        "optimal_threshold": opt_threshold,
        "validation_metrics": val_metrics_default,
        "validation_metrics_tuned_threshold": val_metrics_opt,
        "test_metrics": test_metrics_default,
        "test_metrics_tuned_threshold": test_metrics_opt,
        "hist_gradient_boosting_test_metrics": hgb_test_metrics,
    }

    out_json = out_dir / "rf_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nRANDOM FOREST TEST METRICS (Default threshold 0.5):")
    print(json.dumps(test_metrics_default, indent=2))
    print(f"\nRANDOM FOREST TEST METRICS (Tuned threshold {opt_threshold:.3f}):")
    print(json.dumps(test_metrics_opt, indent=2))
    print(f"\nHIST GRADIENT BOOSTING TEST METRICS (Tuned threshold {hgb_opt_thresh:.3f}):")
    print(json.dumps(hgb_test_metrics, indent=2))
    print(f"\nModel saved to {model_path}")
    print(f"Results saved to {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest baseline for seizure prediction")
    parser.add_argument("--features", type=str, default=str(DEFAULT_OUT / "features.npz"), help="Path to features.npz")
    parser.add_argument("--meta", type=str, default=str(DEFAULT_OUT / "features_metadata.csv"), help="Path to metadata CSV")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_OUT / "split_manifest.json"), help="Path to split_manifest.json")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_random_forest(
        features_npz=Path(args.features),
        meta_csv=Path(args.meta),
        split_manifest=Path(args.manifest),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
