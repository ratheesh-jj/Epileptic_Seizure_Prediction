"""
Patient-Specific (Personalized) Seizure Prediction Module.
Clinically, seizure onset zones and preictal electrophysiology (voltage attenuation vs.
spike-and-wave amplification) vary fundamentally between individual epilepsy patients.

This module evaluates patient-calibrated seizure prediction across each patient
using stratified K-fold cross-validation, training on the patient's individual EEG dynamics
with the enhanced 453-dimensional spectral, temporal, and spatial synchrony feature space.
"""

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"
DEFAULT_MODELS = ROOT / "models"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None

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


def evaluate_personalized_models(
    features_npz: Path = DEFAULT_OUT / "features.npz",
    meta_csv: Path = DEFAULT_OUT / "features_metadata.csv",
    out_dir: Path = DEFAULT_OUT,
    models_dir: Path = DEFAULT_MODELS,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not features_npz.exists() or not meta_csv.exists():
        raise FileNotFoundError("Run prepare_dataset.py and extract_features.py first.")

    with np.load(features_npz) as d:
        X_all = d["X"]
        y_all = d["y"]

    df_meta = pd.read_csv(meta_csv)
    patients = sorted(df_meta["patient_id"].unique())

    print("=" * 70)
    print("PATIENT-SPECIFIC (PERSONALIZED) SEIZURE PREDICTION EVALUATION")
    print("=" * 70)

    summary_rows = []
    patient_results = {}

    for p in patients:
        p_mask = (df_meta["patient_id"] == p).to_numpy()
        X_p = X_all[p_mask]
        y_p = y_all[p_mask]

        n_pos = int((y_p == 1).sum())
        n_neg = int((y_p == 0).sum())

        if n_pos < n_splits or n_neg < n_splits:
            print(f"Skipping {p}: insufficient samples for {n_splits}-fold CV (pos={n_pos}, neg={n_neg})")
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_metrics = []

        for tr_idx, te_idx in skf.split(X_p, y_p):
            pipe = Pipeline([
                ("scaler", RobustScaler()),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=2,
                )),
            ])
            pipe.fit(X_p[tr_idx], y_p[tr_idx])
            probs = pipe.predict_proba(X_p[te_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            m = compute_metrics(y_p[te_idx], preds, probs)
            fold_metrics.append(m)

        # Average metrics across folds
        avg_auroc = round(float(np.mean([f["auroc"] for f in fold_metrics])), 4)
        avg_bal_acc = round(float(np.mean([f["balanced_accuracy"] for f in fold_metrics])), 4)
        avg_recall = round(float(np.mean([f["sensitivity_recall"] for f in fold_metrics])), 4)
        avg_spec = round(float(np.mean([f["specificity"] for f in fold_metrics])), 4)
        avg_f1 = round(float(np.mean([f["f1_score"] for f in fold_metrics])), 4)

        patient_results[p] = {
            "total_windows": len(y_p),
            "preictal_windows": n_pos,
            "interictal_windows": n_neg,
            "mean_auroc": avg_auroc,
            "mean_balanced_accuracy": avg_bal_acc,
            "mean_sensitivity_recall": avg_recall,
            "mean_specificity": avg_spec,
            "mean_f1_score": avg_f1,
            "fold_details": fold_metrics,
        }

        summary_rows.append({
            "Patient": p,
            "Windows": len(y_p),
            "Preictal": n_pos,
            "Interictal": n_neg,
            "AUROC": f"{avg_auroc:.4f}",
            "Balanced Acc": f"{avg_bal_acc * 100:.1f}%",
            "Sensitivity (Recall)": f"{avg_recall * 100:.1f}%",
            "Specificity": f"{avg_spec * 100:.1f}%",
            "F1-Score": f"{avg_f1:.4f}",
        })

        print(f"Patient {p}: AUROC={avg_auroc:.4f} | Balanced Acc={avg_bal_acc*100:.1f}% | "
              f"Sensitivity={avg_recall*100:.1f}% | Specificity={avg_spec*100:.1f}% (Windows: {len(y_p)})", flush=True)

    df_summary = pd.DataFrame(summary_rows)
    csv_path = out_dir / "personalized_metrics.csv"
    df_summary.to_csv(csv_path, index=False)

    json_path = out_dir / "personalized_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(patient_results, f, indent=2)

    print("\n" + "=" * 70)
    print("PERSONALIZED CLINICAL EVALUATION SUMMARY:")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary to: {csv_path}")
    print(f"Saved full details to: {json_path}")
    return patient_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate patient-specific personalized seizure prediction")
    parser.add_argument("--features", type=str, default=str(DEFAULT_OUT / "features.npz"))
    parser.add_argument("--meta", type=str, default=str(DEFAULT_OUT / "features_metadata.csv"))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_personalized_models(
        features_npz=Path(args.features),
        meta_csv=Path(args.meta),
        n_splits=args.splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
