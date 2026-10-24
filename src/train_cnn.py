"""
Baseline B: 1D Multichannel Convolutional Neural Network (CNN) for Seizure Prediction.
Trained directly on standardized raw EEG windows (18 channels x 7680 samples).
Saves best model checkpoint to models/cnn_baseline.pt and metrics to outputs/cnn_evaluation.json.
"""

from pathlib import Path
import sys
import argparse
import random
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"
DEFAULT_MODELS = ROOT / "models"


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    """Memory-efficient Dataset that caches loaded NPZ files in RAM across shuffled batches."""
    def __init__(self, meta_df: pd.DataFrame, shared_cache: dict = None):
        self.meta = meta_df.reset_index(drop=True)
        self.cache = shared_cache if shared_cache is not None else {}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        p = str(ROOT / row["npz_path"])
        if p not in self.cache:
            with np.load(p) as npz:
                self.cache[p] = npz["X"].astype(np.float32)

        arr_idx = int(row["array_index"])
        x = self.cache[p][arr_idx]
        y = float(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class ConvNet1D(nn.Module):
    """Multichannel 1D CNN for EEG window classification."""
    def __init__(self, in_channels: int = 18, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out.squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
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


def train_cnn(
    meta_path: Path = DEFAULT_OUT / "window_metadata.csv",
    split_manifest: Path = DEFAULT_OUT / "split_manifest.json",
    models_dir: Path = DEFAULT_MODELS,
    out_dir: Path = DEFAULT_OUT,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
):
    seed_all(seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists() or not split_manifest.exists():
        raise FileNotFoundError("Run prepare_dataset.py and split_dataset.py first.")

    df_meta = pd.read_csv(meta_path)
    with open(split_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_pats = set(p.lower() for p in manifest["train"]["patients"])
    val_pats = set(p.lower() for p in manifest["validation"]["patients"])
    test_pats = set(p.lower() for p in manifest["test"]["patients"])

    pat_col = df_meta["patient_id"].str.lower()
    df_train = df_meta[pat_col.isin(train_pats)].copy()
    df_val = df_meta[pat_col.isin(val_pats)].copy()
    df_test = df_meta[pat_col.isin(test_pats)].copy()

    print("=" * 60)
    print("BASELINE B: 1D CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("=" * 60)
    print(f"Train set: {len(df_train)} windows (Preictal: {(df_train['label']==1).sum()})")
    print(f"Val set:   {len(df_val)} windows (Preictal: {(df_val['label']==1).sum()})")
    print(f"Test set:  {len(df_test)} windows (Preictal: {(df_test['label']==1).sum()})")

    ds_train = WindowDataset(df_train)
    ds_val = WindowDataset(df_val)
    ds_test = WindowDataset(df_test)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Compute class imbalance weighting
    y_train = df_train["label"].to_numpy()
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight_val = float(n_neg / max(n_pos, 1))
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    in_channels = int(df_train.iloc[0]["n_channels"])
    model = ConvNet1D(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_score = -float("inf")
    best_opt_thresh = 0.5
    best_model_path = models_dir / "cnn_baseline.pt"
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            n_samples += len(y)
        train_loss /= max(n_samples, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        val_probs = []
        val_true = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(y.cpu().numpy())
        val_loss /= max(n_val, 1)

        val_probs = np.array(val_probs)
        val_true = np.array(val_true)
        val_m = compute_metrics(val_true, val_probs) if len(val_true) else {}
        val_score = float(val_m.get("balanced_accuracy", 0.0))

        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_bal_acc": val_m.get("balanced_accuracy"),
            "val_recall": val_m.get("sensitivity_recall"),
            "val_auroc": val_m.get("auroc"),
        })

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Bal Acc: {val_m.get('balanced_accuracy')} | Val Rec: {val_m.get('sensitivity_recall')} | "
              f"Val AUROC: {val_m.get('auroc')}", flush=True)

        if val_score > best_score:
            best_score = val_score
            best_opt_thresh = find_optimal_threshold(val_true, val_probs)
            torch.save({
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "epoch": ep,
                "val_loss": val_loss,
                "val_bal_acc": val_score,
                "optimal_threshold": best_opt_thresh,
            }, best_model_path)

    # Evaluate best checkpoint on Held-out Test Set
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_probs = []
    test_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            test_probs.extend(probs)
            test_true.extend(y.numpy())

    test_probs = np.array(test_probs)
    test_true = np.array(test_true)
    test_metrics = compute_metrics(test_true, test_probs, threshold=0.5) if len(test_true) else {}
    test_metrics_opt = compute_metrics(test_true, test_probs, threshold=best_opt_thresh) if len(test_true) else {}

    results = {
        "model": "1D-CNN",
        "train_patients": sorted(list(train_pats)),
        "val_patients": sorted(list(val_pats)),
        "test_patients": sorted(list(test_pats)),
        "hyperparameters": {"in_channels": in_channels, "epochs": epochs, "batch_size": batch_size, "lr": lr},
        "optimal_threshold": best_opt_thresh,
        "history": history,
        "test_metrics": test_metrics,
        "test_metrics_tuned_threshold": test_metrics_opt,
    }

    out_json = out_dir / "cnn_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTEST METRICS (Default threshold 0.5):")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nTEST METRICS (Tuned threshold {best_opt_thresh:.3f}):")
    print(json.dumps(test_metrics_opt, indent=2))
    print(f"\nBest model saved to {best_model_path}")
    print(f"Results saved to {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train 1D CNN baseline for seizure prediction")
    parser.add_argument("--meta", type=str, default=str(DEFAULT_OUT / "window_metadata.csv"), help="Path to window_metadata.csv")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_OUT / "split_manifest.json"), help="Path to split_manifest.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_cnn(
        meta_path=Path(args.meta),
        split_manifest=Path(args.manifest),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
