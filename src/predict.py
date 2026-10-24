"""
Inference and Prediction Pipeline for Epileptic Seizure Prediction.
Loads a trained model checkpoint (CNN-LSTM, 1D-CNN, or Random Forest),
preprocesses continuous EDF recordings using standard 18 channels,
computes preictal probabilities, applies alarm persistence filters,
and outputs predicted seizure warning events.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocess_eeg import (
    STANDARD_CHANNELS,
    preprocess_recording,
    robust_normalize,
)
from train_cnn_lstm import SpatialTemporalCNNLSTM
from train_cnn import ConvNet1D
from extract_features import extract_features_from_window

DEFAULT_MODELS = ROOT / "models"


def predict_recording(
    edf_path: Path,
    model_checkpoint: Path = DEFAULT_MODELS / "cnn_baseline.pt",
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    alarm_threshold: float = 0.5,
    persistence_k: int = 2,
) -> pd.DataFrame:
    """
    Runs end-to-end inference on continuous EDF recording.
    Auto-detects model type:
      - .joblib: RandomForest / Gradient Boosting Pipeline with feature extraction
      - .pt with seq_len: Spatial-Temporal CNN-LSTM
      - .pt without seq_len: 1D-CNN Baseline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess recording
    data, sfreq, _ = preprocess_recording(edf_path, target_channels=STANDARD_CHANNELS)
    n_samples = data.shape[1]
    samples_per_win = int(round(window_sec * sfreq))
    samples_per_stride = int(round(stride_sec * sfreq))

    windows = []
    starts = []
    ends = []

    # Recording-level baseline statistics
    rec_med = np.median(data, axis=-1, keepdims=True)
    rec_mad = np.median(np.abs(data - rec_med), axis=-1, keepdims=True) * 1.4826
    rec_std = np.std(data, axis=-1, keepdims=True)
    rec_scale = np.where(rec_mad > 1e-6, rec_mad, np.where(rec_std > 1e-6, rec_std, 1.0))

    for s in range(0, n_samples - samples_per_win + 1, samples_per_stride):
        e = s + samples_per_win
        win_raw = data[:, s:e]
        win_norm = robust_normalize(win_raw, ref_med=rec_med, ref_scale=rec_scale)
        windows.append(win_norm)
        starts.append(round(s / sfreq, 2))
        ends.append(round(e / sfreq, 2))

    X_all = np.stack(windows).astype(np.float32)  # (N, C, S)
    print(f"Generated {len(X_all)} windows from {edf_path.name}")

    probs = []
    aligned_starts = starts
    aligned_ends = ends

    if model_checkpoint.suffix == ".joblib":
        print(f"Loaded tabular model pipeline from {model_checkpoint}")
        model = joblib.load(model_checkpoint)
        features = [extract_features_from_window(w, sfreq) for w in X_all]
        X_feats = np.stack(features).astype(np.float32)
        probs = model.predict_proba(X_feats)[:, 1].tolist()

    elif model_checkpoint.suffix == ".pt":
        checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
        in_channels = checkpoint.get("in_channels", 18)
        seq_len = checkpoint.get("seq_len", None)

        if seq_len is not None:
            # CNN-LSTM Sequence Model
            print(f"Loaded CNN-LSTM model from {model_checkpoint} (seq_len={seq_len})")
            model = SpatialTemporalCNNLSTM(in_channels=in_channels).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            if len(X_all) < seq_len:
                print(f"Recording has only {len(X_all)} windows, fewer than required seq_len={seq_len}.")
                return pd.DataFrame()

            for end_idx in range(seq_len - 1, len(X_all)):
                sub_seq = X_all[end_idx - seq_len + 1 : end_idx + 1]
                seq_tensor = torch.from_numpy(sub_seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit = model(seq_tensor)
                    prob = torch.sigmoid(logit).item()
                probs.append(prob)

            aligned_starts = starts[seq_len - 1 :]
            aligned_ends = ends[seq_len - 1 :]
        else:
            # 1D-CNN Baseline
            print(f"Loaded 1D-CNN baseline model from {model_checkpoint}")
            model = ConvNet1D(in_channels=in_channels).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            with torch.no_grad():
                tensor_x = torch.from_numpy(X_all).to(device)
                logits = model(tensor_x)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
    else:
        raise ValueError(f"Unsupported model format: {model_checkpoint}")

    df_out = pd.DataFrame({
        "edf_name": edf_path.name,
        "start_sec": aligned_starts,
        "end_sec": aligned_ends,
        "start_min": [round(s / 60.0, 2) for s in aligned_starts],
        "end_min": [round(e / 60.0, 2) for e in aligned_ends],
        "preictal_probability": [round(p, 4) for p in probs],
        "raw_positive": [int(p >= alarm_threshold) for p in probs],
    })

    # Persistence filter
    consecutive = 0
    alarms = []
    for pos in df_out["raw_positive"]:
        if pos == 1:
            consecutive += 1
        else:
            consecutive = 0
        alarms.append(1 if consecutive >= persistence_k else 0)

    df_out["alarm_triggered"] = alarms

    n_alarms = (df_out["alarm_triggered"] == 1).sum()
    print(f"\nInference Complete for {edf_path.name}:")
    print(f"  - Windows scored: {len(df_out)}")
    print(f"  - Mean preictal probability: {df_out['preictal_probability'].mean():.4f}")
    print(f"  - Windows triggering alarm (persistence >= {persistence_k}): {n_alarms}")

    if n_alarms > 0:
        alarm_starts = df_out[df_out["alarm_triggered"] == 1]["start_min"].tolist()
        print(f"  - Alarms active at times (minutes): {alarm_starts[:10]}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Predict seizure warnings on EDF recording")
    parser.add_argument("edf", type=str, help="Path to EDF recording")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODELS / "cnn_lstm_best.pt"), help="Model checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for alarm")
    parser.add_argument("--persistence", type=int, default=2, help="Consecutive positive windows required")
    parser.add_argument("--out-csv", type=str, default=None, help="Save predictions to CSV")
    args = parser.parse_args()

    edf_p = Path(args.edf)
    if not edf_p.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_p}")

    df_preds = predict_recording(
        edf_path=edf_p,
        model_checkpoint=Path(args.model),
        alarm_threshold=args.threshold,
        persistence_k=args.persistence,
    )


    if args.out_csv and len(df_preds):
        out_p = Path(args.out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(out_p, index=False)
        print(f"Saved predictions to {out_p}")


# Alias for backward compatibility
predict_recording_cnn_lstm = predict_recording


if __name__ == "__main__":
    main()
