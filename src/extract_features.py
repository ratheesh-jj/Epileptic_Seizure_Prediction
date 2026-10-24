"""
Fast Vectorized EEG Feature Extraction for Baseline A (Random Forest).
Computes time-domain and frequency-domain features across all 18 channels simultaneously:
  - Mean, Standard Deviation, Variance, RMS, Peak-to-Peak Amplitude, Energy
  - Hjorth Parameters: Activity, Mobility, Complexity
  - Spectral Band Powers (Delta, Theta, Alpha, Beta, Gamma) via vectorized Welch PSD
  - Relative Band Powers & Normalized Spectral Entropy
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import simpson
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"


def extract_features_vectorized(window: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Computes all features across all channels in a single vectorized pass.
    window shape: (n_channels, n_samples) = (18, 7680)
    Returns: 1D float32 array of length (18 * 21 = 378).
    """
    # 1. Time-domain statistics across axis=-1
    mean_val = np.mean(window, axis=-1)
    std_val = np.std(window, axis=-1)
    var_val = np.var(window, axis=-1)
    rms_val = np.sqrt(np.mean(window ** 2, axis=-1))
    ptp_val = np.ptp(window, axis=-1)
    energy_val = np.sum(window ** 2, axis=-1)

    # 2. Vectorized Hjorth parameters
    d1 = np.diff(window, axis=-1)
    var_d1 = np.var(d1, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var_d2 = np.var(d2, axis=-1)

    activity = var_val
    mobility = np.sqrt(var_d1 / np.maximum(var_val, 1e-12))
    mobility_d1 = np.sqrt(var_d2 / np.maximum(var_d1, 1e-12))
    complexity = mobility_d1 / np.maximum(mobility, 1e-12)

    # 3. Vectorized Welch PSD across all channels simultaneously
    nperseg = min(window.shape[-1], int(sfreq * 2))
    freqs, psd = welch(window, fs=sfreq, nperseg=nperseg, axis=-1)  # psd shape: (18, n_freqs)

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    band_powers = []
    for (low, high) in bands.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        if np.any(idx):
            bp = simpson(psd[:, idx], freqs[idx], axis=-1)
        else:
            bp = np.zeros(window.shape[0])
        band_powers.append(np.maximum(0.0, bp))

    band_powers = np.stack(band_powers, axis=1)  # (18, 5)

    total_idx = np.logical_and(freqs >= 0.5, freqs <= 40.0)
    total_power = np.maximum(1e-12, simpson(psd[:, total_idx], freqs[total_idx], axis=-1))  # (18,)
    rel_powers = band_powers / total_power[:, np.newaxis]  # (18, 5)

    # Vectorized Spectral Entropy
    psd_sub = psd[:, total_idx]
    sum_psd = np.sum(psd_sub, axis=-1, keepdims=True) + 1e-12
    psd_norm = psd_sub / sum_psd
    psd_safe = np.maximum(psd_norm, 1e-12)
    spec_entropy = -np.sum(psd_norm * np.log2(psd_safe), axis=-1)
    n_bins = int(np.sum(total_idx))
    max_entropy = np.log2(n_bins) if n_bins > 1 else 1.0
    norm_spec_entropy = spec_entropy / max(max_entropy, 1.0)  # (18,)

    # 4. Band power ratios per channel
    theta_beta = rel_powers[:, 1] / (rel_powers[:, 3] + 1e-6)
    delta_alpha = rel_powers[:, 0] / (rel_powers[:, 2] + 1e-6)
    slow_fast = (rel_powers[:, 1] + rel_powers[:, 2]) / (rel_powers[:, 3] + rel_powers[:, 4] + 1e-6)
    band_ratios = np.column_stack([theta_beta, delta_alpha, slow_fast])  # (18, 3)

    # 5. Spatial cross-channel correlation & synchrony
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(window)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Mean connectivity per channel with all other channels (18,)
    n_ch = window.shape[0]
    ch_connectivity = (np.sum(corr_matrix, axis=-1) - 1.0) / max(n_ch - 1, 1)

    # Global cross-channel synchrony statistics
    triu_idx = np.triu_indices(n_ch, k=1)
    global_sync_mean = float(np.mean(corr_matrix[triu_idx]))
    global_sync_std = float(np.std(corr_matrix[triu_idx]))

    # Hemispheric energy asymmetry (left 8 channels: 0..7, right 8 channels: 8..15)
    left_energy = float(np.mean(energy_val[0:8])) if n_ch >= 16 else 0.0
    right_energy = float(np.mean(energy_val[8:16])) if n_ch >= 16 else 0.0
    asym_energy = float((left_energy - right_energy) / (left_energy + right_energy + 1e-6))

    # Combine 25 features per channel:
    # [mean, std, var, rms, ptp, energy, act, mob, comp, 5 band, 5 rel band, 3 ratios, entropy, total_p, connectivity]
    ch_features = np.column_stack([
        mean_val, std_val, var_val, rms_val, ptp_val, energy_val,
        activity, mobility, complexity,
        band_powers,
        rel_powers,
        band_ratios,
        norm_spec_entropy,
        total_power,
        ch_connectivity,
    ])  # shape: (18, 25)

    global_features = np.array([global_sync_mean, global_sync_std, asym_energy], dtype=np.float32)
    full_vector = np.concatenate([ch_features.flatten(), global_features])
    return full_vector.astype(np.float32)


# Alias for backward compatibility
extract_features_from_window = extract_features_vectorized


def extract_dataset_features(
    meta_path: Path = DEFAULT_OUT / "window_metadata.csv",
    out_dir: Path = DEFAULT_OUT,
    max_windows: int = None,
):
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {meta_path}. Run prepare_dataset.py first.")

    df_meta = pd.read_csv(meta_path)
    if max_windows:
        df_meta = df_meta.iloc[:max_windows].copy()

    print(f"Extracting vectorized features for {len(df_meta)} windows...")

    feature_matrix = []
    labels = []
    kept_indices = []

    cached_path = None
    cached_x = None

    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting features"):
        npz_p = ROOT / row["npz_path"]
        if npz_p != cached_path:
            with np.load(npz_p) as npz:
                cached_x = npz["X"].astype(np.float32)
            cached_path = npz_p

        arr_idx = int(row["array_index"])
        win = cached_x[arr_idx]
        sfreq = float(row["sfreq"])

        feat_vec = extract_features_vectorized(win, sfreq)
        feature_matrix.append(feat_vec)
        labels.append(int(row["label"]))
        kept_indices.append(idx)

    X_feats = np.stack(feature_matrix).astype(np.float32)
    y_feats = np.array(labels, dtype=np.int64)

    out_npz = out_dir / "features.npz"
    np.savez_compressed(out_npz, X=X_feats, y=y_feats)

    sub_meta = df_meta.loc[kept_indices].copy()
    sub_meta["feature_vector_dim"] = X_feats.shape[1]
    sub_meta.to_csv(out_dir / "features_metadata.csv", index=False)

    print(f"\nVectorized feature extraction complete.")
    print(f"Features shape: {X_feats.shape} (windows x features)")
    print(f"Labels: Interictal={np.sum(y_feats == 0)}, Preictal={np.sum(y_feats == 1)}")
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_dir / 'features_metadata.csv'}")
    return X_feats, y_feats, sub_meta


def main():
    parser = argparse.ArgumentParser(description="Extract statistical and spectral features from EEG windows")
    parser.add_argument("--meta", type=str, default=str(DEFAULT_OUT / "window_metadata.csv"), help="Path to window_metadata.csv")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--max-windows", type=int, default=None, help="Cap number of windows for testing")
    args = parser.parse_args()

    extract_dataset_features(
        meta_path=Path(args.meta),
        out_dir=Path(args.out_dir),
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
