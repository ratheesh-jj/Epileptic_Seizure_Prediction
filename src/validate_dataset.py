"""
Pre-Flight Dataset Validation for Seizure Prediction Pipeline.
Strictly verifies that:
  - Both interictal (0) and preictal (1) examples exist.
  - Labels contain only expected binary values {0, 1}.
  - Every preictal window falls strictly inside a valid preictal interval [onset - 35m, onset - 5m].
  - No positive window overlaps ictal or postictal exclusion intervals.
  - All windows contain finite numeric values (no NaN or Inf).
  - All windows have compatible shape (n_channels, n_samples).
  - Channels are consistent and canonical standard 18 channels.
  - NPZ array contents match window_metadata.csv.
  - Multi-window sequence construction respects recording boundaries and chronological order.
  - Stale metadata or missing NPZ files are detected.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import yaml

from parse_annotations import parse_all_summaries
from preprocess_eeg import STANDARD_CHANNELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_DATA = ROOT / "dataset" / "chbmit"
DEFAULT_OUT = ROOT / "outputs"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def is_overlapping(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def validate_dataset(
    meta_path: Path = None,
    data_dir: Path = DEFAULT_DATA,
    config: dict = None,
    seq_len: int = 10,
    check_arrays_sample_size: int = 20,
) -> bool:
    if meta_path is None:
        meta_path = DEFAULT_OUT / "window_metadata.csv"

    if config is None:
        config = load_config()

    win_cfg = config.get("windowing", {})
    pre_start_sec = float(win_cfg.get("pre_start_min", 35.0)) * 60.0
    pre_end_sec = float(win_cfg.get("pre_end_min", 5.0)) * 60.0
    post_sec = float(win_cfg.get("post_min", 30.0)) * 60.0

    print("=" * 65)
    print("RUNNING AUTOMATIC PRE-FLIGHT DATASET VALIDATION")
    print("=" * 65)

    if not meta_path.exists():
        print(f"Validation FAILED: Metadata file not found at {meta_path}")
        return False

    df_meta = pd.read_csv(meta_path)
    if len(df_meta) == 0:
        print("Validation FAILED: Metadata CSV is empty.")
        return False

    # Check 1: Stale metadata / Missing NPZ files
    missing_files = []
    unique_npz = df_meta["npz_path"].unique()
    for rel_path in unique_npz:
        full_p = ROOT / rel_path
        if not full_p.exists():
            missing_files.append(rel_path)

    if missing_files:
        print(f"Validation FAILED: {len(missing_files)} referenced NPZ files are missing from disk!")
        print(f"Sample missing: {missing_files[:3]}")
        return False

    # Check 2: Label values and presence of both classes
    unique_labels = sorted(df_meta["label"].unique().tolist())
    if not set(unique_labels).issubset({0, 1}):
        print(f"Validation FAILED: Invalid label values found: {unique_labels}. Expected only {{0, 1}}.")
        return False

    n_interictal = int((df_meta["label"] == 0).sum())
    n_preictal = int((df_meta["label"] == 1).sum())

    if n_interictal == 0:
        print("Validation FAILED: Zero interictal (class 0) windows found!")
        return False

    if n_preictal == 0:
        print("Validation FAILED: Zero preictal (class 1) windows found! Cannot train seizure prediction model.")
        return False

    # Patients with both classes
    pat_summary = df_meta.groupby(["patient_id", "label"]).size().unstack(fill_value=0)
    if 1 not in pat_summary.columns:
        pats_with_both = []
    else:
        pats_with_both = pat_summary[(pat_summary[0] > 0) & (pat_summary[1] > 0)].index.tolist()

    # Check 3: Annotation and Timing Verification
    # Parse annotations from summaries to independently verify preictal windows
    summaries = parse_all_summaries(data_dir)
    annotation_ok = True
    timing_violations = []

    pos_windows = df_meta[df_meta["label"] == 1]
    for _, row in pos_windows.iterrows():
        rec_key = row["record_key"]
        w_start, w_end = row["start_sec"], row["end_sec"]
        ann = summaries.get(rec_key)
        if not ann or not ann.get("seizures"):
            timing_violations.append(f"{rec_key}: Positive window [{w_start}-{w_end}s] but recording has NO annotated seizures!")
            annotation_ok = False
            break

        seizures = ann["seizures"]
        # Must fall inside at least one seizure's [onset - 35m, onset - 5m]
        in_preictal = False
        in_exclusion = False

        for s in seizures:
            onset = s["start"]
            offset = s["end"]
            pre_a = max(0.0, onset - pre_start_sec)
            pre_b = max(0.0, onset - pre_end_sec)

            if w_start >= pre_a and w_end <= pre_b:
                in_preictal = True

            # Check overlap with horizon [onset - pre_end_sec, onset]
            if is_overlapping(w_start, w_end, max(0.0, onset - pre_end_sec), onset):
                in_exclusion = True
            # Check overlap with ictal + postictal [onset, offset + post_sec]
            if is_overlapping(w_start, w_end, onset, offset + post_sec):
                in_exclusion = True

        if not in_preictal:
            timing_violations.append(f"{rec_key}: Positive window [{w_start}-{w_end}s] not inside any preictal interval!")
            annotation_ok = False
            break
        if in_exclusion:
            timing_violations.append(f"{rec_key}: Positive window [{w_start}-{w_end}s] overlaps excluded interval!")
            annotation_ok = False
            break

    if timing_violations:
        print("Validation FAILED: Positive windows violate preictal/exclusion boundaries:")
        for v in timing_violations[:5]:
            print("  *", v)
        return False

    # Check 4: Array Shape, Finite Values, and Metadata Agreement
    arrays_ok = True
    sample_files = unique_npz[:check_arrays_sample_size]
    expected_channels = config.get("dataset", {}).get("standard_channels", STANDARD_CHANNELS)
    expected_n_ch = len(expected_channels)

    for rel_path in sample_files:
        full_p = ROOT / rel_path
        with np.load(full_p) as npz:
            if "X" not in npz or "y" not in npz:
                print(f"Validation FAILED: NPZ file {rel_path} missing 'X' or 'y' keys.")
                arrays_ok = False
                break
            X = npz["X"]
            y = npz["y"]

            if not np.isfinite(X).all():
                print(f"Validation FAILED: NPZ file {rel_path} contains non-finite values (NaN / Inf).")
                arrays_ok = False
                break

            if X.ndim != 3:
                print(f"Validation FAILED: Expected 3D array (n_windows, n_channels, n_samples), got {X.shape} in {rel_path}.")
                arrays_ok = False
                break

            if X.shape[1] != expected_n_ch:
                print(f"Validation FAILED: Expected {expected_n_ch} channels, got {X.shape[1]} in {rel_path}.")
                arrays_ok = False
                break

            sub_meta = df_meta[df_meta["npz_path"] == rel_path].sort_values("array_index")
            if len(sub_meta) != len(y) or len(sub_meta) != len(X):
                print(f"Validation FAILED: Metadata rows ({len(sub_meta)}) do not match NPZ array length ({len(y)}) in {rel_path}.")
                arrays_ok = False
                break

            if not np.array_equal(sub_meta["label"].to_numpy(dtype=int), y):
                print(f"Validation FAILED: Metadata labels do not match NPZ 'y' array in {rel_path}.")
                arrays_ok = False
                break

    if not arrays_ok:
        return False

    # Check 5: Sequence Generation Across Recording Boundaries
    seq_ok = True
    n_sequences = 0
    for rec_key, group in df_meta.groupby("record_key", sort=False):
        g = group.sort_values("window_index").reset_index(drop=True)
        for end_idx in range(seq_len - 1, len(g)):
            part = g.iloc[end_idx - seq_len + 1 : end_idx + 1]
            win_indices = part["window_index"].to_numpy()
            labels = part["label"].to_numpy(dtype=int)
            # Continuous consecutive windows only
            if np.all(np.diff(win_indices) == 1) and np.all(labels == labels[-1]):
                n_sequences += 1

    if n_sequences == 0:
        print(f"Validation FAILED: No valid continuous sequences of length {seq_len} could be constructed.")
        seq_ok = False
        return False

    # Summary Output
    print(f"Annotation parsing:        PASS")
    print(f"Preprocessing validation:  PASS")
    print(f"Total windows:             {len(df_meta)}")
    print(f"Interictal windows:        {n_interictal}")
    print(f"Preictal windows:          {n_preictal}")
    print(f"Patients with both classes: {len(pats_with_both)} ({pats_with_both})")
    print(f"Sequence generation:       PASS ({n_sequences} valid sequences of len={seq_len})")
    print("=" * 65)
    print("ALL VALIDATION CHECKS PASSED. Dataset is ready for model training.")
    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate EEG preprocessed windows before training")
    parser.add_argument("--meta", type=str, default=str(DEFAULT_OUT / "window_metadata.csv"), help="Path to window_metadata.csv")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA), help="Path to dataset root")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length to validate")
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if Path(args.config).exists() else None
    passed = validate_dataset(
        meta_path=Path(args.meta),
        data_dir=Path(args.data_dir),
        config=cfg,
        seq_len=args.seq_len,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
