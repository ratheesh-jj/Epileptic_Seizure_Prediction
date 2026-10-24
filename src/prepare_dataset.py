"""
EEG Dataset Preparation and Windowing Pipeline for CHB-MIT Seizure Prediction.
Windows continuous EEG recordings, applies robust preprocessing, labels windows according
to a strict seizure prediction protocol (interictal vs preictal with postictal and horizon exclusions),
and saves standardized NPZ files and metadata.
"""

import sys
from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parse_annotations import parse_all_summaries
from preprocess_eeg import (
    STANDARD_CHANNELS,
    preprocess_recording,
    robust_normalize,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_DATA = ROOT / "dataset" / "chbmit"
DEFAULT_OUT = ROOT / "outputs"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def is_overlapping(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Returns True if interval [start_a, end_a] overlaps with [start_b, end_b]."""
    return max(start_a, start_b) < min(end_a, end_b)


def label_window(
    win_start: float,
    win_end: float,
    seizures: list,
    pre_start_sec: float,
    pre_end_sec: float,
    post_sec: float,
) -> int or None:
    """
    Labels a window based on seizure timings:
      - 1 (Preictal): The entire window falls strictly inside [onset - pre_start_sec, onset - pre_end_sec]
                      for at least one seizure, AND does not overlap with any exclusion zone.
      - None (Excluded): Overlaps with prediction horizon [onset - pre_end_sec, onset],
                         ictal period [onset, offset], or postictal period [offset, offset + post_sec].
      - 0 (Interictal): Clean period outside all preictal and exclusion zones.
    """
    # First: check if window overlaps ANY exclusion zone for ANY seizure
    for s in seizures:
        onset = s["start"]
        offset = s["end"]
        # Prediction horizon: [onset - pre_end_sec, onset]
        horizon_start = max(0.0, onset - pre_end_sec)
        horizon_end = onset
        if is_overlapping(win_start, win_end, horizon_start, horizon_end):
            return None

        # Ictal + Postictal exclusion zone: [onset, offset + post_sec]
        post_end = offset + post_sec
        if is_overlapping(win_start, win_end, onset, post_end):
            return None

    # Second: check if window is fully inside ANY valid preictal interval
    is_preictal = False
    for s in seizures:
        onset = s["start"]
        pre_a = max(0.0, onset - pre_start_sec)
        pre_b = max(0.0, onset - pre_end_sec)

        # Window must be completely enclosed within the preictal zone
        if win_start >= pre_a and win_end <= pre_b:
            is_preictal = True
            break
        # If it partially overlaps preictal boundary without being fully enclosed
        elif is_overlapping(win_start, win_end, pre_a, pre_b):
            return None

    if is_preictal:
        return 1

    # Otherwise clean interictal
    return 0


def prepare_dataset(
    data_dir: Path = DEFAULT_DATA,
    out_dir: Path = DEFAULT_OUT,
    config: dict = None,
    patients: list = None,
    max_recordings: int = None,
    max_interictal_per_patient: int = None,
    overwrite: bool = True,
):
    """
    Processes EDF recordings, generates normalized windows, and writes NPZ files and metadata.
    """
    if config is None:
        config = load_config()

    win_cfg = config.get("windowing", {})
    prep_cfg = config.get("preprocessing", {})
    ds_cfg = config.get("dataset", {})

    window_sec = float(win_cfg.get("window_sec", 30.0))
    stride_sec = float(win_cfg.get("stride_sec", 30.0))
    pre_start_min = float(win_cfg.get("pre_start_min", 35.0))
    pre_end_min = float(win_cfg.get("pre_end_min", 5.0))
    post_min = float(win_cfg.get("post_min", 30.0))

    pre_start_sec = pre_start_min * 60.0
    pre_end_sec = pre_end_min * 60.0
    post_sec = post_min * 60.0

    target_channels = ds_cfg.get("standard_channels", STANDARD_CHANNELS)
    bandpass_low = float(prep_cfg.get("bandpass_low", 0.5))
    bandpass_high = float(prep_cfg.get("bandpass_high", 40.0))
    notch_freq = float(prep_cfg.get("notch_freq", 50.0))
    notch_q = float(prep_cfg.get("notch_q", 30.0))
    clip_val = float(prep_cfg.get("clip_value", 20.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    # Parse all summary annotations
    summaries = parse_all_summaries(data_dir)
    print(f"Loaded annotations for {len(summaries)} recordings from summary files.")

    # Select patients
    available_patient_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.lower().startswith("chb")]
    )
    if patients:
        wanted_patients = {p.lower() for p in patients}
        patient_dirs = [p for p in available_patient_dirs if p.name.lower() in wanted_patients]
    else:
        # Default to first 5 patients or all if specified
        patient_dirs = available_patient_dirs[:5]

    print(f"Target patients ({len(patient_dirs)}): {[p.name for p in patient_dirs]}")

    edfs = []
    for p in patient_dirs:
        pat_edfs = sorted(set(list(p.glob("*.edf")) + list(p.glob("*.EDF"))))
        if max_interictal_per_patient is not None:
            seizure_edfs = []
            interictal_edfs = []
            for e in pat_edfs:
                r_key = f"{p.name}/{e.name}"
                ann = summaries.get(r_key)
                if ann and ann.get("seizures"):
                    seizure_edfs.append(e)
                else:
                    interictal_edfs.append(e)
            selected = seizure_edfs + interictal_edfs[:max_interictal_per_patient]
            edfs.extend(selected)
        else:
            edfs.extend(pat_edfs)

    if max_recordings:
        edfs = edfs[:max_recordings]

    print(f"Total EDF recordings queued for processing: {len(edfs)}")

    meta_csv_path = out_dir / "window_metadata.csv"
    existing_meta_df = None
    if not overwrite and meta_csv_path.exists():
        try:
            existing_meta_df = pd.read_csv(meta_csv_path)
        except Exception:
            existing_meta_df = None

    all_metadata = []
    skipped_recordings = []

    for edf_path in tqdm(edfs, desc="Processing recordings"):
        patient_id = edf_path.parent.name
        edf_name = edf_path.name
        record_key = f"{patient_id}/{edf_name}"
        record_stem = f"{patient_id}_{edf_path.stem}"
        npz_dest = windows_dir / f"{record_stem}.npz"

        # Fast incremental skip if recording was already prepared
        if not overwrite and npz_dest.exists():
            if existing_meta_df is not None and "record_key" in existing_meta_df.columns:
                existing_rows = existing_meta_df[existing_meta_df["record_key"] == record_key]
                if len(existing_rows) > 0:
                    all_metadata.extend(existing_rows.to_dict("records"))
                    continue

        ann_info = summaries.get(record_key)
        seizures = ann_info["seizures"] if ann_info else []
        valid_seizures = [s for s in seizures if "start" in s and "end" in s and s["start"] < s["end"]]

        try:
            # Preprocess continuous EEG
            data, sfreq, ch_names = preprocess_recording(
                edf_path=edf_path,
                target_channels=target_channels,
                bandpass_low=bandpass_low,
                bandpass_high=bandpass_high,
                notch_freq=notch_freq,
                notch_q=notch_q,
            )

            n_samples = data.shape[1]
            samples_per_win = int(round(window_sec * sfreq))
            samples_per_stride = int(round(stride_sec * sfreq))

            if n_samples < samples_per_win:
                skipped_recordings.append({"record_key": record_key, "reason": "Recording shorter than window length"})
                continue

            # Compute recording-level robust baseline parameters across continuous recording
            rec_med = np.median(data, axis=-1, keepdims=True)
            rec_mad = np.median(np.abs(data - rec_med), axis=-1, keepdims=True) * 1.4826
            rec_std = np.std(data, axis=-1, keepdims=True)
            rec_scale = np.where(rec_mad > 1e-6, rec_mad, np.where(rec_std > 1e-6, rec_std, 1.0))

            windows_list = []
            labels_list = []
            starts_list = []
            ends_list = []
            orig_indices = []

            for s_idx in range(0, n_samples - samples_per_win + 1, samples_per_stride):
                e_idx = s_idx + samples_per_win
                win_start = s_idx / sfreq
                win_end = e_idx / sfreq

                label = label_window(
                    win_start=win_start,
                    win_end=win_end,
                    seizures=valid_seizures,
                    pre_start_sec=pre_start_sec,
                    pre_end_sec=pre_end_sec,
                    post_sec=post_sec,
                )

                if label is None:
                    continue  # Excluded window (ictal, postictal, or prediction horizon)

                win_raw = data[:, s_idx:e_idx]
                if not np.isfinite(win_raw).all():
                    continue

                # Robust window normalization relative to recording baseline
                win_norm = robust_normalize(win_raw, clip_val=clip_val, ref_med=rec_med, ref_scale=rec_scale)

                windows_list.append(win_norm)
                labels_list.append(label)
                starts_list.append(round(win_start, 3))
                ends_list.append(round(win_end, 3))
                orig_indices.append(s_idx // samples_per_stride)

            if not windows_list:
                skipped_recordings.append({"record_key": record_key, "reason": "No valid labeled windows in recording"})
                continue

            X = np.stack(windows_list).astype(np.float32)
            y = np.asarray(labels_list, dtype=np.int64)

            # Save compressed NPZ
            np.savez_compressed(
                npz_dest,
                X=X,
                y=y,
                channels=np.array(ch_names),
                sfreq=sfreq,
                record_key=record_key,
            )

            rel_npz_path = str(npz_dest.relative_to(ROOT))
            for arr_i, (w_start, w_end, lbl, orig_i) in enumerate(
                zip(starts_list, ends_list, labels_list, orig_indices)
            ):
                all_metadata.append({
                    "record_key": record_key,
                    "patient_id": patient_id,
                    "edf_name": edf_name,
                    "window_index": orig_i,
                    "array_index": arr_i,
                    "start_sec": w_start,
                    "end_sec": w_end,
                    "label": lbl,
                    "n_channels": X.shape[1],
                    "samples_per_channel": X.shape[2],
                    "sfreq": sfreq,
                    "npz_path": rel_npz_path,
                })

            n_preictal = int((y == 1).sum())
            n_interictal = int((y == 0).sum())
            if n_preictal > 0:
                print(f"  [+] {record_key}: {len(y)} windows -> {n_preictal} preictal, {n_interictal} interictal (Seizures: {len(valid_seizures)})")

        except Exception as e:
            print(f"ERROR processing {record_key}: {e}")
            skipped_recordings.append({"record_key": record_key, "reason": str(e)})

    if not all_metadata:
        raise RuntimeError("No windows were successfully generated. Please check annotations and recording paths.")

    df_meta = pd.DataFrame(all_metadata)
    df_meta.to_csv(out_dir / "window_metadata.csv", index=False)

    # Label distribution per patient
    dist_df = df_meta.groupby(["patient_id", "label"]).size().unstack(fill_value=0)
    if 0 not in dist_df.columns:
        dist_df[0] = 0
    if 1 not in dist_df.columns:
        dist_df[1] = 0
    dist_df.rename(columns={0: "interictal_windows", 1: "preictal_windows"}, inplace=True)
    dist_df["total_windows"] = dist_df["interictal_windows"] + dist_df["preictal_windows"]
    dist_df["preictal_pct"] = round(100 * dist_df["preictal_windows"] / dist_df["total_windows"], 2)
    dist_df.to_csv(out_dir / "label_distribution.csv")

    total_preictal = int((df_meta["label"] == 1).sum())
    total_interictal = int((df_meta["label"] == 0).sum())

    prep_report = {
        "status": "SUCCESS",
        "processed_recordings": int(df_meta["record_key"].nunique()),
        "processed_patients": sorted(df_meta["patient_id"].unique().tolist()),
        "total_windows": len(df_meta),
        "preictal_windows": total_preictal,
        "interictal_windows": total_interictal,
        "window_duration_sec": window_sec,
        "window_stride_sec": stride_sec,
        "preictal_protocol_min_before_onset": [pre_start_min, pre_end_min],
        "prediction_horizon_min": pre_end_min,
        "postictal_exclusion_min": post_min,
        "channels_used": target_channels,
        "n_channels": len(target_channels),
        "skipped_recordings_count": len(skipped_recordings),
    }

    with open(out_dir / "preparation_report.json", "w", encoding="utf-8") as f:
        json.dump(prep_report, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY:")
    print("=" * 60)
    print(f"Total windows generated: {len(df_meta)}")
    print(f"  - Preictal windows (Class 1): {total_preictal}")
    print(f"  - Interictal windows (Class 0): {total_interictal}")
    print("\nLabel Distribution per Patient:")
    print(dist_df.to_string())
    print("=" * 60)
    return df_meta, prep_report


def main():
    parser = argparse.ArgumentParser(description="Prepare CHB-MIT EEG windows and labels")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA), help="Path to CHB-MIT dataset root")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--patients", nargs="*", default=None, help="Specific patients to process (e.g. chb01 chb02 chb03)")
    parser.add_argument("--max-recordings", type=int, default=None, help="Cap number of recordings for quick test")
    parser.add_argument("--max-interictal-per-patient", type=int, default=None, help="Cap number of pure interictal recordings per patient")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing prepared windows")
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if Path(args.config).exists() else None
    prepare_dataset(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        config=cfg,
        patients=args.patients,
        max_recordings=args.max_recordings,
        max_interictal_per_patient=args.max_interictal_per_patient,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
