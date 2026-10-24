"""
Robust annotation parser and dataset auditor for CHB-MIT Scalp EEG Database.
Parses all patient summary files, extracts seizure intervals, checks validity,
and generates dataset inventory, report, and annotation audit CSV.
"""

from pathlib import Path
import re
import argparse
import pandas as pd
import mne

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "chbmit"
DEFAULT_OUT = ROOT / "outputs"


def parse_summary_file(summary_path: Path) -> dict:
    """
    Parses a single CHB-MIT summary file.
    Handles variations:
      - 'Seizure Start Time: X seconds' (no index)
      - 'Seizure 1 Start Time: X seconds' (numbered index)
      - Variable whitespace and case
      - Header sampling rate and channel list
    """
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    patient = summary_path.parent.name

    # Parse header info
    sfreq = None
    sfreq_match = re.search(r"Data Sampling Rate:\s*(\d+(?:\.\d+)?)\s*Hz", text, re.I)
    if sfreq_match:
        sfreq = float(sfreq_match.group(1))

    # Parse channel list if present in header
    channels = []
    for line in text.splitlines():
        ch_m = re.match(r"\s*Channel\s+\d+:\s*(.+?)\s*$", line, re.I)
        if ch_m:
            channels.append(ch_m.group(1).strip())

    records = {}
    current_edf = None
    current_info = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # File Name line
        fn_match = re.match(r"^File\s+Name:\s*(.+?)\s*$", line, re.I)
        if fn_match:
            if current_edf and current_info:
                records[current_edf] = current_info
            current_edf = fn_match.group(1).strip()
            # Normalize extension if missing
            if not current_edf.lower().endswith(".edf"):
                current_edf += ".edf"
            current_info = {
                "patient_id": patient,
                "edf_name": current_edf,
                "summary_file": summary_path.name,
                "file_start_time": None,
                "file_end_time": None,
                "declared_seizures": 0,
                "seizures": [],
            }
            continue

        if current_edf is None:
            continue

        # Start / End time clock strings
        m_clock_start = re.match(r"^File\s+Start\s+Time:\s*(.+?)\s*$", line, re.I)
        if m_clock_start:
            current_info["file_start_time"] = m_clock_start.group(1).strip()
            continue

        m_clock_end = re.match(r"^File\s+End\s+Time:\s*(.+?)\s*$", line, re.I)
        if m_clock_end:
            current_info["file_end_time"] = m_clock_end.group(1).strip()
            continue

        # Declared seizure count
        m_count = re.match(r"^Number\s+of\s+Seizures\s+in\s+File:\s*(\d+)", line, re.I)
        if m_count:
            current_info["declared_seizures"] = int(m_count.group(1))
            continue

        # Seizure start time (with or without seizure index number)
        m_start = re.search(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*(\d+)\s+seconds", line, re.I)
        if m_start:
            current_info["seizures"].append({"start": int(m_start.group(1))})
            continue

        # Seizure end time (with or without seizure index number)
        m_end = re.search(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*(\d+)\s+seconds", line, re.I)
        if m_end:
            if current_info["seizures"] and "end" not in current_info["seizures"][-1]:
                current_info["seizures"][-1]["end"] = int(m_end.group(1))
            else:
                current_info["seizures"].append({"end": int(m_end.group(1))})
            continue

    if current_edf and current_info:
        records[current_edf] = current_info

    return {
        "patient": patient,
        "summary_path": summary_path,
        "sfreq": sfreq,
        "channels": channels,
        "records": records,
    }


def parse_all_summaries(dataset_root: Path) -> dict:
    """
    Finds and parses all unique summary files in the dataset directory.
    Returns dictionary keyed by patient-qualified record key: '{patient}/{edf_name}'.
    """
    summary_files = sorted(set(dataset_root.rglob("*summary*.txt")))
    all_records = {}

    for summary_path in summary_files:
        try:
            parsed = parse_summary_file(summary_path)
            for edf_name, info in parsed["records"].items():
                record_key = f"{info['patient_id']}/{edf_name}"
                all_records[record_key] = info
        except Exception as e:
            print(f"ERROR parsing {summary_path}: {e}")

    return all_records


def audit_dataset(dataset_root: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUT):
    """
    Cross-checks all EDF files on disk against summary annotations.
    Generates:
      - dataset_inventory.csv
      - annotation_audit.csv
      - dataset_report.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = parse_all_summaries(dataset_root)

    # Find all EDF files on disk
    all_edfs = sorted(set(dataset_root.rglob("*.edf")) | set(dataset_root.rglob("*.EDF")))
    print(f"Found {len(all_edfs)} EDF files and {len(summaries)} annotated records in summary files.")

    inventory_rows = []
    audit_rows = []

    for edf_path in all_edfs:
        rel_path = edf_path.relative_to(dataset_root)
        parts = rel_path.parts
        patient_id = parts[0] if len(parts) > 1 else "unknown"
        edf_name = edf_path.name
        record_key = f"{patient_id}/{edf_name}"

        # Quick header read without loading data
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            duration_sec = float(raw.n_times / sfreq)
            ch_count = len(raw.ch_names)
            read_error = None
        except Exception as e:
            sfreq = None
            duration_sec = None
            ch_count = None
            read_error = str(e)

        file_size_mb = round(edf_path.stat().st_size / (1024 * 1024), 2)
        ann = summaries.get(record_key)

        declared_seizures = ann["declared_seizures"] if ann else 0
        seizures = ann["seizures"] if ann else []
        valid_seizures = [s for s in seizures if "start" in s and "end" in s and s["start"] < s["end"]]

        # Check preictal availability: 35 minutes before onset is 2100 seconds
        has_full_preictal = any(s["start"] >= 2100 for s in valid_seizures)
        seizure_str = "; ".join(f"{s['start']}-{s['end']}s" for s in valid_seizures)

        inventory_rows.append({
            "record_key": record_key,
            "patient_id": patient_id,
            "edf_name": edf_name,
            "relative_path": str(rel_path),
            "size_mb": file_size_mb,
            "sampling_frequency": sfreq,
            "duration_sec": duration_sec,
            "duration_min": round(duration_sec / 60, 2) if duration_sec else None,
            "channel_count": ch_count,
            "declared_seizures": declared_seizures,
            "annotated_seizures": len(valid_seizures),
            "seizure_intervals": seizure_str,
            "has_full_preictal_35m": has_full_preictal,
            "read_error": read_error,
        })

        # Detailed audit row per seizure event or per recording
        if valid_seizures:
            for s_idx, s in enumerate(valid_seizures, 1):
                onset, offset = s["start"], s["end"]
                is_valid = onset < offset and (duration_sec is None or offset <= duration_sec + 5)
                preictal_avail_sec = onset
                preictal_avail_min = round(onset / 60, 2)
                exclusion_reason = []
                if not is_valid:
                    exclusion_reason.append("Invalid onset/offset times")
                if onset < 300: # Less than 5 min available
                    exclusion_reason.append("Onset within 5m prediction horizon (no preictal data available)")
                elif onset < 2100:
                    exclusion_reason.append("Partial preictal (<35m before onset in this recording)")

                audit_rows.append({
                    "record_key": record_key,
                    "patient_id": patient_id,
                    "edf_name": edf_name,
                    "summary_file": ann["summary_file"] if ann else None,
                    "seizure_index": s_idx,
                    "onset_sec": onset,
                    "offset_sec": offset,
                    "duration_sec": offset - onset,
                    "recording_duration_sec": duration_sec,
                    "is_annotation_valid": is_valid,
                    "preictal_available_min": preictal_avail_min,
                    "has_full_35m_preictal": onset >= 2100,
                    "status_note": "; ".join(exclusion_reason) if exclusion_reason else "Full 35m preictal available",
                })
        else:
            audit_rows.append({
                "record_key": record_key,
                "patient_id": patient_id,
                "edf_name": edf_name,
                "summary_file": ann["summary_file"] if ann else None,
                "seizure_index": 0,
                "onset_sec": None,
                "offset_sec": None,
                "duration_sec": None,
                "recording_duration_sec": duration_sec,
                "is_annotation_valid": True,
                "preictal_available_min": None,
                "has_full_35m_preictal": False,
                "status_note": "Interictal recording (0 seizures)",
            })

    df_inv = pd.DataFrame(inventory_rows)
    df_audit = pd.DataFrame(audit_rows)

    df_inv.to_csv(output_dir / "dataset_inventory.csv", index=False)
    df_audit.to_csv(output_dir / "annotation_audit.csv", index=False)

    # Generate dataset report
    n_total_edf = len(df_inv)
    n_patients = df_inv["patient_id"].nunique()
    seizure_records = df_inv[df_inv["annotated_seizures"] > 0]
    total_seizures = df_inv["annotated_seizures"].sum()
    seizures_with_full_preictal = (df_audit["has_full_35m_preictal"] == True).sum()

    report_lines = [
        "=" * 70,
        "CHB-MIT SCALP EEG DATASET INVENTORY AND ANNOTATION AUDIT",
        "=" * 70,
        f"Dataset root: {dataset_root}",
        f"Total EDF recordings found: {n_total_edf}",
        f"Total unique patients: {n_patients}",
        f"Recordings with annotated seizures: {len(seizure_records)} (out of {n_total_edf})",
        f"Total annotated seizure events: {total_seizures}",
        f"Seizures with full 35m preictal in single recording: {seizures_with_full_preictal}",
        "",
        "PATIENT BREAKDOWN:",
        "-" * 70,
    ]

    patient_summary = df_inv.groupby("patient_id").agg(
        recordings=("edf_name", "count"),
        seizure_recordings=("annotated_seizures", lambda x: (x > 0).sum()),
        total_seizures=("annotated_seizures", "sum"),
        full_preictal_seizures=("has_full_preictal_35m", "sum"),
    )
    report_lines.append(patient_summary.to_string())

    report_lines.extend([
        "",
        "=" * 70,
        "ANNOTATED SEIZURE RECORDINGS DETAIL:",
        "-" * 70,
    ])
    for _, row in seizure_records.iterrows():
        report_lines.append(
            f"{row['record_key']} | Duration: {row['duration_min']} min | "
            f"Seizures: {row['annotated_seizures']} | "
            f"Times: {row['seizure_intervals']} | Full preictal >=35m: {row['has_full_preictal_35m']}"
        )

    report_text = "\n".join(report_lines)
    (output_dir / "dataset_report.txt").write_text(report_text, encoding="utf-8")

    print(f"\nAudit complete.")
    print(f"Total EDF files: {n_total_edf}")
    print(f"Recordings with seizures: {len(seizure_records)}")
    print(f"Total seizure events: {total_seizures}")
    print(f"Saved: {output_dir / 'dataset_inventory.csv'}")
    print(f"Saved: {output_dir / 'annotation_audit.csv'}")
    print(f"Saved: {output_dir / 'dataset_report.txt'}")
    return df_inv, df_audit


def main():
    parser = argparse.ArgumentParser(description="Audit and parse CHB-MIT EEG dataset annotations")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA), help="Path to CHB-MIT dataset root")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help="Output directory")
    args = parser.parse_args()

    audit_dataset(Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
