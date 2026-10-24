"""
Comprehensive Evaluation Module for Epileptic Seizure Prediction.
Implements:
  1. Window-Level Evaluation:
     - Sensitivity/Recall, Specificity, Precision, F1-Score, Balanced Accuracy, AUROC, PR-AUC, Confusion Matrix.
     - Comparative table across Baseline A (Random Forest), Baseline B (1D CNN), and Primary (CNN-LSTM).
  2. Event-Level Seizure Prediction Simulation:
     - Continuous chronological evaluation on complete held-out test recordings.
     - Alarm thresholding with persistence requirement (consecutive positive windows).
     - Refractory period handling.
     - Event Sensitivity (% seizures predicted within intervention horizon).
     - False Prediction Rate per hour (FPR/h).
     - Warning lead time (minutes before seizure onset).
     - Proportion of total time spent in warning state.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def simulate_event_alarms(
    predictions_df: pd.DataFrame,
    seizures_dict: dict,
    alarm_threshold: float = 0.5,
    persistence_k: int = 2,
    pre_start_min: float = 35.0,
    pre_end_min: float = 5.0,
    refractory_min: float = 60.0,
) -> dict:
    """
    Simulates clinical event-level seizure alarm system chronologically.
    An alarm is triggered only when model probability >= alarm_threshold for
    persistence_k consecutive windows.
    """
    total_monitoring_sec = 0.0
    total_alarms = 0
    false_alarms = 0
    predicted_seizures = 0
    total_seizures = 0
    warning_lead_times = []
    time_in_warning_sec = 0.0

    refractory_sec = refractory_min * 60.0
    pre_start_sec = pre_start_min * 60.0
    pre_end_sec = pre_end_min * 60.0

    for rec_key, group in predictions_df.groupby("record_key", sort=False):
        g = group.sort_values("start_sec").reset_index(drop=True)
        rec_duration = float(g["end_sec"].max()) if len(g) else 0.0
        total_monitoring_sec += rec_duration

        seizures = seizures_dict.get(rec_key, [])
        total_seizures += len(seizures)
        seizure_predicted_flags = [False] * len(seizures)

        consecutive_positives = 0
        last_alarm_time = -float("inf")

        for _, row in g.iterrows():
            prob = float(row["prob"])
            win_start = float(row["start_sec"])
            win_end = float(row["end_sec"])
            win_len = win_end - win_start

            if prob >= alarm_threshold:
                consecutive_positives += 1
            else:
                consecutive_positives = 0

            # Check if alarm is triggered
            if consecutive_positives >= persistence_k:
                time_in_warning_sec += win_len

                # Check refractory period
                if (win_end - last_alarm_time) >= refractory_sec:
                    last_alarm_time = win_end
                    total_alarms += 1

                    # Check if alarm falls within valid preictal window of any seizure
                    is_true_alarm = False
                    for s_i, s in enumerate(seizures):
                        onset = s["start"]
                        valid_alarm_start = max(0.0, onset - pre_start_sec)
                        valid_alarm_end = max(0.0, onset - pre_end_sec)

                        if valid_alarm_start <= win_end <= valid_alarm_end:
                            is_true_alarm = True
                            if not seizure_predicted_flags[s_i]:
                                seizure_predicted_flags[s_i] = True
                                lead_time_min = round((onset - win_end) / 60.0, 2)
                                warning_lead_times.append(lead_time_min)

                    if not is_true_alarm:
                        false_alarms += 1

        predicted_seizures += sum(seizure_predicted_flags)

    monitoring_hours = round(total_monitoring_sec / 3600.0, 2)
    fpr_per_hour = round(false_alarms / max(monitoring_hours, 1e-4), 4)
    event_sensitivity = round(predicted_seizures / max(total_seizures, 1), 4) if total_seizures > 0 else "N/A"
    pct_time_in_warning = round(100.0 * time_in_warning_sec / max(total_monitoring_sec, 1e-4), 2)
    mean_lead_time = round(float(np.mean(warning_lead_times)), 2) if warning_lead_times else "N/A"

    return {
        "total_monitoring_hours": monitoring_hours,
        "total_seizures_in_test": total_seizures,
        "seizures_predicted": predicted_seizures,
        "event_level_sensitivity": event_sensitivity,
        "total_alarms_raised": total_alarms,
        "false_alarms": false_alarms,
        "false_prediction_rate_per_hour": fpr_per_hour,
        "mean_warning_lead_time_min": mean_lead_time,
        "warning_lead_times_min": warning_lead_times,
        "percent_time_in_warning_state": pct_time_in_warning,
        "parameters": {
            "alarm_threshold": alarm_threshold,
            "persistence_consecutive_windows": persistence_k,
            "refractory_period_min": refractory_min,
            "preictal_window_min": [pre_start_min, pre_end_min],
        },
    }


def generate_evaluation_report(out_dir: Path = DEFAULT_OUT, config: dict = None):
    if config is None:
        config = load_config()

    rf_file = out_dir / "rf_evaluation.json"
    cnn_file = out_dir / "cnn_evaluation.json"
    cnn_lstm_file = out_dir / "cnn_lstm_evaluation.json"

    models_data = {}
    if rf_file.exists():
        with open(rf_file, "r", encoding="utf-8") as f:
            models_data["RandomForest"] = json.load(f)
    if cnn_file.exists():
        with open(cnn_file, "r", encoding="utf-8") as f:
            models_data["1D-CNN"] = json.load(f)
    if cnn_file.exists() and cnn_lstm_file.exists():
        with open(cnn_lstm_file, "r", encoding="utf-8") as f:
            models_data["CNN-LSTM"] = json.load(f)

    # Compile comparative table
    comparison_rows = []
    for m_name, d in models_data.items():
        tm = d.get("test_metrics", {})
        cm = tm.get("confusion_matrix", {})
        comparison_rows.append({
            "Model": m_name,
            "Accuracy": tm.get("accuracy", "N/A"),
            "Balanced Acc": tm.get("balanced_accuracy", "N/A"),
            "Sensitivity (Recall)": tm.get("sensitivity_recall", "N/A"),
            "Specificity": tm.get("specificity", "N/A"),
            "Precision": tm.get("precision", "N/A"),
            "F1-Score": tm.get("f1_score", "N/A"),
            "AUROC": tm.get("auroc", "N/A"),
            "PR-AUC": tm.get("pr_auc", "N/A"),
            "TP": cm.get("TP", "N/A"),
            "FP": cm.get("FP", "N/A"),
            "TN": cm.get("TN", "N/A"),
            "FN": cm.get("FN", "N/A"),
        })

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(out_dir / "comparative_metrics.csv", index=False)

    report_lines = [
        "=" * 70,
        "AI-BASED EPILEPTIC SEIZURE PREDICTION: EXPERIMENTAL EVALUATION REPORT",
        "=" * 70,
        "",
        "1. WINDOW-LEVEL CLASSIFICATION PERFORMANCE ON UNTOUCHED HELD-OUT TEST PATIENTS:",
        "-" * 70,
        df_comp.to_string(index=False) if len(df_comp) else "No model evaluations found.",
        "",
        "2. KEY METHODOLOGICAL NOTES & LEAKAGE PREVENTION:",
        "-" * 70,
        "- Patient-Independent Evaluation: The test patients were never seen during model training,",
        "  feature scaling, early stopping, or hyperparameter selection.",
        "- Prediction Protocol: 30s windows with 5-minute intervention horizon and 30-minute postictal exclusion.",
        "- Standardized 18-channel longitudinal bipolar montage (International 10-20) used across all models.",
        "",
        "3. CLINICAL REALITY & EVENT-LEVEL ALARM INTERPRETATION:",
        "-" * 70,
        "- High window-level accuracy does NOT guarantee clinically actionable seizure prediction.",
        "- In continuous monitoring, the vast majority of EEG is interictal (>95%). Even a false positive rate",
        "  of 2-5% at the window level can produce unacceptably high false alarm rates (FPR/h) in clinical practice.",
        "- Persistence filtering (requiring consecutive positive windows) and refractory periods significantly",
        "  reduce false alarm cascades.",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    (out_dir / "evaluation_summary.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    return df_comp


def main():
    generate_evaluation_report()


if __name__ == "__main__":
    main()
