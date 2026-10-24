"""
Unit tests for EEG preprocessing, filtering, and labeling logic.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocess_eeg import (
    clean_channel_name,
    apply_bandpass_filter,
    apply_notch_filter,
    robust_normalize,
    STANDARD_CHANNELS,
)
from prepare_dataset import label_window, is_overlapping


def test_channel_cleaning():
    assert clean_channel_name("fp1-f7") == "FP1-F7"
    assert clean_channel_name("T8-P8-0") == "T8-P8"
    assert clean_channel_name("T8-P8-1") == "T8-P8"
    assert clean_channel_name("CZ-PZ ") == "CZ-PZ"
    print("test_channel_cleaning: PASSED")


def test_filters():
    sfreq = 256.0
    t = np.linspace(0, 10, int(10 * sfreq), endpoint=False)
    # Signal: 10 Hz alpha wave (should pass) + 60 Hz high freq noise (should be attenuated)
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
    data = sig[np.newaxis, :]  # shape: (1, n_samples)

    filtered = apply_bandpass_filter(data, sfreq=sfreq, low_freq=0.5, high_freq=40.0)
    assert filtered.shape == data.shape
    assert np.isfinite(filtered).all()

    # 10 Hz power should remain largely intact, 60 Hz should be strongly attenuated
    orig_std = np.std(sig)
    filt_std = np.std(filtered)
    assert filt_std < orig_std  # attenuated high frequency noise
    print("test_filters: PASSED")


def test_robust_normalization():
    np.random.seed(42)
    x = np.random.normal(loc=50.0, scale=15.0, size=(18, 7680)).astype(np.float32)
    # Add an extreme outlier
    x[0, 100] = 50000.0

    norm_x = robust_normalize(x, clip_val=20.0)
    assert norm_x.shape == x.shape
    assert np.isfinite(norm_x).all()
    assert np.all(norm_x >= -20.0) and np.all(norm_x <= 20.0)
    # Median should be approximately 0
    assert abs(np.median(norm_x)) < 0.1
    print("test_robust_normalization: PASSED")


def test_window_labeling():
    # Seizure at 3000s - 3050s
    # Protocol: 35m to 5m before onset = [900s, 2700s]
    # Prediction horizon: [2700s, 3000s]
    # Ictal + Postictal: [3000s, 4850s] (with 30m postictal)
    seizures = [{"start": 3000, "end": 3050}]
    pre_start_sec = 35 * 60  # 2100s
    pre_end_sec = 5 * 60     # 300s
    post_sec = 30 * 60       # 1800s

    # 1. Clean interictal well before preictal
    assert label_window(100, 130, seizures, pre_start_sec, pre_end_sec, post_sec) == 0

    # 2. Inside preictal window [900, 2700]
    assert label_window(1000, 1030, seizures, pre_start_sec, pre_end_sec, post_sec) == 1
    assert label_window(900, 930, seizures, pre_start_sec, pre_end_sec, post_sec) == 1
    assert label_window(2670, 2700, seizures, pre_start_sec, pre_end_sec, post_sec) == 1

    # 3. Crossing preictal start boundary partially
    assert label_window(880, 910, seizures, pre_start_sec, pre_end_sec, post_sec) is None

    # 4. Inside 5-minute prediction horizon [2700, 3000] -> Excluded
    assert label_window(2750, 2780, seizures, pre_start_sec, pre_end_sec, post_sec) is None

    # 5. Overlapping seizure onset [2990, 3020] -> Excluded
    assert label_window(2990, 3020, seizures, pre_start_sec, pre_end_sec, post_sec) is None

    # 6. Inside 30-min postictal [3050, 4850] -> Excluded
    assert label_window(3500, 3530, seizures, pre_start_sec, pre_end_sec, post_sec) is None

    # 7. Beyond postictal [4900, 4930] -> Interictal (0)
    assert label_window(4900, 4930, seizures, pre_start_sec, pre_end_sec, post_sec) == 0
    print("test_window_labeling: PASSED")


if __name__ == "__main__":
    test_channel_cleaning()
    test_filters()
    test_robust_normalization()
    test_window_labeling()
    print("ALL PREPROCESSING TESTS PASSED!")
