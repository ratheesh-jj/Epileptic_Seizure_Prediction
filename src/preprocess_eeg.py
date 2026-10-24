"""
EEG Signal Preprocessing Module for CHB-MIT Scalp EEG.
Implements:
  - Channel name harmonization to standard 18 International 10-20 longitudinal bipolar channels
  - Zero-phase Butterworth band-pass filtering (0.5 - 40 Hz)
  - Zero-phase 50 Hz notch filtering
  - Conservative artifact rejection and finite value checks
  - Robust channel-wise normalization (Median / MAD)
"""

from pathlib import Path
import re
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
import mne

# Canonical 18-channel longitudinal bipolar montage (International 10-20)
STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]


def clean_channel_name(ch_name: str) -> str:
    """
    Normalizes EEG channel name by:
      - Converting to uppercase
      - Stripping whitespace
      - Removing disambiguation suffixes added by EDF readers (e.g. -0, -1, .0)
    """
    name = ch_name.upper().strip()
    # Remove trailing -0, -1, .0, etc.
    name = re.sub(r"[-\.]\d+$", "", name)
    return name


def select_and_order_channels(raw: mne.io.Raw, target_channels: list = None) -> tuple:
    """
    Identifies target channels in the MNE Raw object regardless of ordering,
    case, or disambiguation suffixes. Returns (reordered_data_indices, cleaned_names).
    """
    if target_channels is None:
        target_channels = STANDARD_CHANNELS

    raw_names = raw.ch_names
    cleaned_to_orig = {}

    for orig_name in raw_names:
        cleaned = clean_channel_name(orig_name)
        # If not yet registered or preferred (e.g. without suffix)
        if cleaned not in cleaned_to_orig:
            cleaned_to_orig[cleaned] = orig_name

    missing = [ch for ch in target_channels if ch not in cleaned_to_orig]
    if missing:
        raise ValueError(
            f"Recording is missing required standard channels: {missing}. "
            f"Available cleaned channels: {sorted(list(cleaned_to_orig.keys()))}"
        )

    ordered_orig_names = [cleaned_to_orig[ch] for ch in target_channels]
    ch_indices = [raw_names.index(name) for name in ordered_orig_names]
    return ch_indices, target_channels


def apply_bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter using Second-Order Sections (SOS).
    """
    nyquist = sfreq * 0.5
    high = min(high_freq, nyquist * 0.95)
    if high <= low_freq:
        raise ValueError(f"High frequency ({high} Hz) must be greater than low frequency ({low_freq} Hz).")

    sos = butter(order, [low_freq, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, data, axis=-1).astype(np.float32)
    return filtered


def apply_notch_filter(
    data: np.ndarray,
    sfreq: float,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Applies zero-phase 50 Hz IIR notch filter if notch_freq is below Nyquist.
    """
    nyquist = sfreq * 0.5
    if notch_freq < nyquist:
        b, a = iirnotch(notch_freq, Q=quality_factor, fs=sfreq)
        data = filtfilt(b, a, data, axis=-1).astype(np.float32)
    return data


def robust_normalize(
    data: np.ndarray,
    clip_val: float = 20.0,
    ref_med: np.ndarray = None,
    ref_scale: np.ndarray = None,
) -> np.ndarray:
    """
    Robust channel-wise normalization using Median and Median Absolute Deviation (MAD).
    Normalizes along the time axis (axis=-1) for each channel independently.
    If ref_med and ref_scale are provided (e.g. from recording-level baseline),
    uses those reference statistics; otherwise computes them per input slice.
    Clips extreme outliers to [-clip_val, clip_val].
    """
    if ref_med is not None and ref_scale is not None:
        med = ref_med
        scale = ref_scale
    else:
        med = np.median(data, axis=-1, keepdims=True)
        mad = np.median(np.abs(data - med), axis=-1, keepdims=True) * 1.4826
        std = np.std(data, axis=-1, keepdims=True)
        scale = np.where(mad > 1e-6, mad, np.where(std > 1e-6, std, 1.0))

    normalized = (data - med) / np.maximum(scale, 1e-6)
    clipped = np.clip(normalized, -clip_val, clip_val).astype(np.float32)
    return clipped


def preprocess_recording(
    edf_path: Path,
    target_channels: list = None,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
) -> tuple:
    """
    Loads EDF file, extracts harmonized standard channels, filters, and returns
    continuous preprocessed float32 array of shape (n_channels, n_samples) and sfreq.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])

    ch_indices, channel_names = select_and_order_channels(raw, target_channels)
    # Extract data in microvolts (uV)
    data = raw.get_data(picks=ch_indices, units="uV").astype(np.float32)

    # Check finite values
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Filtering
    data = apply_bandpass_filter(data, sfreq, low_freq=bandpass_low, high_freq=bandpass_high)
    data = apply_notch_filter(data, sfreq, notch_freq=notch_freq, quality_factor=notch_q)

    return data, sfreq, channel_names
