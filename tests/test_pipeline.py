"""
End-to-End Smoke Test for Seizure Prediction Pipeline.
Verifies config loading, metadata integrity, array shapes, and model forward passes.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocess_eeg import STANDARD_CHANNELS
from train_cnn import ConvNet1D
from train_cnn_lstm import SpatialTemporalCNNLSTM


def test_config():
    config_p = ROOT / "configs" / "config.yaml"
    assert config_p.exists(), f"Config missing at {config_p}"
    with open(config_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "dataset" in cfg
    assert "preprocessing" in cfg
    assert "windowing" in cfg
    assert len(cfg["dataset"]["standard_channels"]) == 18
    print("test_config: PASSED")


def test_metadata_and_windows():
    meta_p = ROOT / "outputs" / "window_metadata.csv"
    assert meta_p.exists(), f"Metadata missing at {meta_p}"
    df = pd.read_csv(meta_p)
    assert len(df) > 0, "Metadata is empty"
    assert set(df["label"].unique()).issubset({0, 1})
    assert (df["label"] == 1).sum() > 0, "No preictal windows in metadata"
    assert (df["label"] == 0).sum() > 0, "No interictal windows in metadata"

    # Test loading first NPZ
    first_npz = ROOT / df.iloc[0]["npz_path"]
    assert first_npz.exists(), f"NPZ file missing: {first_npz}"
    with np.load(first_npz) as d:
        assert "X" in d and "y" in d
        X = d["X"]
        y = d["y"]
        assert X.ndim == 3, f"Expected 3D array, got {X.shape}"
        assert X.shape[1] == 18, f"Expected 18 channels, got {X.shape[1]}"
        assert X.shape[2] == 7680, f"Expected 7680 samples, got {X.shape[2]}"
        assert np.isfinite(X).all(), "Non-finite values found in NPZ"
        assert len(X) == len(y)
    print("test_metadata_and_windows: PASSED")


def test_cnn_forward():
    model = ConvNet1D(in_channels=18)
    dummy_x = torch.randn(4, 18, 7680)
    out = model(dummy_x)
    assert out.shape == (4,), f"Expected output shape (4,), got {out.shape}"
    assert torch.isfinite(out).all()
    print("test_cnn_forward: PASSED")


def test_cnn_lstm_forward():
    model = SpatialTemporalCNNLSTM(in_channels=18, cnn_out_dim=64, lstm_hidden_dim=64)
    # Batch=2, Seq=5, Channels=18, Samples=7680
    dummy_seq = torch.randn(2, 5, 18, 7680)
    out = model(dummy_seq)
    assert out.shape == (2,), f"Expected output shape (2,), got {out.shape}"
    assert torch.isfinite(out).all()
    print("test_cnn_lstm_forward: PASSED")


if __name__ == "__main__":
    test_config()
    test_metadata_and_windows()
    test_cnn_forward()
    test_cnn_lstm_forward()
    print("ALL PIPELINE SMOKE TESTS PASSED!")
