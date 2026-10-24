"""
CHB-MIT Dataset Inspector.
Delegates to parse_annotations for complete inventory, audit, and reporting.
"""

from pathlib import Path
import argparse
from parse_annotations import audit_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "chbmit"
OUT = ROOT / "outputs"


def main():
    parser = argparse.ArgumentParser(description="Inspect CHB-MIT Scalp EEG Dataset")
    parser.add_argument("--data-dir", type=str, default=str(DATA), help="Path to CHB-MIT dataset root")
    parser.add_argument("--out-dir", type=str, default=str(OUT), help="Output directory")
    args = parser.parse_args()

    audit_dataset(Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
