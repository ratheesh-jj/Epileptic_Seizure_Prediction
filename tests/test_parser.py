"""
Unit tests for CHB-MIT annotation parser.
Verifies parsing of both unnumbered and numbered seizure entries across patients.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parse_annotations import parse_summary_file, parse_all_summaries


def test_chb01_parsing():
    summary_p = ROOT / "dataset" / "chbmit" / "chb01" / "chb01-summary.txt"
    assert summary_p.exists(), f"Missing {summary_p}"

    res = parse_summary_file(summary_p)
    records = res["records"]

    # Verify chb01_03.edf has 1 seizure at 2996s - 3036s
    assert "chb01_03.edf" in records, "chb01_03.edf missing from parsed records"
    s03 = records["chb01_03.edf"]
    assert len(s03["seizures"]) == 1, f"Expected 1 seizure in chb01_03, got {len(s03['seizures'])}"
    assert s03["seizures"][0]["start"] == 2996
    assert s03["seizures"][0]["end"] == 3036

    # Verify chb01_04.edf has 1 seizure at 1467s - 1494s
    assert "chb01_04.edf" in records
    s04 = records["chb01_04.edf"]
    assert s04["seizures"][0]["start"] == 1467
    assert s04["seizures"][0]["end"] == 1494

    # Verify chb01_01.edf has 0 seizures
    assert "chb01_01.edf" in records
    assert len(records["chb01_01.edf"]["seizures"]) == 0
    print("test_chb01_parsing: PASSED")


def test_chb02_parsing():
    summary_p = ROOT / "dataset" / "chbmit" / "chb02" / "chb02-summary.txt"
    res = parse_summary_file(summary_p)
    records = res["records"]

    assert "chb02_16+.edf" in records
    s16plus = records["chb02_16+.edf"]
    assert len(s16plus["seizures"]) == 1
    assert s16plus["seizures"][0]["start"] == 2972
    assert s16plus["seizures"][0]["end"] == 3053

    assert "chb02_19.edf" in records
    s19 = records["chb02_19.edf"]
    assert s19["seizures"][0]["start"] == 3369
    assert s19["seizures"][0]["end"] == 3378
    print("test_chb02_parsing: PASSED")


def test_chb06_multi_seizure_parsing():
    summary_p = ROOT / "dataset" / "chbmit" / "chb06" / "chb06-summary.txt"
    res = parse_summary_file(summary_p)
    records = res["records"]

    assert "chb06_01.edf" in records
    s01 = records["chb06_01.edf"]
    assert len(s01["seizures"]) == 3, f"Expected 3 seizures in chb06_01, got {len(s01['seizures'])}"
    assert s01["seizures"][0] == {"start": 1724, "end": 1738}
    assert s01["seizures"][1] == {"start": 7461, "end": 7476}
    assert s01["seizures"][2] == {"start": 13525, "end": 13540}
    print("test_chb06_multi_seizure_parsing: PASSED")


if __name__ == "__main__":
    test_chb01_parsing()
    test_chb02_parsing()
    test_chb06_multi_seizure_parsing()
    print("ALL PARSER TESTS PASSED!")
