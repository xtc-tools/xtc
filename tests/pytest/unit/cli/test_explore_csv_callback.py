import csv
from pathlib import Path

from xtc.cli.explore import CSVCallback


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as infile:
        return list(csv.reader(infile, delimiter=","))


def test_csv_callback_resume_dedup_skips_existing_and_keeps_new(tmp_path: Path):
    output = tmp_path / "results.csv"
    sample_names = ["M", "N"]

    # Existing output already has sample [8, 16] for backend mlir.
    with output.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(sample_names + ["X", "time", "peak", "backend"])
        writer.writerow([8, 16, "[8; 16]", 0.2, 5.0, "mlir"])

    cb = CSVCallback(
        str(output),
        peak_time=1.0,
        sample_names=sample_names,
        resume=True,
    )

    # Duplicate for same backend must be skipped in resume mode.
    cb(([8, 16], 0, 0.2, "mlir"))
    # Different backend for same sample must be recorded.
    cb(([8, 16], 0, 0.25, "tvm"))
    # Different sample must be recorded.
    cb(([8, 32], 0, 0.3, "mlir"))
    del cb

    rows = _read_rows(output)

    # Header + original row + two new rows.
    assert len(rows) == 4
    assert rows[0] == ["M", "N", "X", "time", "peak", "backend"]

    # Check appended rows (order matters).
    assert rows[2][0:2] == ["8", "16"]
    assert rows[2][-1] == "tvm"
    assert rows[3][0:2] == ["8", "32"]
    assert rows[3][-1] == "mlir"


def test_csv_callback_default_mode_overwrites_existing_file(tmp_path: Path):
    output = tmp_path / "results.csv"
    sample_names = ["M", "N"]

    with output.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(sample_names + ["X", "time", "peak", "backend"])
        writer.writerow([9, 9, "[9; 9]", 0.9, 1.1, "mlir"])

    cb = CSVCallback(
        str(output),
        peak_time=2.0,
        sample_names=sample_names,
    )

    # Default mode is neither --resume nor --append, so file is rewritten.
    cb(([2, 3], 0, 0.5, "mlir"))
    del cb

    rows = _read_rows(output)
    assert rows[0] == ["M", "N", "X", "time", "peak", "backend"]
    assert len(rows) == 2
    assert rows[1][0:2] == ["2", "3"]
    assert rows[1][-1] == "mlir"


def test_csv_callback_append_mode_allows_duplicates(tmp_path: Path):
    output = tmp_path / "results.csv"
    sample_names = ["M", "N"]

    with output.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(sample_names + ["X", "time", "peak", "backend"])
        writer.writerow([4, 4, "[4; 4]", 0.1, 10.0, "mlir"])

    cb = CSVCallback(
        str(output),
        peak_time=1.0,
        sample_names=sample_names,
        append=True,
    )

    # Same sample/backend is allowed in append mode.
    cb(([4, 4], 0, 0.12, "mlir"))
    del cb

    rows = _read_rows(output)

    # Header + original row + duplicate appended row.
    assert len(rows) == 3
    assert rows[1][0:2] == ["4", "4"]
    assert rows[2][0:2] == ["4", "4"]
    assert rows[1][-1] == "mlir"
    assert rows[2][-1] == "mlir"


def test_csv_callback_append_writes_header_for_new_file(tmp_path: Path):
    output = tmp_path / "results.csv"

    cb = CSVCallback(
        str(output),
        peak_time=2.0,
        sample_names=["M", "N"],
        append=True,
    )
    cb(([2, 3], 0, 0.5, "mlir"))
    del cb

    rows = _read_rows(output)
    assert rows[0] == ["M", "N", "X", "time", "peak", "backend"]
    assert rows[1][0:2] == ["2", "3"]
