import csv
from pathlib import Path

import pytest

import xtc.graphs.xtc.op as O
from xtc.search.callback import read_csv_results
from xtc.search.explore import CSVCallback, Exploration, ExplorationConfig


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as infile:
        return list(csv.reader(infile, delimiter=","))


def _matmul_relu_graph():
    lhs = O.tensor((4, 8), "float32", name="lhs")
    rhs = O.tensor((8, 16), "float32", name="rhs")
    with O.graph(name="matmul_relu") as gb:
        matmul = O.matmul(lhs, rhs, name="matmul")
        O.relu(matmul, name="relu")
    return gb.graph


def test_exploration_strategy_targets_selected_node():
    exploration = Exploration(ExplorationConfig(node="matmul"))
    strategy = exploration.get_strategy(_matmul_relu_graph())

    assert strategy.sample_names == ["x0", "x1", "x2"]


def test_exploration_strategy_rejects_unknown_node():
    exploration = Exploration(ExplorationConfig(node="unknown"))

    with pytest.raises(ValueError, match="graph node named 'unknown'"):
        exploration.get_strategy(_matmul_relu_graph())


def test_csv_callback_resume_dedup_skips_existing_and_keeps_new(tmp_path: Path):
    output = tmp_path / "results.csv"
    sample_names = ["M", "N"]

    # Existing output already has sample [8, 16] for backend mlir.
    with output.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(sample_names + ["X", "time", "peak", "backend"])
        writer.writerow([8, 16, "[8; 16]", 2.0, 0.5, "mlir"])

    cb = CSVCallback(
        str(output),
        peak_time=1.0,
        sample_names=sample_names,
        resume=True,
    )

    # Duplicate for same backend must be skipped in resume mode.
    cb(([8, 16], 0, 3.0, "mlir"))
    # Different backend for same sample must be recorded.
    cb(([8, 16], 0, 2.5, "tvm"))
    # Different sample must be recorded.
    cb(([8, 32], 0, 3.0, "mlir"))
    del cb

    rows = _read_rows(output)

    # Header + original row + two new rows.
    assert len(rows) == 4
    assert rows[0] == ["M", "N", "X", "time", "peak", "backend"]

    # Check appended rows (order matters).
    assert rows[1][0:2] == ["8", "16"]
    assert rows[1][-3:] == ["2.0", "0.5", "mlir"]
    assert rows[2][0:2] == ["8", "16"]
    assert rows[2][-3:] == ["2.5", "0.4", "tvm"]
    assert rows[3][0:2] == ["8", "32"]
    assert rows[3][-1] == "mlir"


def test_csv_callback_default_mode_overwrites_existing_file(tmp_path: Path):
    output = tmp_path / "results.csv"
    sample_names = ["M", "N"]

    with output.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(sample_names + ["X", "time", "peak", "backend"])
        writer.writerow([9, 9, "[9; 9]", 8.0, 0.25, "mlir"])

    cb = CSVCallback(
        str(output),
        peak_time=2.0,
        sample_names=sample_names,
    )

    # Default mode is neither --resume nor --append, so file is rewritten.
    cb(([2, 3], 0, 6.0, "mlir"))
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
        writer.writerow([4, 4, "[4; 4]", 10.0, 0.1, "mlir"])

    cb = CSVCallback(
        str(output),
        peak_time=1.0,
        sample_names=sample_names,
        append=True,
    )

    # Same sample/backend is allowed in append mode.
    cb(([4, 4], 0, 1.6, "mlir"))
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
    cb(([2, 3], 0, 8.0, "mlir"))
    del cb

    rows = _read_rows(output)
    assert rows[0] == ["M", "N", "X", "time", "peak", "backend"]
    assert rows[1] == ["2", "3", "[2; 3]", "8.0", "0.25", "mlir"]


def test_read_csv_results_filters_backend_and_bad_rows(tmp_path: Path):
    output = tmp_path / "results.csv"
    sample_names = ["M", "N"]

    with output.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(sample_names + ["X", "time", "peak", "backend"])
        writer.writerow([8, 16, "[8; 16]", 2.0, 0.5, "mlir"])
        writer.writerow([8, 16, "[8; 16]", 4.0, 0.25, "tvm"])
        writer.writerow(["x", 16, "[x; 16]", 2.0, 0.5, "mlir"])
        writer.writerow([8, 32, "[8; 32]", "bad", 0.5, "mlir"])
        writer.writerow([4, 4, "[4; 4]", 0, 0, "mlir"])
        writer.writerow([2, 3, "[2; 3]", 1.0, 1.0, "mlir"])

    assert read_csv_results(str(output), sample_names, "mlir") == [
        ([8, 16], 2.0),
        ([2, 3], 1.0),
    ]
    assert read_csv_results(str(output), sample_names, "tvm") == [([8, 16], 4.0)]


def test_read_csv_results_missing_or_empty_file(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    empty = tmp_path / "empty.csv"
    empty.touch()

    assert read_csv_results(str(missing), ["M"], "mlir") == []
    assert read_csv_results(str(empty), ["M"], "mlir") == []


def _run_iterative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trials: int,
    resume: bool,
    optimizer: str = "random-forest-default",
) -> list[list[int]]:
    """Runs an iterative search with a fake evaluation, returns evaluated samples."""
    evaluated: list[list[int]] = []

    def fake_evaluate_all_parallel(self, strategy, all_in_x, graph, callbacks={}):
        results = []
        for x in all_in_x.tolist():
            evaluated.append(x)
            result = (x, 0, 1.0 + sum(x) % 7, "mlir")
            for callback in callbacks["result"]:
                callback(result)
            results.append(result)
        return results

    monkeypatch.setattr(
        Exploration, "evaluate_all_parallel", fake_evaluate_all_parallel
    )
    config = tmp_path / "optimizer.yaml"
    config.write_text("n_estimators: 5\nbatch_candidates: 50\n")
    exploration = Exploration(
        ExplorationConfig(
            node="matmul",
            search="iterative",
            optimizer=optimizer,
            optimizer_config=str(config),
            trials=trials,
            batch=4,
            output=str(tmp_path / "results.csv"),
            resume=resume,
            peak_flops=1e9,
            quiet=True,
        )
    )
    graph = _matmul_relu_graph()
    strategy = exploration.get_strategy(graph)
    exploration.search_some(strategy, graph)
    exploration.config.csv_callback = None
    return evaluated


def test_iterative_resume_runs_remaining_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _run_iterative(tmp_path, monkeypatch, trials=8, resume=False)
    assert len(first) == 8

    second = _run_iterative(tmp_path, monkeypatch, trials=16, resume=True)
    # Only the remaining trials are evaluated.
    assert len(second) == 16 - 8

    # Resume mode only records samples that are not already in the output.
    names = ["x0", "x1", "x2"]
    recorded = read_csv_results(str(tmp_path / "results.csv"), names, "mlir")
    new = {tuple(x) for x in second} - {tuple(x) for x in first}
    assert len(recorded) == len(first) + len(new)


@pytest.mark.parametrize("optimizer", ["random-forest-default", "random-forest-ucb"])
@pytest.mark.parametrize("pause", [4, 6, 8, 16, 19])
def test_iterative_resume_matches_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer: str, pause: int
):
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"
    uninterrupted_dir.mkdir()
    interrupted_dir.mkdir()
    uninterrupted = _run_iterative(
        uninterrupted_dir, monkeypatch, trials=24, resume=False, optimizer=optimizer
    )
    first = _run_iterative(
        interrupted_dir, monkeypatch, trials=pause, resume=False, optimizer=optimizer
    )
    second = _run_iterative(
        interrupted_dir, monkeypatch, trials=24, resume=True, optimizer=optimizer
    )
    # Same samples before the pause, then the same suggestions after the resume,
    # which requires both the sampling state and the model state to be restored.
    # An incomplete last batch (batch is 4) is evaluated again on resume.
    rewind = pause - pause % 4
    assert first == uninterrupted[:pause]
    assert second == uninterrupted[rewind:]

    names = ["x0", "x1", "x2"]
    assert read_csv_results(
        str(interrupted_dir / "results.csv"), names, "mlir"
    ) == read_csv_results(str(uninterrupted_dir / "results.csv"), names, "mlir")


def test_iterative_resume_with_all_trials_done_evaluates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _run_iterative(tmp_path, monkeypatch, trials=8, resume=False)
    again = _run_iterative(tmp_path, monkeypatch, trials=8, resume=True)
    assert again == []
