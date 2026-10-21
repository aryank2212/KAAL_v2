"""CLI contract tests for KAAL Bench: output directories, metrics, artifacts, errors."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from MODEL.bench import BenchUsageError, main, parse_metric_option, parse_metric_options

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli() -> int:
    """Invoke the bench CLI in-process and return its exit code."""
    try:
        main()
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


def capture_argv(output: Path, *extra: str) -> list[str]:
    return [
        "capture",
        "--run-id",
        "cli-fixture",
        "--model",
        "test-model",
        "--dataset",
        "fixture",
        "--split",
        "test",
        "--seed",
        "42",
        "--output",
        str(output),
        *extra,
    ]


def test_capture_creates_missing_output_directories(tmp_path, monkeypatch):
    output = tmp_path / "runs" / "bench" / "cli-fixture.json"
    monkeypatch.setattr(sys, "argv", ["bench", *capture_argv(output)])
    assert run_cli() == 0
    assert output.is_file()
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "cli-fixture"


def test_capture_records_metrics_and_artifacts(tmp_path, monkeypatch):
    artifact = tmp_path / "predictions.json"
    artifact.write_text('{"count": 0}\n', encoding="utf-8")
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            *capture_argv(
                output,
                "--metric",
                "map50=0.5",
                "--metric",
                "latency_p95_ms=null",
                "--artifact",
                str(artifact),
            ),
        ],
    )
    assert run_cli() == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["metrics"] == {"map50": 0.5, "latency_p95_ms": None}
    assert len(manifest["artifacts"]) == 1
    entry = manifest["artifacts"][0]
    assert entry["kind"] == "json"
    assert len(entry["sha256"]) == 64


def test_capture_output_is_deterministic(tmp_path, monkeypatch):
    artifact = tmp_path / "predictions.json"
    artifact.write_text('{"count": 0}\n', encoding="utf-8")
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    for target in (first, second):
        monkeypatch.setattr(
            sys,
            "argv",
            ["bench", *capture_argv(target, "--metric", "map50=0.5", "--artifact", str(artifact))],
        )
        assert run_cli() == 0
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("map50=0.5", ("map50", 0.5)),
        ("map50=1", ("map50", 1)),
        ("latency=null", ("latency", None)),
        ("latency=NONE", ("latency", None)),
    ],
)
def test_parse_metric_option(option, expected):
    assert parse_metric_option(option) == expected


@pytest.mark.parametrize("option", ["map50", "=0.5", "map50=abc", "map50=true", "map50=nan"])
def test_parse_metric_option_rejects_invalid_values(option):
    with pytest.raises(BenchUsageError):
        parse_metric_option(option)


def test_duplicate_metric_names_are_rejected():
    with pytest.raises(BenchUsageError):
        parse_metric_options(["map50=0.5", "map50=0.6"])


def test_invalid_metric_exits_with_a_clean_error(tmp_path, monkeypatch, capsys):
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(sys, "argv", ["bench", *capture_argv(output, "--metric", "map50=abc")])
    assert run_cli() == 2
    captured = capsys.readouterr()
    assert "bench: error:" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_missing_weights_file_exits_with_a_clean_error(tmp_path, monkeypatch, capsys):
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["bench", *capture_argv(output, "--weights", str(tmp_path / "missing.pth"))]
    )
    assert run_cli() == 2
    captured = capsys.readouterr()
    assert "Model weights does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_manifest_exits_with_a_clean_error(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 2}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["bench", "validate", str(bad)])
    assert run_cli() == 2
    captured = capsys.readouterr()
    assert "bench: error:" in captured.err
    assert "Traceback" not in captured.err


def test_module_entry_point_writes_a_manifest(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "MODEL.bench",
            *capture_argv(output, "--metric", "map50=0.5"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["metrics"] == {"map50": 0.5}
