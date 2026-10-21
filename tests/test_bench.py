"""CPU-only contract tests for KAAL Bench manifest validation and reports."""
import json
from pathlib import Path
import pytest
from MODEL.bench import (
    ManifestValidationError, canonical_json, capture_manifest,
    render_markdown, sha256_file, validate_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "benchmark_results" / "v1_manifest.json"


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_validates_and_reports_are_deterministic():
    data = fixture()
    validate_manifest(data)
    assert canonical_json(data) == canonical_json(data)
    assert render_markdown(data) == render_markdown(data)
    assert canonical_json(data).endswith("\n")
    assert "not measured" in render_markdown(data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(unexpected=True), "unknown keys"),
        (lambda d: d["protocol"].update(seed=True), "non-negative integer"),
        (lambda d: d["metrics"].update(latency=float("nan")), "finite number"),
        (lambda d: d["model"].update(weights_sha256="bad"), "SHA-256"),
        (lambda d: d["dataset"].update(extra="no"), "unknown keys"),
        (lambda d: d["artifacts"].append({"path": "x"}), "missing keys"),
    ],
)
def test_invalid_manifests_are_rejected(mutate, message):
    data = fixture()
    mutate(data)
    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest(data)


def test_capture_hashes_inputs_and_records_environment(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"fixture weights")
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"images": []}\n', encoding="utf-8")
    artifact = tmp_path / "predictions.json"
    artifact.write_text('{"count": 0}\n', encoding="utf-8")
    data = capture_manifest(
        run_id="cpu-fixture", model_name="test-model", dataset_id="fixture",
        split="test", seed=42, model_weights=weights, dataset_manifest=dataset,
        artifacts=[artifact], repository=tmp_path, metrics={"map50_95": None},
    )
    assert data["model"]["weights_sha256"] == sha256_file(weights)
    assert data["dataset"]["manifest_sha256"] == sha256_file(dataset)
    assert data["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert data["git_commit"] is None
    assert data["environment"]["python_version"]
    validate_manifest(data)


def test_null_is_not_rendered_as_zero():
    report = render_markdown(fixture())
    assert "not measured" in report
    assert "| latency_p95_ms | 0 |" not in report
