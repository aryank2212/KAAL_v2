"""Versioned experiment metadata and deterministic reports for KAAL Bench."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")
TOP_KEYS = {
    "schema_version",
    "run_id",
    "git_commit",
    "model",
    "dataset",
    "protocol",
    "environment",
    "metrics",
    "artifacts",
}


class ManifestValidationError(ValueError):
    """Invalid or unsupported experiment manifest."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        value = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            .stdout.strip()
            .lower()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return value if GIT_SHA_RE.fullmatch(value) else None


def _keys(obj: dict[str, Any], expected: set[str], label: str) -> None:
    missing, extra = expected - obj.keys(), obj.keys() - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing keys: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("unknown keys: " + ", ".join(sorted(extra)))
        raise ManifestValidationError(f"{label}: {'; '.join(parts)}")


def _string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{label} must be a non-empty string")


def _hash(value: Any, label: str, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ManifestValidationError(f"{label} must be a lowercase SHA-256 hex digest")


def validate_manifest(data: dict[str, Any]) -> None:
    """Strict schema-v1 validation; unknown fields are rejected."""
    if not isinstance(data, dict):
        raise ManifestValidationError("manifest must be an object")
    _keys(data, TOP_KEYS, "manifest")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ManifestValidationError(f"schema_version must be {SCHEMA_VERSION}")
    _string(data["run_id"], "run_id")
    commit = data["git_commit"]
    if commit is not None and (not isinstance(commit, str) or not GIT_SHA_RE.fullmatch(commit)):
        raise ManifestValidationError("git_commit must be a full lowercase Git SHA or null")

    model = data["model"]
    if not isinstance(model, dict):
        raise ManifestValidationError("model must be an object")
    _keys(model, {"name", "weights_sha256"}, "model")
    _string(model["name"], "model.name")
    _hash(model["weights_sha256"], "model.weights_sha256", nullable=True)

    dataset = data["dataset"]
    if not isinstance(dataset, dict):
        raise ManifestValidationError("dataset must be an object")
    _keys(dataset, {"registry_id", "split", "manifest_sha256"}, "dataset")
    _string(dataset["registry_id"], "dataset.registry_id")
    _string(dataset["split"], "dataset.split")
    _hash(dataset["manifest_sha256"], "dataset.manifest_sha256", nullable=True)

    protocol = data["protocol"]
    if not isinstance(protocol, dict):
        raise ManifestValidationError("protocol must be an object")
    _keys(protocol, {"seed", "evaluation_version"}, "protocol")
    if type(protocol["seed"]) is not int or protocol["seed"] < 0:
        raise ManifestValidationError("protocol.seed must be a non-negative integer")
    _string(protocol["evaluation_version"], "protocol.evaluation_version")

    env = data["environment"]
    if not isinstance(env, dict):
        raise ManifestValidationError("environment must be an object")
    _keys(env, {"python_version", "platform"}, "environment")
    _string(env["python_version"], "environment.python_version")
    _string(env["platform"], "environment.platform")

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise ManifestValidationError("metrics must be an object")
    for name, value in metrics.items():
        _string(name, "metric name")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ManifestValidationError(f"metrics.{name} must be a finite number or null")

    artifacts = data["artifacts"]
    if not isinstance(artifacts, list):
        raise ManifestValidationError("artifacts must be a list")
    for i, item in enumerate(artifacts):
        label = f"artifacts[{i}]"
        if not isinstance(item, dict):
            raise ManifestValidationError(f"{label} must be an object")
        _keys(item, {"path", "sha256", "kind"}, label)
        _string(item["path"], f"{label}.path")
        _hash(item["sha256"], f"{label}.sha256")
        _string(item["kind"], f"{label}.kind")


def capture_manifest(
    *,
    run_id: str,
    model_name: str,
    dataset_id: str,
    split: str,
    seed: int,
    evaluation_version: str = "1",
    model_weights: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    metrics: dict[str, float | int | None] | None = None,
    artifacts: list[str | Path] | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Capture provenance; does not execute an evaluator."""
    entries = []
    for path in sorted((Path(p) for p in (artifacts or [])), key=lambda p: p.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(f"Artifact does not exist or is not a file: {path}")
        entries.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "kind": path.suffix.lstrip(".").lower() or "file",
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": _git_commit(repository),
        "model": {
            "name": model_name,
            "weights_sha256": sha256_file(model_weights) if model_weights else None,
        },
        "dataset": {
            "registry_id": dataset_id,
            "split": split,
            "manifest_sha256": sha256_file(dataset_manifest) if dataset_manifest else None,
        },
        "protocol": {"seed": seed, "evaluation_version": evaluation_version},
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "metrics": dict(metrics or {}),
        "artifacts": entries,
    }
    validate_manifest(manifest)
    return manifest


def canonical_json(manifest: dict[str, Any]) -> str:
    validate_manifest(manifest)
    return (
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )


def render_markdown(manifest: dict[str, Any]) -> str:
    validate_manifest(manifest)
    model, dataset = manifest["model"], manifest["dataset"]
    protocol, env = manifest["protocol"], manifest["environment"]
    lines = [
        f"# KAAL experiment: {manifest['run_id']}",
        "",
        f"- **Schema version:** {manifest['schema_version']}",
        f"- **Git commit:** {manifest['git_commit'] or 'unavailable'}",
        f"- **Model:** {model['name']}",
        f"- **Model weights SHA-256:** {model['weights_sha256'] or 'unavailable'}",
        f"- **Dataset:** {dataset['registry_id']} ({dataset['split']})",
        f"- **Dataset manifest SHA-256:** {dataset['manifest_sha256'] or 'unavailable'}",
        f"- **Seed:** {protocol['seed']}",
        f"- **Evaluation version:** {protocol['evaluation_version']}",
        f"- **Python:** {env['python_version']}",
        f"- **Platform:** {env['platform']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    if manifest["metrics"]:
        for name in sorted(manifest["metrics"]):
            value = manifest["metrics"][name]
            display = "not measured" if value is None else str(value)
            lines.append(f"| {name.replace('|', chr(92) + '|')} | {display} |")
    else:
        lines.append("| No metrics recorded | not measured |")
    lines.extend(["", "## Artifacts", ""])
    if manifest["artifacts"]:
        for item in sorted(manifest["artifacts"], key=lambda value: value["path"]):
            lines.append(f"- {item['path']} ({item['kind']}, SHA-256: {item['sha256']})")
    else:
        lines.append("- None recorded")
    lines.extend(["", "> Null metrics mean not measured; they are not zero.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="KAAL experiment manifest tools")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--model", required=True, dest="model_name")
    capture.add_argument("--dataset", required=True, dest="dataset_id")
    capture.add_argument("--split", required=True)
    capture.add_argument("--seed", required=True, type=int)
    capture.add_argument("--evaluation-version", default="1")
    capture.add_argument("--weights", type=Path)
    capture.add_argument("--dataset-manifest", type=Path)
    capture.add_argument("--repository", type=Path)
    capture.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    report = commands.add_parser("report")
    report.add_argument("manifest", type=Path)
    report.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "capture":
        result = capture_manifest(
            run_id=args.run_id,
            model_name=args.model_name,
            dataset_id=args.dataset_id,
            split=args.split,
            seed=args.seed,
            evaluation_version=args.evaluation_version,
            model_weights=args.weights,
            dataset_manifest=args.dataset_manifest,
            repository=args.repository,
        )
        Path(args.output).write_text(canonical_json(result), encoding="utf-8")
        print(f"Wrote {args.output}")
        return
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(data)
    if args.command == "validate":
        print(f"Valid schema-v{SCHEMA_VERSION} manifest: {args.manifest}")
    else:
        output = render_markdown(data)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            sys.stdout.write(output)


if __name__ == "__main__":
    main()
