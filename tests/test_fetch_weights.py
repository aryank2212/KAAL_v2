"""
test_fetch_weights.py — Weights come back from the Hub verified, or not at all.
"""

import hashlib
import json
from pathlib import Path

import pytest

from MODEL.fetch_weights import fetch, targets

REVISION = "a" * 40
PTH, ONNX = b"pytorch-weights", b"onnx-graph"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(**hub):
    return {
        "id": "m-v1",
        "weights": "runs/release/m-v1/m-v1.pth",
        "sha256": _sha(PTH),
        "onnx": "runs/export/m-v1/m-v1.onnx",
        "onnx_sha256": _sha(ONNX),
        "hub": {"repo_id": "someone/kaal-test", "tag": "v1", "revision": REVISION, **hub},
    }


class _FakeHub:
    """Stands in for hf_hub_download: writes the repo file into local_dir."""

    def __init__(self, files):
        self.files = files
        self.calls = []

    def __call__(self, repo_id, filename, *, revision, local_dir, token=None):
        self.calls.append((repo_id, filename, revision))
        path = Path(local_dir) / filename
        path.write_bytes(self.files[filename])
        return str(path)


def _hub_files(onnx_sha=None):
    sidecar = json.dumps({"class_names": ["a"], "onnx_sha256": onnx_sha or _sha(ONNX)}).encode()
    return {"m-v1.pth": PTH, "m-v1.onnx": ONNX, "m-v1.onnx.json": sidecar}


def test_downloads_pinned_revision_into_registry_paths(tmp_path):
    hub = _FakeHub(_hub_files())
    written = fetch(_entry(), root=tmp_path, download=hub)
    assert [p.relative_to(tmp_path).as_posix() for p in written] == [
        "runs/release/m-v1/m-v1.pth",
        "runs/export/m-v1/m-v1.onnx",
        "runs/export/m-v1/m-v1.onnx.json",
    ]
    assert written[0].read_bytes() == PTH and written[1].read_bytes() == ONNX
    assert {rev for _, _, rev in hub.calls} == {REVISION}
    assert not list(tmp_path.rglob(".fetch-*")), "temporary download dirs must be removed"


def test_hash_mismatch_raises_and_writes_nothing(tmp_path):
    files = _hub_files()
    files["m-v1.pth"] = b"tampered"
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(_entry(), fmt="pth", root=tmp_path, download=_FakeHub(files))
    assert not (tmp_path / "runs/release/m-v1/m-v1.pth").exists()
    assert not list(tmp_path.rglob(".fetch-*"))


def test_sidecar_for_another_onnx_is_rejected(tmp_path):
    hub = _FakeHub(_hub_files(onnx_sha="0" * 64))
    with pytest.raises(ValueError, match="different ONNX"):
        fetch(_entry(), fmt="onnx", root=tmp_path, download=hub)


def test_verified_file_on_disk_is_not_downloaded_again(tmp_path):
    dest = tmp_path / "runs/release/m-v1/m-v1.pth"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PTH)
    hub = _FakeHub({})  # any download would raise KeyError
    assert fetch(_entry(), fmt="pth", root=tmp_path, download=hub) == [dest]
    assert hub.calls == []


def test_stale_file_on_disk_is_replaced(tmp_path):
    dest = tmp_path / "runs/release/m-v1/m-v1.pth"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old")
    fetch(_entry(), fmt="pth", root=tmp_path, download=_FakeHub(_hub_files()))
    assert dest.read_bytes() == PTH


def test_format_selects_files():
    entry = _entry()
    assert [t[0] for t in targets(entry, "pth")] == ["m-v1.pth"]
    assert [t[0] for t in targets(entry, "onnx")] == ["m-v1.onnx", "m-v1.onnx.json"]


def test_unpublished_model_is_an_error(tmp_path):
    entry = _entry()
    del entry["hub"]
    with pytest.raises(ValueError, match="no hub"):
        fetch(entry, root=tmp_path, download=_FakeHub({}))
