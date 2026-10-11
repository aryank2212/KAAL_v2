"""
test_publish_model.py — A Hub release ships only verified, machine-neutral files.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from MODEL.publish_model import (
    CARD_TEMPLATE,
    default_tag,
    local_path_leaks,
    release_files,
    render_card,
    upload,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(tmp_path, onnx_sha=None):
    (tmp_path / "rel").mkdir()
    (tmp_path / "rel" / "m-v1.pth").write_bytes(b"w")
    (tmp_path / "rel" / "m-v1.onnx").write_bytes(b"o")
    sidecar = {"class_names": ["a"], "onnx_sha256": onnx_sha or _sha(b"o")}
    (tmp_path / "rel" / "m-v1.onnx.json").write_text(json.dumps(sidecar))
    return {
        "id": "m-v1",
        "weights": "rel/m-v1.pth",
        "sha256": _sha(b"w"),
        "onnx": "rel/m-v1.onnx",
        "onnx_sha256": _sha(b"o"),
    }


def test_default_tag_comes_from_the_model_id():
    assert default_tag("rfdetr-small-v4") == "v4"
    assert default_tag("rfdetr-small") is None


def test_release_files_are_hash_checked(tmp_path):
    entry = _release(tmp_path)
    assert [name for name, _ in release_files(entry, tmp_path)] == [
        "m-v1.pth",
        "m-v1.onnx",
        "m-v1.onnx.json",
    ]
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        release_files(entry, tmp_path)


def test_missing_file_and_foreign_sidecar_are_errors(tmp_path):
    entry = _release(tmp_path, onnx_sha="0" * 64)
    with pytest.raises(ValueError, match="different ONNX"):
        release_files(entry, tmp_path)
    (tmp_path / "rel" / "m-v1.pth").unlink()
    with pytest.raises(FileNotFoundError):
        release_files(entry, tmp_path)


def test_card_template_renders_completely():
    entry = {
        "id": "rfdetr-small-v4",
        "weights": "x/rfdetr-small-v4.pth",
        "sha256": "1" * 64,
        "onnx": "y/rfdetr-small-v4.onnx",
        "onnx_sha256": "2" * 64,
    }
    card = render_card(entry, "someone/kaal-rfdetr-small", "v4")
    assert "{{" not in card
    assert "someone/kaal-rfdetr-small" in card and "1" * 64 in card and "2" * 64 in card
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["license"] == "cc-by-nc-4.0"
    assert meta["pipeline_tag"] == "object-detection"
    assert all(v == "checkbox" for v in meta["extra_gated_fields"].values())
    assert meta["model-index"][0]["name"] == "rfdetr-small-v4"


def test_card_template_is_the_committed_file():
    assert CARD_TEMPLATE.exists()


def test_card_metadata_keys_the_hub_rejects_are_caught(tmp_path):
    template = tmp_path / "card.md"
    template.write_text(
        "---\nextra_gated_fields:\n  I accept CC BY-NC 4.0: checkbox\n---\n# {{MODEL_ID}}\n"
    )
    entry = {"id": "m", "weights": "m.pth", "sha256": "1", "onnx": "m.onnx", "onnx_sha256": "2"}
    with pytest.raises(ValueError, match="must not contain"):
        render_card(entry, "someone/repo", "v1", template)


class _FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", kw))

    def update_repo_settings(self, repo_id, **kw):
        self.calls.append(("update_repo_settings", kw))

    def create_commit(self, repo_id, operations, commit_message):
        self.calls.append(("create_commit", [op.path_in_repo for op in operations]))
        return SimpleNamespace(oid="c" * 40)

    def create_tag(self, repo_id, **kw):
        self.calls.append(("create_tag", kw))


def test_upload_is_private_and_gated_before_any_file_lands(tmp_path):
    pytest.importorskip("huggingface_hub")
    f = tmp_path / "README.md"
    f.write_text("card")
    api = _FakeApi()
    hub = upload(api, [("README.md", f)], "someone/repo", "v1", gated="auto")
    assert hub == {"repo_id": "someone/repo", "tag": "v1", "revision": "c" * 40}
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "update_repo_settings", "create_commit", "create_tag"]
    assert api.calls[0][1]["private"] is True
    assert api.calls[1][1] == {"gated": "auto"}
    assert api.calls[3][1]["revision"] == "c" * 40


def test_public_flag_flips_visibility_last(tmp_path):
    pytest.importorskip("huggingface_hub")
    f = tmp_path / "README.md"
    f.write_text("card")
    api = _FakeApi()
    upload(api, [("README.md", f)], "someone/repo", "v1", gated="auto", public=True)
    assert api.calls[-1] == ("update_repo_settings", {"private": False})


def test_strip_keeps_inference_state_and_drops_machine_details(tmp_path):
    torch = pytest.importorskip("torch")
    from MODEL.publish_model import strip_checkpoint

    root = tmp_path / "repo"
    src, dst = tmp_path / "train.pth", tmp_path / "out" / "release.pth"
    torch.save(
        {
            "model": {"w": torch.ones(3)},
            "args": {"dataset_dir": f"{root}/MODEL/Datasets/d", "class_names": ["tank"]},
            "model_config": {
                "pretrain_weights": f"{root}/p.pth",
                "device": "cuda",
                "resolution": 512,
            },
            "model_name": "RFDETRSmall",
            "state_dict": {"model.w": torch.ones(3)},
            "optimizer_states": [{"lr": 1.0}],
            "callbacks": {"ema": {"n": 1}},
        },
        src,
    )
    report = strip_checkpoint(src, dst, repo_root=root)
    out = torch.load(dst, weights_only=True)
    assert set(out) == {"model", "args", "model_config", "model_name"}
    assert out["args"]["dataset_dir"] == "MODEL/Datasets/d"
    assert out["model_config"] == {"resolution": 512}
    assert report["rewritten"] == ["args.dataset_dir"]
    assert local_path_leaks(dst, root) == []


def test_strip_refuses_a_checkpoint_that_still_leaks_a_path(tmp_path):
    torch = pytest.importorskip("torch")
    from MODEL.publish_model import strip_checkpoint

    src, dst = tmp_path / "train.pth", tmp_path / "release.pth"
    leaked = str(Path.home() / "secret-run")
    torch.save({"model": {}, "args": {"log_dir": leaked}}, src)
    with pytest.raises(ValueError, match="local paths"):
        strip_checkpoint(src, dst, repo_root=tmp_path / "repo")
    assert not dst.exists()
