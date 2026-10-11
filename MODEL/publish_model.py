"""
publish_model.py — Release a registry model to the Hugging Face Hub.

Status: active (Phase 4.7). Weights are not committed to git; the Hub repo is
where a fresh clone gets them (``python -m MODEL.fetch_weights``). Three
steps, each runnable on its own:

  strip   Write a release checkpoint from a training checkpoint: model
          weights, class names and architecture config only. Optimizer, EMA
          and loop state are dropped and absolute paths are rewritten
          repo-relative, so the file leaks nothing about the training machine
          and loads with ``torch.load(weights_only=True)`` (no pickle code).
  stage   Check every release file against the SHA-256 in
          ``MODEL/datasets_registry.yaml`` and render the Hub model card from
          ``MODEL/hub/model_card_template.md`` into ``MODEL/runs/hub/<model>/``.
  upload  Stage, then create the Hub repo (private), switch on gated access,
          upload everything in one commit, tag it, and print the ``hub:``
          block to record in the registry. ``--public`` flips visibility
          last, after gating is on.

    python -m MODEL.publish_model strip \\
        --src MODEL/runs/train/rfdetr-small-v4/checkpoint_best_total.pth \\
        --dst MODEL/runs/release/rfdetr-small-v4/rfdetr-small-v4.pth
    python -m MODEL.publish_model stage --repo-id <user>/kaal-rfdetr-small
    python -m MODEL.publish_model upload --repo-id <user>/kaal-rfdetr-small   # tag v4

Needs ``huggingface_hub`` (upload) and ``torch`` (strip) — the ``ml`` extra —
and a Hub login with write access (``hf auth login``).
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .detectors import REGISTRY, REPO_ROOT, model_entry
from .export_model import sha256

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
CARD_TEMPLATE = HERE / "hub" / "model_card_template.md"
DEFAULT_REPO_NAME = "kaal-rfdetr-small"

# What RFDETR.from_checkpoint needs; everything else in a training checkpoint
# (state_dict copy, EMA callback, optimizer, loops) is training state.
KEEP_KEYS = ("model", "args", "model_name", "model_config", "rfdetr_version", "best_total_source")
# model_config fields that describe the training machine, not the model:
# from_checkpoint sets pretrain_weights to the file it loads, and device
# falls back to the loader's default.
DROP_CONFIG_KEYS = ("pretrain_weights", "device")


def _relativize(obj: Any, root: str, rewritten: List[str], where: str = "") -> Any:
    """Rewrite strings under ``root`` to repo-relative paths, recursively."""
    if isinstance(obj, str) and obj.startswith(root):
        rewritten.append(where)
        return obj[len(root) :].lstrip("/\\") or "."
    if isinstance(obj, dict):
        return {k: _relativize(v, root, rewritten, f"{where}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _relativize(v, root, rewritten, f"{where}[{i}]") for i, v in enumerate(obj)
        )
    return obj


def strip_checkpoint(src: Path, dst: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Write the inference-only release checkpoint for ``src`` to ``dst``."""
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "args" not in ckpt:
        raise ValueError(f"{src} is not an RF-DETR inference checkpoint (needs 'model' and 'args')")
    release = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
    if isinstance(release.get("model_config"), dict):
        release["model_config"] = {
            k: v for k, v in release["model_config"].items() if k not in DROP_CONFIG_KEYS
        }
    rewritten: List[str] = []
    for key in ("args", "model_config"):
        if key in release:
            release[key] = _relativize(release[key], str(repo_root), rewritten, key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, dst)
    torch.load(dst, map_location="cpu", weights_only=True)  # must load without pickle code
    leaks = local_path_leaks(dst, repo_root)
    if leaks:
        dst.unlink()
        raise ValueError(f"release checkpoint still contains local paths: {leaks}")
    return {
        "dropped": sorted(set(ckpt) - set(release)),
        "dropped_config": [k for k in DROP_CONFIG_KEYS if k in (ckpt.get("model_config") or {})],
        "rewritten": rewritten,
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
    }


def local_path_leaks(path: Path, repo_root: Path = REPO_ROOT) -> List[str]:
    """Machine-specific strings (repo root, home directory) found in a file's bytes."""
    data = path.read_bytes()
    markers = {str(repo_root), str(Path.home())}
    return sorted(m for m in markers if len(m) > 1 and m.encode() in data)


def default_tag(model_id: str) -> Optional[str]:
    """``rfdetr-small-v4`` -> ``v4``."""
    m = re.search(r"-(v\d+)$", model_id)
    return m.group(1) if m else None


def release_files(entry: Dict[str, Any], root: Path = REPO_ROOT) -> List[Tuple[str, Path]]:
    """(path in the Hub repo, local file) for a registry model, hashes verified."""
    files: List[Tuple[str, Path]] = []
    checks = [("weights", "sha256")]
    if entry.get("onnx"):
        checks.append(("onnx", "onnx_sha256"))
    for key, sha_key in checks:
        path = root / entry[key]
        if not path.exists():
            raise FileNotFoundError(f"{entry['id']}: {key} file {path} is missing")
        expected = entry.get(sha_key)
        if not expected:
            raise ValueError(f"{entry['id']}: registry has no {sha_key}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{entry['id']}: {path} SHA-256 {actual} != registry {expected}")
        files.append((path.name, path))
    if entry.get("onnx"):
        meta_path = root / f"{entry['onnx']}.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("onnx_sha256") != entry["onnx_sha256"]:
            raise ValueError(f"{meta_path} describes a different ONNX file")
        files.append((meta_path.name, meta_path))
    return files


def render_card(
    entry: Dict[str, Any], repo_id: str, tag: str, template: Path = CARD_TEMPLATE
) -> str:
    """Fill the Hub model card template with the release's names and hashes."""
    values = {
        "REPO_ID": repo_id,
        "TAG": tag,
        "MODEL_ID": entry["id"],
        "WEIGHTS_FILE": Path(entry["weights"]).name,
        "WEIGHTS_SHA256": entry["sha256"],
        "ONNX_FILE": Path(entry["onnx"]).name,
        "ONNX_SHA256": entry["onnx_sha256"],
    }
    card = template.read_text(encoding="utf-8")
    for key, value in values.items():
        card = card.replace("{{" + key + "}}", value)
    left = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", card)))
    if left:
        raise ValueError(f"model card template has unfilled placeholders: {left}")
    return card


def stage(
    entry: Dict[str, Any], repo_id: str, tag: str, out_dir: Path, root: Path = REPO_ROOT
) -> List[Tuple[str, Path]]:
    """Verify the release files and write the rendered card; returns the upload plan."""
    files = release_files(entry, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    card = out_dir / "README.md"
    card.write_text(render_card(entry, repo_id, tag), encoding="utf-8")
    return [("README.md", card)] + files


def upload(
    api: Any,
    plan: List[Tuple[str, Path]],
    repo_id: str,
    tag: str,
    gated: Optional[str] = "auto",
    public: bool = False,
    message: str = "",
) -> Dict[str, str]:
    """Create/update the Hub repo from ``plan``; returns the registry ``hub`` block."""
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if gated:
        api.update_repo_settings(repo_id, gated=gated)
    info = api.create_commit(
        repo_id,
        operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for p, f in plan],
        commit_message=message or f"Release {tag}",
    )
    api.create_tag(repo_id, tag=tag, revision=info.oid, tag_message=message or None)
    if public:
        api.update_repo_settings(repo_id, private=False)
    return {"repo_id": repo_id, "tag": tag, "revision": info.oid}


def main() -> None:
    parser = argparse.ArgumentParser(description="Release a KAAL model to the Hugging Face Hub.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_strip = sub.add_parser("strip", help="write an inference-only release checkpoint")
    p_strip.add_argument("--src", required=True, help="training checkpoint (.pth)")
    p_strip.add_argument("--dst", required=True, help="release checkpoint to write")
    for name in ("stage", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=None, help="registry model id (default: production)")
        p.add_argument("--repo-id", default=None, help=f"default: <hub user>/{DEFAULT_REPO_NAME}")
        p.add_argument("--tag", default=None, help="Hub tag (default: from the model id, e.g. v4)")
        p.add_argument("--out", default=None, help="staging dir (default: MODEL/runs/hub/<model>)")
    p_up = sub.choices["upload"]
    p_up.add_argument(
        "--gated", choices=("auto", "manual", "none"), default="auto", help="Hub access requests"
    )
    p_up.add_argument("--public", action="store_true", help="make the repo public after upload")
    p_up.add_argument("--message", default="", help="commit and tag message")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "strip":
        src, dst = Path(args.src), Path(args.dst)
        if dst.exists():
            parser.error(f"{dst} exists; move it away first")
        report = strip_checkpoint(src, dst)
        logger.info("Wrote %s", dst)
        print(json.dumps(report, indent=2))
        return

    entry = model_entry(args.model, REGISTRY)
    if entry is None:
        parser.error(f"no registry model {args.model or '(production)'}")
    tag = args.tag or default_tag(entry["id"])
    if not tag:
        parser.error("--tag is required when the model id has no -vN suffix")
    api = None
    repo_id = args.repo_id
    if args.cmd == "upload" or not repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = repo_id or f"{api.whoami()['name']}/{DEFAULT_REPO_NAME}"
    out = Path(args.out) if args.out else HERE / "runs" / "hub" / entry["id"]
    plan = stage(entry, repo_id, tag, out)
    for path_in_repo, local in plan:
        logger.info("%-28s <- %s (%.1f MB)", path_in_repo, local, local.stat().st_size / 1e6)
    if args.cmd == "stage":
        logger.info("Card rendered to %s; run 'upload' to publish", out / "README.md")
        return
    hub = upload(
        api,
        plan,
        repo_id,
        tag,
        gated=None if args.gated == "none" else args.gated,
        public=args.public,
        message=args.message or f"{entry['id']}: KAAL release {tag}",
    )
    logger.info("Published https://huggingface.co/%s (tag %s)", repo_id, tag)
    print("Record in MODEL/datasets_registry.yaml under the model entry:")
    print(f"    hub:\n      repo_id: {hub['repo_id']}\n      tag: {hub['tag']}")
    print(f"      revision: {hub['revision']}")


if __name__ == "__main__":
    main()
