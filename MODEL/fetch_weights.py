"""
fetch_weights.py — Download a registry model's weights from the Hugging Face Hub.

Status: active (Phase 4.7). Weights are not committed to git. Each model in
``MODEL/datasets_registry.yaml`` that has a ``hub:`` block can be restored
into the paths the registry names, so ``production_weights()`` (and with it
the backend) finds them on a fresh clone:

    python -m MODEL.fetch_weights                    # production model: .pth + .onnx
    python -m MODEL.fetch_weights --format onnx      # ONNX only (runs without torch)
    python -m MODEL.fetch_weights --model rfdetr-small-v4

Files are downloaded at the pinned Hub commit (``hub.revision``) and checked
against the SHA-256 recorded in the registry before they are moved into
place; a file already on disk with the right hash is left alone. The Hub
repo is gated: accept the terms on its page once, then log in with
``hf auth login``.

Needs ``huggingface_hub`` (``pip install huggingface_hub``).
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detectors import REGISTRY, REPO_ROOT, model_entry
from .export_model import sha256

logger = logging.getLogger(__name__)

# (file name in the Hub repo, local path relative to the repo root, expected SHA-256 or None)
Target = Tuple[str, str, Optional[str]]


def targets(entry: Dict[str, Any], fmt: str = "all") -> List[Target]:
    """Files to fetch for a registry model; ``fmt`` is "all", "pth" or "onnx"."""
    out: List[Target] = []
    if fmt in ("all", "pth"):
        out.append((Path(entry["weights"]).name, entry["weights"], entry["sha256"]))
    if fmt in ("all", "onnx") and entry.get("onnx"):
        out.append((Path(entry["onnx"]).name, entry["onnx"], entry["onnx_sha256"]))
        out.append((f"{Path(entry['onnx']).name}.json", f"{entry['onnx']}.json", None))
    return out


def _check(path: Path, expected: Optional[str], entry: Dict[str, Any]) -> None:
    if expected is not None:
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"{path.name}: SHA-256 {actual} does not match the registry {expected}"
            )
    else:  # ONNX sidecar: must describe the registry's ONNX file
        meta = json.loads(path.read_text())
        if meta.get("onnx_sha256") != entry.get("onnx_sha256"):
            raise ValueError(f"{path.name} describes a different ONNX file than the registry")


def fetch(
    entry: Dict[str, Any],
    fmt: str = "all",
    root: Path = REPO_ROOT,
    download: Optional[Callable[..., str]] = None,
    token: Optional[str] = None,
) -> List[Path]:
    """Download and verify a model's files into the registry paths; returns them."""
    hub = entry.get("hub")
    if not hub:
        raise ValueError(f"registry model {entry['id']} has no hub: block (not published)")
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    written = []
    for filename, rel, expected in targets(entry, fmt):
        dest = root / rel
        if dest.exists():
            try:
                _check(dest, expected, entry)
                logger.info("%s already present and verified", dest)
                written.append(dest)
                continue
            except ValueError:
                logger.info("%s differs from the registry; downloading again", dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".fetch-") as tmp:
            got = Path(
                download(
                    hub["repo_id"],
                    filename,
                    revision=hub["revision"],
                    local_dir=tmp,
                    token=token,
                )
            )
            _check(got, expected, entry)
            os.replace(got, dest)
        logger.info("Fetched %s -> %s", filename, dest)
        written.append(dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KAAL model weights from the Hub.")
    parser.add_argument("--model", default=None, help="registry model id (default: production)")
    parser.add_argument("--format", choices=("all", "pth", "onnx"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    entry = model_entry(args.model, REGISTRY)
    if entry is None:
        parser.error(f"no registry model {args.model or '(production)'}")
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    page = f"https://huggingface.co/{(entry.get('hub') or {}).get('repo_id', '')}"
    try:
        fetch(entry, args.format)
    except GatedRepoError:
        parser.exit(
            1, f"Access not granted yet: accept the terms at {page}, then run 'hf auth login'.\n"
        )
    except RepositoryNotFoundError:
        parser.exit(
            1,
            f"{page} not found, or private to its owner: log in with an account that can see it.\n",
        )
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
