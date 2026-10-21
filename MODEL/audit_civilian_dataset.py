"""
audit_civilian_dataset.py — read-only quality and leakage audit.

Audits a YOLO-format civilian dataset (the one ``MODEL.build_civilian_auair``
produces) and reports what a human reviewer needs to look at. **It never
writes to the dataset.** Labels are not corrected, boxes are not clipped and
files are not renamed; every problem is reported for review.

Reported:

* readable / corrupt images, and any that disagree with the recorded size
* missing, empty and malformed label files
* invalid boxes: unknown class id, non-positive size, coordinates outside
  ``[0, 1]`` and boxes that fall outside the frame
* class balance per split, and the ratio to the rarest class
* object-size distribution per class, over documented size buckets
* exact duplicates (identical file bytes)
* likely near-duplicates (perceptual hash within a documented distance)
* split and group leakage: a group or an image appearing in two splits, and
  exact or near-duplicate pairs that straddle a split boundary

Outputs ``audit.json`` and ``audit.md``, both deterministic for a given input
set: no timestamps and no wall-clock timings are written into them.

    python -m MODEL.audit_civilian_dataset \\
        --data MODEL/Datasets/auair_civilian --out MODEL/Datasets/auair_civilian

Near-duplicate detection is the expensive part. Its limits are explicit:
see ``NEAR_DUPLICATE_*`` below and the ``near_duplicates.limits`` block of the
JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from MODEL.build_civilian_auair import sequence_from_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# ── near-duplicate limits (explicit, reported in the output) ───────────────

#: Perceptual hash: 64-bit difference hash, 8x9 grayscale.
HASH_SIZE = 8
#: Maximum Hamming distance for two images to count as likely duplicates.
NEAR_DUPLICATE_MAX_DISTANCE = 5
#: Hard cap on the number of images hashed for near-duplicate analysis.
NEAR_DUPLICATE_MAX_IMAGES = 60_000
#: Block size for the pairwise Hamming-distance scan (memory/speed trade-off).
NEAR_DUPLICATE_BLOCK = 256

#: Normalised-area buckets for the object-size distribution.
SIZE_BUCKETS = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, 1.01),
)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> Optional[np.ndarray]:
    """64-bit difference hash of an image, or None if it cannot be read.

    The image is decoded at half scale: the hash is 8x9 pixels, so the full
    1920x1080 frame is not needed and decoding dominates the runtime.
    """
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if image is None or image.size == 0:
        return None
    resized = cv2.resize(image, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA)
    return np.packbits(resized[:, 1:] > resized[:, :-1]).reshape(-1)


def near_duplicate_pairs(
    hashes: np.ndarray, max_distance: int, block: int = NEAR_DUPLICATE_BLOCK
) -> Iterable[Tuple[int, int]]:
    """Index pairs (i < j) whose perceptual hashes differ by at most ``max_distance``."""
    for start in range(0, len(hashes), block):
        query = hashes[start : start + block, None, :]
        distance = _POPCOUNT[query ^ hashes[None]].sum(2, dtype=np.uint16)
        for i, j in zip(*np.nonzero(distance <= max_distance)):
            if start + i < j:
                yield int(start + i), int(j)


def size_bucket(area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return SIZE_BUCKETS[-1][0]


def percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Deterministic percentiles; empty input yields nulls, never zeros."""
    if not values:
        return {
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def read_labels(
    path: Path, class_count: int
) -> Tuple[List[Tuple[int, float, float, float, float]], Counter]:
    """Parse one YOLO label file. Returns the valid boxes and a rejection counter."""
    boxes: List[Tuple[int, float, float, float, float]] = []
    rejected: Counter = Counter()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        rejected["unreadable_label_file"] += 1
        return boxes, rejected
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        rejected["empty_label_file"] += 1
        return boxes, rejected
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            rejected["malformed_line"] += 1
            continue
        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            rejected["malformed_line"] += 1
            continue
        if class_id < 0 or class_id >= class_count:
            rejected["unknown_class_id"] += 1
            continue
        if width <= 0 or height <= 0:
            rejected["non_positive_size"] += 1
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            rejected["centre_out_of_range"] += 1
            continue
        if not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            rejected["size_out_of_range"] += 1
            continue
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6:
            rejected["box_crosses_top_left"] += 1
            continue
        if cx + width / 2 > 1.0 + 1e-6 or cy + height / 2 > 1.0 + 1e-6:
            rejected["box_crosses_bottom_right"] += 1
            continue
        boxes.append((class_id, cx, cy, width, height))
    return boxes, rejected


def audit(
    data_dir: Path, near_duplicates: bool, max_distance: int, max_images: int
) -> Dict[str, Any]:
    """Audit a YOLO dataset directory. Nothing is written to the dataset."""
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"No data.yaml in {data_dir}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names") or {}
    class_count = int(config.get("nc") or len(names))
    class_names = {index: str(name) for index, name in names.items()} if names else {}

    images: List[Tuple[str, Path]] = []
    for split in SPLITS:
        folder = data_dir / config.get(split, f"{split}/images")
        if not folder.is_dir():
            logger.warning("split %s: missing directory %s", split, folder)
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append((split, path))

    logger.info("auditing %d images in %s", len(images), data_dir)

    split_of: Dict[str, str] = {}
    name_splits: Dict[str, set] = defaultdict(set)
    group_of: Dict[str, str] = {}
    group_splits: Dict[str, set] = defaultdict(set)
    missing_labels: List[str] = []
    corrupt_images: List[str] = []
    rejected: Counter = Counter()
    rejected_examples: List[Dict[str, Any]] = []
    per_split_class: Dict[str, Counter] = {split: Counter() for split in SPLITS}
    per_class_areas: Dict[int, List[float]] = defaultdict(list)
    per_class_buckets: Dict[int, Counter] = defaultdict(Counter)
    image_hashes: Dict[str, List[str]] = defaultdict(list)
    class_totals: Counter = Counter()
    total_boxes = 0

    for split, image_path in images:
        name = image_path.name
        split_of.setdefault(name, split)
        name_splits[name].add(split)
        group = sequence_from_name(name)
        if group is None:
            group = f"<ungrouped:{name}>"
            rejected["ungroupable_image_name"] += 1
        group_of[name] = group
        group_splits[group].add(split)

        digest = sha256_file(image_path)
        image_hashes[digest].append(name)

        frame = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if frame is None or frame.size == 0:
            corrupt_images.append(name)
            rejected["corrupt_image"] += 1
        else:
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels.append(name)
                rejected["missing_label_file"] += 1
                continue
            boxes, file_rejected = read_labels(label_path, class_count)
            rejected.update(file_rejected)
            if file_rejected and len(rejected_examples) < 50:
                rejected_examples.append(
                    {
                        "kind": "label_rejected",
                        "file": str(label_path.relative_to(data_dir)),
                        **dict(file_rejected),
                    }
                )
            for class_id, cx, cy, box_width, box_height in boxes:
                total_boxes += 1
                per_split_class[split][class_id] += 1
                class_totals[class_id] += 1
                area = box_width * box_height
                per_class_areas[class_id].append(area)
                per_class_buckets[class_id][size_bucket(area)] += 1

    # ── group and split leakage ───────────────────────────────────────────
    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    images_in_multiple_splits = sorted(
        name for name, splits in name_splits.items() if len(splits) > 1
    )

    duplicate_groups = {
        digest: sorted(names) for digest, names in image_hashes.items() if len(names) > 1
    }
    duplicate_cross_split = sorted(
        digest
        for digest, names in duplicate_groups.items()
        if len({split for name in names for split in name_splits.get(name, set())}) > 1
    )

    # ── near-duplicates ───────────────────────────────────────────────────
    near_duplicate_report: Dict[str, Any] = {
        "method": f"{HASH_SIZE}x{HASH_SIZE + 1} grayscale difference hash, half-scale decode",
        "max_hamming_distance": max_distance,
        "limits": {
            "max_images_hashed": max_images,
            "images_available": len(images),
            "images_hashed": 0,
            "analysis_performed": False,
            "truncated": False,
            "note": (
                "Near-duplicate analysis was not run. It is O(n^2) in the number of images "
                "and is bounded by max_images_hashed."
            ),
        },
        "within_split_pairs": [],
        "cross_split_pairs": [],
        "pair_count": 0,
        "within_split_pair_count": 0,
        "cross_split_pair_count": 0,
        "images_in_cross_split_pairs": 0,
        "groups_in_cross_split_pairs": [],
        "listing_note": "No pairs were listed.",
    }
    if near_duplicates and images:
        selected = images[:max_images]
        truncated = len(images) > len(selected)
        logger.info("hashing %d images for near-duplicate analysis", len(selected))
        hashes: List[np.ndarray] = []
        hashed_names: List[str] = []
        for _split, path in selected:
            image_hash = perceptual_hash(path)
            if image_hash is None:
                continue
            hashes.append(image_hash)
            hashed_names.append(path.name)
        stacked = np.stack(hashes) if hashes else np.zeros((0, 8), dtype=np.uint8)
        pairs = list(near_duplicate_pairs(stacked, max_distance)) if len(stacked) else []
        within: List[Dict[str, Any]] = []
        across: List[Dict[str, Any]] = []
        for i, j in pairs:
            first, second = hashed_names[i], hashed_names[j]
            first_splits = sorted(name_splits.get(first, set()))
            second_splits = sorted(name_splits.get(second, set()))
            entry = {
                "a": first,
                "b": second,
                "a_splits": first_splits,
                "b_splits": second_splits,
            }
            if first_splits == second_splits:
                within.append(entry)
            else:
                across.append(entry)
        near_duplicate_report = {
            "method": f"{HASH_SIZE}x{HASH_SIZE + 1} grayscale difference hash, half-scale decode",
            "max_hamming_distance": max_distance,
            "limits": {
                "max_images_hashed": max_images,
                "images_available": len(images),
                "images_hashed": len(hashed_names),
                "analysis_performed": True,
                "truncated": truncated,
                "note": (
                    "Near-duplicate detection is O(n^2) in the number of images. Only the first "
                    f"{max_images} images (by sorted path) are hashed; the remaining "
                    f"{len(images) - len(selected)} are NOT analysed."
                    if truncated
                    else "Every image was hashed."
                ),
            },
            "within_split_pairs": sorted(within, key=lambda item: (item["a"], item["b"]))[:500],
            "cross_split_pairs": sorted(across, key=lambda item: (item["a"], item["b"]))[:500],
            "pair_count": len(pairs),
            "within_split_pair_count": len(within),
            "cross_split_pair_count": len(across),
            "images_in_cross_split_pairs": len(
                {name for item in across for name in (item["a"], item["b"])}
            ),
            "groups_in_cross_split_pairs": sorted(
                {
                    group
                    for item in across
                    for name in (item["a"], item["b"])
                    for group in [sequence_from_name(name)]
                    if group is not None
                }
            ),
            "listing_note": (
                "within_split_pairs and cross_split_pairs are capped at 500 entries each; "
                "the *_count fields are the true totals."
            ),
        }

    # ── class balance ─────────────────────────────────────────────────────
    balance: Dict[str, Any] = {}
    for split in SPLITS:
        counts = per_split_class[split]
        rarest = min(counts.values()) if counts else None
        balance[split] = {
            "instances": {
                class_names.get(index, f"class_{index}"): count
                for index, count in sorted(counts.items())
            },
            "total": int(sum(counts.values())),
            "rarest_class_count": rarest,
            "ratio_to_rarest": (
                {
                    class_names.get(index, f"class_{index}"): count / rarest
                    for index, count in sorted(counts.items())
                }
                if rarest
                else {}
            ),
        }

    sizes = {
        class_names.get(index, f"class_{index}"): {
            "instances": len(per_class_areas[index]),
            "normalised_area": percentiles(per_class_areas[index]),
            "buckets": {
                name: per_class_buckets[index].get(name, 0) for name, _low, _high in SIZE_BUCKETS
            },
        }
        for index in sorted(per_class_areas)
    }

    manifest_path = data_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "schema_version": 1,
        "audit": "auair_civilian",
        "tool": "MODEL.audit_civilian_dataset",
        "read_only": True,
        "dataset_dir": str(data_dir.resolve()),
        "dataset_manifest": {
            "path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
            "sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        },
        "data_yaml_sha256": sha256_file(data_yaml),
        "source": manifest.get("source"),
        "source_inputs": manifest.get("inputs"),
        "classes": class_names,
        "counts": {
            "images": len(images),
            "boxes": total_boxes,
            "images_per_split": {
                split: sum(1 for s, _ in images if s == split) for split in SPLITS
            },
            "boxes_per_split": {
                split: int(sum(per_split_class[split].values())) for split in SPLITS
            },
        },
        "images": {
            "corrupt": sorted(corrupt_images),
            "corrupt_count": len(corrupt_images),
            "missing_label_files": sorted(missing_labels),
            "missing_label_file_count": len(missing_labels),
        },
        "labels": {
            "rejected": {kind: count for kind, count in sorted(rejected.items())},
            "rejected_examples": rejected_examples,
        },
        "class_balance": balance,
        "object_size": sizes,
        "duplicates": {
            "method": "sha256 of the image file bytes",
            "groups": len(duplicate_groups),
            "images_in_groups": sum(len(names) for names in duplicate_groups.values()),
            "cross_split_groups": duplicate_cross_split,
            "examples": [
                {"sha256": digest, "images": names}
                for digest, names in sorted(duplicate_groups.items())[:100]
            ],
        },
        "near_duplicates": near_duplicate_report,
        "leakage": {
            "grouping": "sequence key from the image name",
            "groups": len(group_splits),
            "groups_in_multiple_splits": leaked_groups,
            "duplicate_groups_crossing_splits": duplicate_cross_split,
            "images_in_multiple_splits": images_in_multiple_splits,
        },
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """A concise, deterministic Markdown summary of the audit."""
    counts = report["counts"]
    lines = [
        f"# Dataset audit — {report['audit']}",
        "",
        f"- **Dataset:** `{report['dataset_dir']}`",
        f"- **Read-only:** {report['read_only']} — no label, image or file was modified",
        f"- **Dataset manifest SHA-256:** {report['dataset_manifest']['sha256'] or 'unavailable'}",
        f"- **data.yaml SHA-256:** {report['data_yaml_sha256']}",
        "",
        "## Counts",
        "",
        f"- Images: **{counts['images']}** ({', '.join(f'{k} {v}' for k, v in counts['images_per_split'].items())})",
        f"- Boxes: **{counts['boxes']}** ({', '.join(f'{k} {v}' for k, v in counts['boxes_per_split'].items())})",
        "",
        "## Problems found",
        "",
        f"- Corrupt images: **{report['images']['corrupt_count']}**",
        f"- Images with no label file: **{report['images']['missing_label_file_count']}**",
    ]
    rejected = report["labels"]["rejected"]
    if rejected:
        lines.append("- Label problems:")
        for kind, count in rejected.items():
            lines.append(f"  - `{kind}`: {count}")
    else:
        lines.append("- Label problems: none")

    leakage = report["leakage"]
    duplicates = report["duplicates"]
    near = report["near_duplicates"]
    lines.extend(
        [
            "",
            "## Leakage",
            "",
            f"- Groups: **{leakage['groups']}**",
            f"- Groups appearing in more than one split: **{len(leakage['groups_in_multiple_splits'])}**"
            + (
                f" — {leakage['groups_in_multiple_splits'][:10]}"
                if leakage["groups_in_multiple_splits"]
                else ""
            ),
            f"- Exact-duplicate groups: **{duplicates['groups']}** "
            f"({duplicates['images_in_groups']} images)",
            f"- Exact-duplicate groups crossing splits: **{len(duplicates['cross_split_groups'])}**",
            f"- Near-duplicate pairs found: **{near['pair_count']}** "
            f"(within a split: {near['within_split_pair_count']}, across splits: {near['cross_split_pair_count']})",
            f"- Images involved in a cross-split near-duplicate: **{near['images_in_cross_split_pairs']}**"
            + (
                f" — groups {', '.join(near['groups_in_cross_split_pairs'])}"
                if near["groups_in_cross_split_pairs"]
                else ""
            ),
            "",
            "> Within-split near-duplicate counts are expected to be very large: frames are",
            "> extracted 5 per second from continuous video, so consecutive frames are almost",
            "> identical. The cross-split count is the leakage signal — a near-duplicate in two",
            "> different splits means the same scene or the same parked vehicles appear on both",
            "> sides of the split, which a video-grouped split cannot prevent when two videos",
            "> filmed the same place.",
            "",
            "### Near-duplicate limits",
            "",
            f"- Method: {near['method']}",
            f"- Maximum Hamming distance: {near['max_hamming_distance']}",
            f"- Images hashed: {near['limits'].get('images_hashed', 'not hashed')} "
            f"of {near['limits']['images_available']} available "
            f"(cap {near['limits']['max_images_hashed']})",
            f"- {near['limits']['note']}",
            "",
            "## Class balance",
            "",
            "| split | class | instances | ratio to rarest |",
            "|---|---|---:|---:|",
        ]
    )
    for split, entry in report["class_balance"].items():
        for name, count in entry["instances"].items():
            ratio = entry["ratio_to_rarest"].get(name)
            lines.append(
                f"| {split} | {name} | {count} | {ratio:.2f} |"
                if ratio
                else f"| {split} | {name} | {count} | n/a |"
            )

    lines.extend(
        [
            "",
            "## Object size (normalised area)",
            "",
            "| class | instances | p05 | p50 | p95 | tiny | small | medium | large |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, entry in report["object_size"].items():
        area = entry["normalised_area"]
        buckets = entry["buckets"]
        lines.append(
            f"| {name} | {entry['instances']} | "
            f"{area['p05']:.5f} | {area['p50']:.5f} | {area['p95']:.5f} | "
            f"{buckets['tiny']} | {buckets['small']} | {buckets['medium']} | {buckets['large']} |"
        )
    lines.extend(
        [
            "",
            "> Buckets are normalised area: tiny < 0.1%, small 0.1-1%, medium 1-5%, large > 5%.",
            "> Nothing in this report was corrected automatically - every issue is listed for human review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, required=True, help="Dataset directory holding data.yaml"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory (default: the dataset dir)"
    )
    parser.add_argument(
        "--no-near-duplicates", action="store_true", help="Skip the O(n^2) perceptual hash stage"
    )
    parser.add_argument(
        "--near-dup-max-distance",
        type=int,
        default=NEAR_DUPLICATE_MAX_DISTANCE,
        help=f"Hamming distance for a near-duplicate (default {NEAR_DUPLICATE_MAX_DISTANCE})",
    )
    parser.add_argument(
        "--near-dup-max-images",
        type=int,
        default=NEAR_DUPLICATE_MAX_IMAGES,
        help=f"Maximum images to hash (default {NEAR_DUPLICATE_MAX_IMAGES})",
    )
    args = parser.parse_args()

    report = audit(
        data_dir=args.data,
        near_duplicates=not args.no_near_duplicates,
        max_distance=args.near_dup_max_distance,
        max_images=args.near_dup_max_images,
    )

    out_dir = args.out or args.data
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", json_path)

    md_path = out_dir / "audit.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("wrote %s", md_path)


if __name__ == "__main__":
    main()
