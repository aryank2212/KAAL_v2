"""
build_civilian_auair.py — build the isolated AU-AIR civilian dataset.

This is a **separate experiment** from ``MODEL/build_dataset.py``. It does not
read, write or touch the production nine-class taxonomy
(``aircraft, helicopter, tank, armored_vehicle, artillery,
engineering_vehicle, soldier, vehicle, person``), the production detector, the
production weights or ``MODEL/datasets_registry.yaml``.

AU-AIR is a civilian low-altitude traffic-surveillance dataset captured with a
Parrot Bebop 2 over an intersection in Aarhus, Denmark. Its own eight traffic
categories are kept as a separate eight-class civilian taxonomy:

    human, car, truck, van, motorbike, bicycle, bus, trailer

Upstream schema (verified against the released annotation file, 2026-10-21):
a single JSON object with ``info``, ``licenses``, ``categories`` and
``annotations``. Each annotation carries ``image_name``, the frame size, the
UAV flight data and a ``bbox`` list whose entries are
``{top, left, height, width, class}`` in pixels with a top-left origin, where
``class`` indexes the ``categories`` array. Frame names follow
``frame_<YYYYMMDDHHMMSS>_<x+>_<7 digits>.jpg``; the timestamp prefix
identifies the source video and is cross-checked against the per-frame
``time`` field.

Splits are assigned **by source video**, never by frame. Frames are extracted
five per second from eight continuous videos, so a random frame split would
leak the test set into training. If a sequence key cannot be recovered, the
builder refuses to split rather than falling back to a random assignment.

    python -m MODEL.build_civilian_auair \\
        --annotations MODEL/Datasets/auair/extracted/annotations.json \\
        --images MODEL/Datasets/auair/images \\
        --out MODEL/Datasets/auair_civilian --seed 42

See ``Docs/CIVILIAN_AERIAL_DATA.md`` for provenance, licensing status and the
acquisition preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")

# The civilian eight-class taxonomy. Index == YOLO class id.
CIVILIAN_CLASSES: Tuple[str, ...] = (
    "human",
    "car",
    "truck",
    "van",
    "motorbike",
    "bicycle",
    "bus",
    "trailer",
)

# Upstream ``categories`` array, as released. Order is recorded in the manifest
# for traceability, but the mapping below is by **name**, so a different array
# order in a future release cannot silently permute the labels.
UPSTREAM_CATEGORIES: Tuple[str, ...] = (
    "Human",
    "Car",
    "Truck",
    "Van",
    "Motorbike",
    "Bicycle",
    "Bus",
    "Trailer",
)

# Upstream category name (lower-cased) -> civilian class id.
UPSTREAM_TO_CIVILIAN: Dict[str, int] = {
    name.lower(): index for index, name in enumerate(CIVILIAN_CLASSES)
}

# frame_<YYYYMMDDHHMMSS>_<x+>_<7 digits>.jpg
FRAME_NAME_RE = re.compile(r"^frame_(?P<sequence>\d{14})_(?P<marker>x+)_(\d{7})\.jpg$")

DEFAULT_FRACTIONS = (0.7, 0.15, 0.15)

# Upstream source, licence status and citation. The licence statements conflict
# upstream; see Docs/CIVILIAN_AERIAL_DATA.md section 4. Nothing here resolves
# the conflict - the manifest records it verbatim.
SOURCE = {
    "name": "AU-AIR",
    "citation": (
        'I. Bozcan and E. Kayacan, "AU-AIR: A Multi-modal Unmanned Aerial Vehicle '
        'Dataset for Low Altitude Traffic Surveillance", ICRA 2020. '
        "arXiv:2001.11737"
    ),
    "upstream_pages": [
        "https://bozcani.github.io/auairdataset",
        "https://web.archive.org/web/20220125151322/https://bozcani.github.io/auairdataset",
        "https://github.com/bozcani/auairdataset",
        "https://arxiv.org/abs/2001.11737",
    ],
    "download": {
        "frames": {
            "google_drive_id": "1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ",
            "url": "https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ",
            "published_size": "2.2 GB",
            "actual_bytes": 2318071652,
            "archive_name": "frames.zip",
        },
        "annotations": {
            "google_drive_id": "1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA",
            "url": "https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA",
            "published_size": "3.9 MB",
            "actual_bytes": 4039375,
            "archive_name": "annotations.json",
            "note": "published as a .json name but the file is a ZIP containing annotations.json",
        },
    },
    "license": {
        "status": "conflicting - unresolved",
        "statements": [
            {
                "source": "official project page (archived 2022-01-25)",
                "text": (
                    "You are free to use AU-AIR dataset and all empirical data and metadata "
                    "under the CC-BY licence, requesting appropriate credit if you do."
                ),
            },
            {
                "source": "FMI Data Index",
                "text": "Dataset licenced under the CC BY 4.0 licence.",
            },
            {
                "source": "the annotation file's own 'licenses' block",
                "text": (
                    "Attribution-NonCommercial-ShareAlike License (CC BY-NC-SA 2.0) and "
                    "Attribution-NonCommercial License (CC BY-NC 2.0)"
                ),
            },
            {
                "source": "auairdataset GitHub LICENSE",
                "text": "MIT - covers the API code, not the dataset.",
            },
        ],
        "operating_assumption": (
            "Non-commercial. Do not redistribute raw frames or annotations, and do not "
            "publish derivatives under terms that permit commercial use, until the authors "
            "clarify. Contact: ilker@eng.au.dk"
        ),
    },
}


class CivilianDatasetError(RuntimeError):
    """The AU-AIR input cannot be converted safely."""


@dataclass
class Frame:
    """One AU-AIR frame with its civilian boxes."""

    image_name: str
    sequence: str
    boxes: List[Tuple[int, float, float, float, float]] = field(default_factory=list)
    split: str = ""


@dataclass
class Rejected:
    """A box or frame that could not be used, and why."""

    kind: str
    detail: str
    count: int = 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_width(record: Dict[str, Any]) -> Optional[float]:
    """Frame width in pixels.

    The released file spells this key ``image_width:`` with a trailing colon;
    the upstream sample spells it ``image_width``. Both are accepted.
    """
    for key in ("image_width", "image_width:"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def frame_height(record: Dict[str, Any]) -> Optional[float]:
    value = record.get("image_height")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def sequence_from_name(image_name: str) -> Optional[str]:
    """Source-video key taken from the frame name, or None if it cannot be read."""
    match = FRAME_NAME_RE.match(image_name)
    return match.group("sequence") if match else None


def sequence_from_time(time_field: Any) -> Optional[str]:
    """Source-video key taken from the per-frame ``time`` record."""
    if not isinstance(time_field, dict):
        return None
    parts = []
    for key, width in (("year", 4), ("month", 2), ("day", 2), ("hour", 2), ("min", 2), ("sec", 2)):
        value = time_field.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        parts.append(f"{value:0{width}d}")
    return "".join(parts)


def civilian_class_ids(categories: Sequence[Any]) -> Dict[int, int]:
    """Map upstream class index -> civilian class id, by category **name**.

    Raises when a category name is not part of the documented eight-class
    civilian taxonomy: an unknown name means the mapping would be incomplete,
    so the build stops instead of guessing.
    """
    mapping: Dict[int, int] = {}
    for index, name in enumerate(categories):
        if not isinstance(name, str):
            raise CivilianDatasetError(f"Category {index} is not a name: {name!r}")
        civilian = UPSTREAM_TO_CIVILIAN.get(name.strip().lower())
        if civilian is None:
            raise CivilianDatasetError(
                f"Unknown upstream category {name!r} at index {index}. "
                f"Known categories: {sorted(UPSTREAM_TO_CIVILIAN)}"
            )
        mapping[index] = civilian
    return mapping


def parse_boxes(
    record: Dict[str, Any],
    class_ids: Dict[int, int],
    rejected: Counter,
    examples: List[Dict[str, Any]],
) -> List[Tuple[int, float, float, float, float]]:
    """Convert one frame's boxes to YOLO format, reporting unusable ones.

    Returns ``(class_id, cx, cy, w, h)`` with normalised coordinates. Boxes
    that cannot be trusted are dropped and counted - never remapped.
    """
    width = frame_width(record)
    height = frame_height(record)
    boxes: List[Tuple[int, float, float, float, float]] = []
    raw = record.get("bbox")
    if raw is None:
        return boxes
    if not isinstance(raw, list):
        rejected["malformed_bbox_list"] += 1
        return boxes

    for box in raw:
        if not isinstance(box, dict):
            rejected["malformed_box"] += 1
            continue
        label = box.get("class")
        if not isinstance(label, int) or isinstance(label, bool) or label not in class_ids:
            rejected["unknown_class_index"] += 1
            if len(examples) < 50:
                examples.append({"kind": "unknown_class_index", "class": repr(label)})
            continue
        if width is None or height is None or width <= 0 or height <= 0:
            rejected["unknown_frame_size"] += 1
            continue
        try:
            left = float(box["left"])
            top = float(box["top"])
            box_width = float(box["width"])
            box_height = float(box["height"])
        except (KeyError, TypeError, ValueError):
            rejected["malformed_box"] += 1
            if len(examples) < 50:
                examples.append({"kind": "malformed_box", "box": repr(box)})
            continue
        if box_width <= 0 or box_height <= 0:
            rejected["non_positive_size"] += 1
            if len(examples) < 50:
                examples.append(
                    {
                        "kind": "non_positive_size",
                        "box": repr(box),
                        "frame": record.get("image_name"),
                    }
                )
            continue
        if left < 0 or top < 0:
            rejected["negative_origin"] += 1
            if len(examples) < 50:
                examples.append(
                    {"kind": "negative_origin", "box": repr(box), "frame": record.get("image_name")}
                )
            continue
        if left + box_width > width or top + box_height > height:
            rejected["out_of_bounds"] += 1
            if len(examples) < 50:
                examples.append(
                    {"kind": "out_of_bounds", "box": repr(box), "frame": record.get("image_name")}
                )
            continue
        boxes.append(
            (
                class_ids[label],
                (left + box_width / 2.0) / width,
                (top + box_height / 2.0) / height,
                box_width / width,
                box_height / height,
            )
        )
    return boxes


def read_frames(
    annotations: Dict[str, Any],
    images_dir: Optional[Path],
    rejected: Counter,
    examples: List[Dict[str, Any]],
) -> List[Frame]:
    """Turn the annotation file into frames with a verified sequence key."""
    categories = annotations.get("categories")
    records = annotations.get("annotations")
    if not isinstance(categories, list) or not categories:
        raise CivilianDatasetError("Annotation file has no 'categories' array")
    if not isinstance(records, list) or not records:
        raise CivilianDatasetError("Annotation file has no 'annotations' array")

    class_ids = civilian_class_ids(categories)
    frames: List[Frame] = []
    for record in records:
        if not isinstance(record, dict):
            rejected["malformed_annotation"] += 1
            continue
        image_name = record.get("image_name")
        if not isinstance(image_name, str) or not image_name:
            rejected["missing_image_name"] += 1
            continue

        name_key = sequence_from_name(image_name)
        time_key = sequence_from_time(record.get("time"))
        if name_key is None:
            rejected["unreadable_frame_name"] += 1
            if len(examples) < 50:
                examples.append({"kind": "unreadable_frame_name", "frame": image_name})
            continue
        if time_key is None:
            rejected["unreadable_frame_time"] += 1
            if len(examples) < 50:
                examples.append({"kind": "unreadable_frame_time", "frame": image_name})
            continue
        if name_key != time_key:
            # The two independent sources of the sequence key disagree, so the
            # frame cannot be placed in a split with confidence.
            rejected["sequence_key_mismatch"] += 1
            if len(examples) < 50:
                examples.append(
                    {
                        "kind": "sequence_key_mismatch",
                        "frame": image_name,
                        "from_name": name_key,
                        "from_time": time_key,
                    }
                )
            continue

        boxes = parse_boxes(record, class_ids, rejected, examples)
        if images_dir is not None and not (images_dir / image_name).is_file():
            rejected["missing_image_file"] += 1
            if len(examples) < 50:
                examples.append({"kind": "missing_image_file", "frame": image_name})
            continue
        frames.append(Frame(image_name=image_name, sequence=name_key, boxes=boxes))
    return frames


def assign_splits(
    frames: Sequence[Frame],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Dict[str, str]:
    """Assign whole source videos to splits.

    Every frame of a video lands in the same split, so adjacent frames can
    never straddle a split boundary. The seed only breaks ties between
    otherwise equally good candidates, which makes the result deterministic.
    """
    sequences: Dict[str, Counter] = defaultdict(Counter)
    for frame in frames:
        sequences[frame.sequence].update(class_id for class_id, *_ in frame.boxes)

    if len(sequences) < 2:
        raise CivilianDatasetError(
            f"Need at least 2 source sequences for a grouped split, found {len(sequences)}. "
            "Refusing to fall back to a random frame split."
        )

    total = len(sequences)
    n_val = max(1, round(total * fractions[1]))
    n_test = max(1, round(total * fractions[2]))
    if n_val + n_test >= total:
        raise CivilianDatasetError(
            f"Cannot split {total} sequences into train/val/test with fractions {fractions}"
        )

    class_frequency: Counter = Counter()
    for counts in sequences.values():
        class_frequency.update(counts)

    # "Rare" classes are the ones with the fewest instances overall; a split
    # that contains them is worth more than one that contains only cars.
    ordered_frequency = sorted(class_frequency.values())
    median = ordered_frequency[len(ordered_frequency) // 2]
    rare_classes = {class_id for class_id, count in class_frequency.items() if count < median}

    rng = random.Random(seed)
    candidates = sorted(sequences)
    rng.shuffle(candidates)
    seeded_order = {key: index for index, key in enumerate(candidates)}

    def score(key: str, covered: Iterable[int]) -> Tuple[int, int, int]:
        """Prefer sequences that add classes, then the ones with rare classes."""
        classes = set(sequences[key])
        new_classes = classes - set(covered)
        rare_support = sum(sequences[key][class_id] for class_id in classes & rare_classes)
        return (-len(new_classes), -rare_support, seeded_order[key])

    chosen: Dict[str, str] = {}
    covered: set = set()
    for split, quota in (("test", n_test), ("val", n_val)):
        for _ in range(quota):
            available = [key for key in candidates if key not in chosen]
            if not available:
                break
            best = min(available, key=lambda key: score(key, covered))
            chosen[best] = split
            covered |= set(sequences[best])
    for key in candidates:
        chosen.setdefault(key, "train")
    return chosen


def write_dataset(frames: Sequence[Frame], images_dir: Path, out: Path) -> Dict[str, Any]:
    """Copy images and write YOLO label files, then the ``data.yaml``."""
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    per_split_frames: Counter = Counter()
    per_split_classes: Dict[str, Counter] = {split: Counter() for split in SPLITS}
    for frame in sorted(frames, key=lambda f: (f.split, f.image_name)):
        source = images_dir / frame.image_name
        target = out / frame.split / "images" / frame.image_name
        shutil.copyfile(source, target)
        label = out / frame.split / "labels" / f"{Path(frame.image_name).stem}.txt"
        label.write_text(
            "".join(
                f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                for class_id, cx, cy, w, h in frame.boxes
            ),
            encoding="utf-8",
        )
        per_split_frames[frame.split] += 1
        per_split_classes[frame.split].update(class_id for class_id, *_ in frame.boxes)

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": len(CIVILIAN_CLASSES),
                "names": dict(enumerate(CIVILIAN_CLASSES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "frames": dict(sorted(per_split_frames.items())),
        "instances": {
            split: {CIVILIAN_CLASSES[c]: n for c, n in sorted(counts.items())}
            for split, counts in per_split_classes.items()
        },
    }


def build_manifest(
    frames: Sequence[Frame],
    annotations_path: Path,
    images_dir: Path,
    out: Path,
    seed: int,
    fractions: Tuple[float, float, float],
    split_assignment: Dict[str, str],
    rejected: Counter,
    examples: List[Dict[str, Any]],
    inputs: List[Dict[str, Any]],
    categories: Sequence[Any],
) -> Dict[str, Any]:
    """Assemble the dataset manifest: provenance, counts, rejects and hashes."""
    per_split_frames: Counter = Counter()
    per_split_classes: Dict[str, Counter] = {split: Counter() for split in SPLITS}
    per_class: Counter = Counter()
    for frame in frames:
        per_split_frames[frame.split] += 1
        per_split_classes[frame.split].update(class_id for class_id, *_ in frame.boxes)
        per_class.update(class_id for class_id, *_ in frame.boxes)

    sequences: Dict[str, Counter] = defaultdict(Counter)
    sequence_frames: Counter = Counter()
    for frame in frames:
        sequence_frames[frame.sequence] += 1
        sequences[frame.sequence].update(class_id for class_id, *_ in frame.boxes)

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "auair_civilian",
        "taxonomy": "civilian-eight",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "MODEL.build_civilian_auair",
        "seed": seed,
        "source": SOURCE,
        "classes": list(CIVILIAN_CLASSES),
        "upstream_categories": list(categories),
        "class_mapping": {
            CIVILIAN_CLASSES[class_id]: name
            for class_id, name in sorted(
                (civilian_id, categories[index])
                for index, civilian_id in civilian_class_ids(categories).items()
            )
        },
        "split_policy": {
            "grouping": "source_video",
            "sequence_key": "YYYYMMDDHHMMSS prefix of the frame name, cross-checked against the per-frame time field",
            "random_frame_split": "not implemented - a frame-level split would leak adjacent frames",
            "fractions": {"train": fractions[0], "val": fractions[1], "test": fractions[2]},
            "assignment": "greedy maximum class coverage, seed breaks ties",
        },
        "counts": {
            "frames_total": len(frames),
            "frames_per_split": {split: per_split_frames.get(split, 0) for split in SPLITS},
            "instances_total": int(sum(per_class.values())),
            "instances_per_split": {
                split: {CIVILIAN_CLASSES[c]: n for c, n in sorted(per_split_classes[split].items())}
                for split in SPLITS
            },
            "instances_per_class": {
                CIVILIAN_CLASSES[c]: per_class.get(c, 0) for c in range(len(CIVILIAN_CLASSES))
            },
        },
        "sequences": [
            {
                "key": key,
                "split": split_assignment.get(key, "train"),
                "frames": sequence_frames.get(key, 0),
                "instances": {
                    CIVILIAN_CLASSES[c]: n for c, n in sorted(sequences.get(key, Counter()).items())
                },
            }
            for key in sorted(sequence_frames)
        ],
        "rejected": {kind: count for kind, count in sorted(rejected.items())},
        "rejected_examples": examples,
        "inputs": inputs,
        "outputs": {
            "dataset_dir": str(out.resolve()),
            "annotations": str(annotations_path.resolve()),
            "images": str(images_dir.resolve()),
        },
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--annotations",
        type=Path,
        default=here / "Datasets" / "auair" / "extracted" / "annotations.json",
        help="Path to the AU-AIR annotations.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=here / "Datasets" / "auair" / "images",
        help="Directory holding the AU-AIR frames",
    )
    parser.add_argument("--out", type=Path, default=here / "Datasets" / "auair_civilian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fractions",
        default=",".join(str(value) for value in DEFAULT_FRACTIONS),
        help="train,val,test fractions of the source sequences (default 0.7,0.15,0.15)",
    )
    parser.add_argument(
        "--frames-archive",
        type=Path,
        default=here / "Datasets" / "auair" / "frames.zip",
        help="Upstream frames archive, hashed into the manifest",
    )
    parser.add_argument(
        "--annotations-archive",
        type=Path,
        default=here / "Datasets" / "auair" / "annotations.json",
        help="Upstream annotations archive, hashed into the manifest",
    )
    args = parser.parse_args()

    if not args.annotations.is_file():
        raise SystemExit(f"Annotations not found: {args.annotations}")
    if not args.images.is_dir():
        raise SystemExit(f"Images directory not found: {args.images}")

    fractions = tuple(float(part) for part in args.fractions.split(","))
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise SystemExit("--fractions needs three positive numbers")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise SystemExit(f"--fractions must sum to 1.0, got {sum(fractions)}")

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    rejected: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    frames = read_frames(annotations, args.images, rejected, examples)
    if not frames:
        raise CivilianDatasetError(
            f"No usable frames - every annotation was rejected. Rejections: {dict(rejected)}"
        )
    logger.info("usable frames: %d (rejected: %s)", len(frames), dict(rejected))

    assignment = assign_splits(frames, fractions, args.seed)
    for frame in frames:
        frame.split = assignment[frame.sequence]
    logger.info(
        "sequence -> split: %s",
        {key: assignment[key] for key in sorted(assignment)},
    )

    inputs = []
    for label, path in (
        ("annotations.json", args.annotations),
        ("frames.zip", args.frames_archive),
        ("annotations archive", args.annotations_archive),
    ):
        if path.is_file():
            inputs.append(
                {
                    "label": label,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        else:
            inputs.append(
                {"label": label, "path": str(path.resolve()), "sha256": None, "bytes": None}
            )

    stats = write_dataset(frames, args.images, args.out)
    logger.info("wrote %s: %s", args.out, stats["frames"])

    manifest = build_manifest(
        frames=frames,
        annotations_path=args.annotations,
        images_dir=args.images,
        out=args.out,
        seed=args.seed,
        fractions=fractions,
        split_assignment=assignment,
        rejected=rejected,
        examples=examples,
        inputs=inputs,
        categories=annotations.get("categories", []),
    )
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s", manifest_path)


if __name__ == "__main__":
    main()
