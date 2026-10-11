"""
build_dataset.py — Build the cleaned, multi-source KAAL detection dataset.

Merges four sources into one YOLO-format dataset with a coarse 9-class
taxonomy that both Ultralytics YOLO and RF-DETR train on directly:

    0 aircraft            fixed-wing aircraft, tilt-rotors, large UAVs
    1 helicopter
    2 tank
    3 armored_vehicle     APCs / IFVs
    4 artillery
    5 engineering_vehicle
    6 soldier
    7 vehicle             civilian / unspecified road vehicles
    8 person              civilians and people with no military marking

Sources
  legacy     MODEL/runs/adversarial/augmented_dataset (the original 52-class
             KAAL set). Images that contain legacy class 0 ("A10") are
             DROPPED: that class is a catch-all that mixes generic aircraft
             and cars (see MODEL/RESULTS.md). ``_noisy`` variants are dropped
             (the originals already carry Roboflow noise), augmented variants
             are kept in train only (``_patch`` for a fraction of photos), and
             val/test keep one image per photo.
             Caveat: the set's own val/test folders hold only aircraft and
             cars — every tank, APC/IFV, artillery, engineering vehicle and
             soldier is in train, and those ``mil_image_*`` files are mostly
             near-identical video frames under unrelated names.
             ``--legacy-split resplit`` re-splits by *content* group
             (flip-aware 256-bit dHash, ``--dup-dist``) stratified on the
             rarest class, but true duplicates reach ~24 bits, and at that
             distance soldier/engineering collapse to a handful of scenes —
             too few for a meaningful held-out test (MODEL/RESULTS.md §6).
  visdrone   VisDrone2019-DET (drone viewpoint). Pedestrians/people map to
             ``person`` — never ``soldier``: civilians must not be conflated
             with combatants. Ignored regions (category 0 / score 0) and
             "others" are dropped.
  aircraft   a2015003713 military-aircraft-detection-dataset (COCO json).
             Rotorcraft types map to ``helicopter``, everything else to
             ``aircraft``. Only images present on disk are used, so a partial
             download (``python -m MODEL.fetch_datasets aircraft``) works.
  uav_tank   Simuletic synthetic drone-view tanks -> ``tank``.

  dvids      llama-farm/military-labeled-yolo — DVIDS U.S. DoD public-domain
             photos (CC0): soldier, tank, apc, artillery, mlrs, missile
             launcher, trucks, aircraft. Split 80/10/10 by upload block.
  openimages_tank  Open Images V7 photos with a Tank box; images with
             group-of boxes or people are dropped (see adapter docstring).

Versions
  kaal_v3    legacy + visdrone(3000) + aircraft + uav_tank, per-source leakage
             check only (git 6581dc5; reproduce with ``--caps`` = V3_CAPS,
             ``--cross-dup-dist -1 --keep-unlabeled-legacy``).
  kaal_v4    (default) + all VisDrone train images + dvids + openimages_tank;
             near-duplicates grouped across ALL sources before splitting, so
             no photo appears in two splits even via another source; legacy
             images without labels dropped.

Usage:
    python -m MODEL.build_dataset                       # -> MODEL/Datasets/kaal_v4
"""

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

TAXONOMY: Tuple[str, ...] = (
    "aircraft",
    "helicopter",
    "tank",
    "armored_vehicle",
    "artillery",
    "engineering_vehicle",
    "soldier",
    "vehicle",
    "person",
)
CLS = {name: i for i, name in enumerate(TAXONOMY)}
SPLITS = ("train", "val", "test")

# Legacy 52-class ids -> taxonomy. ``DROP_IMAGE`` marks an unreliable label:
# any image carrying it is excluded entirely (dropping only the box would turn
# real objects into "background" for the detector).
DROP_IMAGE = -1
LEGACY_MAP: Dict[int, int] = {0: DROP_IMAGE}
LEGACY_MAP.update({i: CLS["aircraft"] for i in range(1, 40)})
LEGACY_MAP.update(
    {
        40: CLS["armored_vehicle"],  # APC
        41: CLS["tank"],
        42: CLS["artillery"],
        43: CLS["aircraft"],  # AIRPLANE
        44: CLS["armored_vehicle"],  # IFV
        45: CLS["helicopter"],
        46: CLS["engineering_vehicle"],
        47: CLS["soldier"],
        48: CLS["vehicle"],  # Aerial Car 0-3
        49: CLS["vehicle"],
        50: CLS["vehicle"],
        51: CLS["vehicle"],
    }
)

# VisDrone2019-DET category ids (0 = ignored region, 11 = others).
VISDRONE_MAP: Dict[int, int] = {
    1: CLS["person"],  # pedestrian
    2: CLS["person"],  # people
    **{c: CLS["vehicle"] for c in range(3, 11)},  # bicycle..motor
}

ROTORCRAFT = {
    "AH64", "CH47", "CH53", "Ka27", "Ka52", "Mi24", "Mi26", "Mi28", "Mi8",
    "NH90", "UH60", "WZ10", "WZ9", "Z10", "Z19", "Z21",
}  # fmt: skip

# llama-farm/military-labeled-yolo (DVIDS) class names -> taxonomy. ``None``
# drops the box: KAAL has no maritime class, and a warship is not a "vehicle".
DVIDS_MAP: Dict[str, Optional[int]] = {
    "soldier": CLS["soldier"],
    "tank": CLS["tank"],
    "apc": CLS["armored_vehicle"],
    "artillery": CLS["artillery"],
    "mlrs": CLS["artillery"],  # multiple rocket launcher
    "missile_launcher": CLS["artillery"],
    "military_truck": CLS["vehicle"],
    "helicopter": CLS["helicopter"],
    "aircraft": CLS["aircraft"],
    "warship": None,
    "car": CLS["vehicle"],
    "truck": CLS["vehicle"],
}

# Open Images leaf classes kept by ``fetch_datasets openimages_tank``.
OPENIMAGES_MAP: Dict[str, int] = {
    "Tank": CLS["tank"],
    "Helicopter": CLS["helicopter"],
    "Airplane": CLS["aircraft"],
    **{c: CLS["vehicle"] for c in ("Car", "Truck", "Van", "Bus", "Motorcycle", "Bicycle")},
}

LICENSES = {
    "legacy": "KAAL project data (Roboflow exports; see original sources)",
    "visdrone": "VisDrone2019-DET — academic, non-commercial use",
    "aircraft": "a2015003713/military-aircraft-detection-dataset (Kaggle mirror; no license declared)",
    "uav_tank": "Simuletic UAV tank sample — CC BY-NC 4.0 per HF metadata (README says CC BY 4.0)",
    "dvids": "llama-farm/military-labeled-yolo — DVIDS U.S. DoD public-domain imagery, labels CC0-1.0",
    "openimages_tank": "Google Open Images V7 — annotations CC BY 4.0, images CC BY 2.0 (per image)",
}

Box = Tuple[int, float, float, float, float]  # cls, cx, cy, w, h (normalized)


@dataclass
class Sample:
    source: str
    split: str
    image: Path
    group: str  # samples sharing a group never land in different splits
    # Returns normalized boxes given the image (h, w); deferred because
    # pixel-coordinate sources need the real image size.
    boxes: Callable[[int, int], List[Box]]
    classes: List[int] = field(default_factory=list)  # for rarity sampling
    augmented: bool = False  # synthetic variant of another image (train only)


# ── label helpers ────────────────────────────────────────────────────────


def _clip_box(
    cx: float, cy: float, w: float, h: float
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def parse_yolo_file(path: Path, class_map: Dict[int, int]) -> Optional[List[Box]]:
    """Parse + remap a YOLO label file. ``None`` means "drop this image"."""
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        dst = class_map.get(src)
        if dst is None:
            continue
        if dst == DROP_IMAGE:
            return None
        clipped = _clip_box(cx, cy, w, h)
        if clipped:
            boxes.append((dst, *clipped))
    return boxes


def pixel_boxes_to_yolo(
    px: Sequence[Tuple[int, float, float, float, float]], h: int, w: int
) -> List[Box]:
    """(cls, x, y, bw, bh) in pixels (top-left origin) -> normalized YOLO boxes."""
    out: List[Box] = []
    for cls, x, y, bw, bh in px:
        clipped = _clip_box((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)
        if clipped:
            out.append((cls, *clipped))
    return out


def parse_visdrone_annotation(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """VisDrone ``x,y,w,h,score,category,trunc,occl`` -> pixel boxes in taxonomy ids."""
    out = []
    for line in path.read_text().splitlines():
        parts = [p for p in line.strip().split(",") if p != ""]
        if len(parts) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
        if score == 0 or cat not in VISDRONE_MAP or w <= 0 or h <= 0:
            continue
        out.append((VISDRONE_MAP[cat], float(x), float(y), float(w), float(h)))
    return out


def _stable_split(key: str, fractions=(0.7, 0.15, 0.15)) -> str:
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < fractions[0]:
        return "train"
    return "val" if u < fractions[0] + fractions[1] else "test"


# ── source adapters ──────────────────────────────────────────────────────


_AUG_SUFFIX = re.compile(r"_(noisy|occlude|patch)$")
_ROBOFLOW_HASH = re.compile(r"(_jpg|_jpeg|_png)?\.rf\.[0-9a-f]+$")


def legacy_source(stem: str) -> str:
    """Source-photo key: strip KAAL augmentation suffixes and Roboflow export hashes."""
    return _ROBOFLOW_HASH.sub("", _AUG_SUFFIX.sub("", stem))


def dhash(path: Path, size: int = 16) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """256-bit difference hash of an image and of its horizontal mirror."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    g = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    m = g[:, ::-1]
    return np.packbits(g[:, 1:] > g[:, :-1]), np.packbits(m[:, 1:] > m[:, :-1])


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(
    hashes: np.ndarray, mirrored: np.ndarray, max_dist: int, block: int = 256
) -> Iterable[Tuple[int, int]]:
    """Index pairs (i < j) whose hashes differ in at most ``max_dist`` bits."""
    for start in range(0, len(hashes), block):
        q = hashes[start : start + block, None, :]
        dist = np.minimum(
            _POPCOUNT[q ^ hashes[None]].sum(2, dtype=np.uint16),
            _POPCOUNT[q ^ mirrored[None]].sum(2, dtype=np.uint16),
        )
        for i, j in zip(*np.nonzero(dist <= max_dist)):
            if start + i < j:
                yield int(start + i), int(j)


def merge_near_duplicate_groups(
    samples: List[Sample], max_dist: int, workers: int = 8
) -> List[Sample]:
    """Union the groups of near-identical images (flip-aware dHash).

    The legacy ``mil_image_*`` files are Roboflow exports in which copies of
    one photo carry unrelated sequential numbers, so names alone cannot group
    them.
    """
    reps = [s for s in samples if not s.augmented]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashed = [(s, h) for s, h in zip(reps, pool.map(lambda s: dhash(s.image), reps)) if h]
    parent = {s.group: s.group for s in samples}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    if hashed:
        plain = np.stack([h[0] for _, h in hashed])
        mirrored = np.stack([h[1] for _, h in hashed])
        for i, j in near_duplicate_pairs(plain, mirrored, max_dist):
            a, b = find(hashed[i][0].group), find(hashed[j][0].group)
            if a != b:
                parent[max(a, b)] = min(a, b)
    merged = [replace(s, group=find(s.group)) for s in samples]
    logger.info(
        "near-duplicate grouping: %d name groups -> %d content groups",
        len({s.group for s in samples}),
        len({s.group for s in merged}),
    )
    return merged


def stratified_group_split(
    samples: List[Sample], fractions: Tuple[float, float, float], seed: int
) -> List[Sample]:
    """Reassign splits by group, stratified on each group's rarest class.

    Every class with at least a handful of groups lands in val and test, and
    all samples of a group share one split.
    """
    freq = Counter(c for s in samples for c in s.classes)
    members: Dict[str, set] = defaultdict(set)
    for s in samples:
        members[s.group].update(s.classes)
    strata: Dict[int, List[str]] = defaultdict(list)
    for group, classes in members.items():
        strata[min(classes, key=lambda c: (freq[c], c)) if classes else -1].append(group)

    rng = random.Random(seed)
    home: Dict[str, str] = {}
    for key in sorted(strata):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        n_test = round(len(groups) * fractions[2])
        n_val = round(len(groups) * fractions[1])
        for i, group in enumerate(groups):
            home[group] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return [replace(s, split=home[s.group]) for s in samples]


def legacy_samples(
    root: Path,
    patch_frac: float,
    seed: int,
    resplit: Optional[Tuple[float, float, float]] = None,
    dup_dist: int = 24,
    workers: int = 8,
    keep_unlabeled: bool = False,
) -> List[Sample]:
    """Legacy KAAL set. Images without any box are dropped unless
    ``keep_unlabeled``: in this set they are unannotated strike-footage frames
    (``mil_*``), not verified empty scenes, so they must not train as negatives."""
    samples: List[Sample] = []
    dropped = unlabeled = 0
    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            stem = img.stem
            if stem.endswith("_noisy"):
                continue
            boxes = parse_yolo_file(lbl_dir / f"{stem}.txt", LEGACY_MAP)
            if boxes is None:
                dropped += 1
                continue
            if not boxes and not keep_unlabeled:
                unlabeled += 1
                continue
            samples.append(
                Sample(
                    "legacy",
                    split,
                    img,
                    f"legacy:{legacy_source(stem)}",
                    (lambda b: lambda h, w: b)(boxes),
                    [b[0] for b in boxes],
                    augmented=bool(_AUG_SUFFIX.search(stem)),
                )
            )
    if resplit is not None:
        if dup_dist >= 0:
            samples = merge_near_duplicate_groups(samples, dup_dist, workers)
        samples = stratified_group_split(samples, resplit, seed)

    rng = random.Random(seed)
    kept: List[Sample] = []
    seen_eval_groups: set = set()
    for s in samples:
        if s.split == "train":
            if s.image.stem.endswith("_patch") and rng.random() >= patch_frac:
                continue
            kept.append(s)
        elif not s.augmented and s.group not in seen_eval_groups:
            seen_eval_groups.add(s.group)  # one image per photo in val/test
            kept.append(s)
    logger.info(
        "legacy: %d samples (%s split), %d images dropped (unreliable class 0), %d unlabeled",
        len(kept),
        "stratified re-" if resplit else "original",
        dropped,
        unlabeled,
    )
    return kept


def visdrone_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for split, sub in (("train", "train"), ("val", "val"), ("test", "test-dev")):
        base = root / f"VisDrone2019-DET-{sub}"
        if not base.is_dir():
            continue
        for ann in sorted((base / "annotations").glob("*.txt")):
            img = base / "images" / f"{ann.stem}.jpg"
            if not img.exists():
                continue
            px = parse_visdrone_annotation(ann)
            samples.append(
                Sample(
                    "visdrone",
                    split,
                    img,
                    f"visdrone:{ann.stem}",
                    (lambda p: lambda h, w: pixel_boxes_to_yolo(p, h, w))(px),
                    [b[0] for b in px],
                )
            )
    logger.info("visdrone: %d samples", len(samples))
    return samples


def aircraft_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for split in SPLITS:
        ann_path = root / "annotations" / "coco" / f"{split}.json"
        if not ann_path.exists():
            continue
        coco = json.loads(ann_path.read_text())
        cat = {
            c["id"]: CLS["helicopter"] if c["name"] in ROTORCRAFT else CLS["aircraft"]
            for c in coco["categories"]
        }
        per_image: Dict[int, list] = defaultdict(list)
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            per_image[a["image_id"]].append((cat[a["category_id"]], x, y, w, h))
        for im in coco["images"]:
            path = root / "images" / im["file_name"][0] / im["file_name"]
            if not path.exists():
                continue
            px = per_image.get(im["id"], [])
            samples.append(
                Sample(
                    "aircraft",
                    split,
                    path,
                    f"aircraft:{path.stem}",
                    (lambda p: lambda h, w: pixel_boxes_to_yolo(p, h, w))(px),
                    [b[0] for b in px],
                )
            )
    logger.info("aircraft: %d samples (images present on disk)", len(samples))
    return samples


def uav_tank_samples(root: Path) -> List[Sample]:
    yamls = list(root.rglob("data.yaml"))
    if not yamls:
        return []
    base = yamls[0].parent
    samples: List[Sample] = []
    for img in sorted((base / "images").iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        boxes = parse_yolo_file(base / "labels" / f"{img.stem}.txt", {0: CLS["tank"]}) or []
        samples.append(
            Sample(
                "uav_tank",
                _stable_split(img.stem),
                img,
                f"uav_tank:{img.stem}",
                (lambda b: lambda h, w: b)(boxes),
                [b[0] for b in boxes],
            )
        )
    logger.info("uav_tank: %d samples", len(samples))
    return samples


# ── assembly ─────────────────────────────────────────────────────────────


def dvids_samples(root: Path) -> List[Sample]:
    """DVIDS military photos. The set ships one ``train`` folder, so it is split
    80/10/10 here. DVIDS ids are sequential per upload, and one upload is
    usually one event, so photos are grouped in blocks of 20 ids to keep an
    event inside one split."""
    data = yaml.safe_load((root / "data.yaml").read_text())
    names = data["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    class_map = {i: c for i, n in enumerate(names) if (c := DVIDS_MAP.get(n)) is not None}
    samples: List[Sample] = []
    for folder in ("train", "val", "valid", "test"):
        img_dir = root / folder / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            label = root / folder / "labels" / f"{img.stem}.txt"
            if not label.exists():
                continue  # unlabeled
            boxes = parse_yolo_file(label, class_map) or []  # unmapped (warship) boxes dropped
            digits = re.sub(r"\D", "", img.stem)
            block = f"dvids:{int(digits) // 20}" if digits else f"dvids:{img.stem}"
            samples.append(
                Sample(
                    "dvids",
                    _stable_split(block, (0.8, 0.1, 0.1)),
                    img,
                    block,
                    (lambda b: lambda h, w: b)(boxes),
                    [b[0] for b in boxes],
                )
            )
    logger.info("dvids: %d samples", len(samples))
    return samples


def openimages_tank_samples(root: Path, drop_person_images: bool = True) -> List[Sample]:
    """Open Images images that contain a Tank box (see ``fetch_datasets``).

    Images with a group-of box are dropped (one box over many objects).
    With ``drop_person_images`` images that contain a Person box are dropped
    too: people next to tanks are mostly crews and soldiers, and labelling
    them ``person`` would contradict the ``soldier`` labels of other sources,
    while leaving them unlabelled would teach the detector to ignore people.
    """
    samples: List[Sample] = []
    skipped: Counter = Counter()
    for split_name, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        rows_path = root / f"{split_name}_boxes.json"
        if not rows_path.exists():
            continue
        rows: Dict[str, list] = json.loads(rows_path.read_text())
        for image_id, boxes in sorted(rows.items()):
            img = root / split_name / f"{image_id}.jpg"
            if not img.exists():
                skipped["missing"] += 1
                continue
            if any(b[5] for b in boxes):
                skipped["group_of"] += 1
                continue
            if drop_person_images and any(b[0] == "Person" for b in boxes):
                skipped["person"] += 1
                continue
            class_map = (
                OPENIMAGES_MAP
                if drop_person_images
                else {**OPENIMAGES_MAP, "Person": CLS["person"]}
            )
            out: List[Box] = []
            for label, x1, x2, y1, y2, _ in boxes:
                if label in class_map:
                    clipped = _clip_box((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                    if clipped:
                        out.append((class_map[label], *clipped))
            samples.append(
                Sample(
                    "openimages_tank",
                    split,
                    img,
                    f"openimages:{image_id}",
                    (lambda b: lambda h, w: b)(out),
                    [b[0] for b in out],
                )
            )
    logger.info("openimages_tank: %d samples, skipped %s", len(samples), dict(skipped))
    return samples


def resolve_leakage(samples: List[Sample]) -> Tuple[List[Sample], int]:
    """Keep each group in a single split (priority test > val > train)."""
    rank = {"test": 0, "val": 1, "train": 2}
    home: Dict[str, str] = {}
    for s in samples:
        if s.group not in home or rank[s.split] < rank[home[s.group]]:
            home[s.group] = s.split
    kept = [s for s in samples if home[s.group] == s.split]
    return kept, len(samples) - len(kept)


def keep_one_per_group_in_eval(samples: List[Sample]) -> Tuple[List[Sample], int]:
    """Keep the first non-augmented image of each group in val/test (train is untouched).

    Leakage resolution already keeps a group inside one split; this also stops
    one photo from being scored twice inside a split via two sources.
    """
    seen: set = set()
    kept: List[Sample] = []
    for s in samples:
        if s.split == "train":
            kept.append(s)
        elif not s.augmented and (s.split, s.group) not in seen:
            seen.add((s.split, s.group))
            kept.append(s)
    return kept, len(samples) - len(kept)


def rarity_sample(samples: List[Sample], cap: int, seed: int) -> List[Sample]:
    """Weighted sampling without replacement favouring images with rare classes."""
    if cap < 0 or cap >= len(samples):
        return list(samples)
    freq = Counter(c for s in samples for c in s.classes)
    rng = random.Random(seed)

    def weight(s: Sample) -> float:
        if not s.classes:
            return 0.2 / max(1, max(freq.values(), default=1))  # background images
        return max(1.0 / freq[c] for c in s.classes)

    keyed = sorted(
        ((rng.random() ** (1.0 / weight(s)), i) for i, s in enumerate(samples)), reverse=True
    )
    return [samples[i] for _, i in keyed[:cap]]


def _write_one(sample: Sample, out: Path, max_side: int, min_box_px: float) -> Optional[Counter]:
    img = cv2.imread(str(sample.image))
    if img is None:
        return None
    h, w = img.shape[:2]
    boxes = sample.boxes(h, w)
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    nh, nw = img.shape[:2]
    boxes = [b for b in boxes if b[3] * nw >= min_box_px and b[4] * nh >= min_box_px]

    name = f"{sample.source}_{sample.image.stem}"
    cv2.imwrite(
        str(out / sample.split / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    (out / sample.split / "labels" / f"{name}.txt").write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in boxes)
    )
    return Counter(TAXONOMY[b[0]] for b in boxes)


def write_dataset(
    selected: Iterable[Sample], out: Path, max_side: int, min_box_px: float, workers: int
) -> Dict:
    selected = list(selected)
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    images: Counter = Counter()
    instances: Dict[str, Counter] = {s: Counter() for s in SPLITS}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sample, counts in zip(
            selected, pool.map(lambda s: _write_one(s, out, max_side, min_box_px), selected)
        ):
            if counts is None:
                logger.warning("Unreadable image skipped: %s", sample.image)
                continue
            images[(sample.split, sample.source)] += 1
            instances[sample.split].update(counts)

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": len(TAXONOMY),
                "names": dict(enumerate(TAXONOMY)),
            },
            sort_keys=False,
        )
    )
    return {
        "images": {f"{sp}/{src}": n for (sp, src), n in sorted(images.items())},
        "instances": {sp: dict(c.most_common()) for sp, c in instances.items()},
    }


V3_CAPS = "legacy=5000:400:400,visdrone=3000:400:400,aircraft=-1:-1:-1,uav_tank=-1:-1:-1"
V4_CAPS = (
    "legacy=5000:400:400,visdrone=-1:400:400,aircraft=-1:-1:-1,uav_tank=-1:-1:-1,"
    "dvids=-1:-1:-1,openimages_tank=-1:-1:-1"
)


def parse_caps(spec: str) -> Dict[str, Tuple[int, int, int]]:
    """``"legacy=5000:400:400,visdrone=-1:-1:-1"`` -> {source: (train, val, test)}."""
    caps = {}
    for item in filter(None, spec.split(",")):
        name, values = item.split("=")
        t, v, s = (int(x) for x in values.split(":"))
        caps[name.strip()] = (t, v, s)
    return caps


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the cleaned KAAL detection dataset.")
    parser.add_argument("--out", default=str(here / "Datasets" / "kaal_v4"))
    parser.add_argument(
        "--legacy", default=str(here / "runs" / "adversarial" / "augmented_dataset")
    )
    parser.add_argument("--visdrone", default=str(here / "Datasets" / "visdrone"))
    parser.add_argument("--aircraft", default=str(here / "Datasets" / "military_aircraft"))
    parser.add_argument("--uav-tank", default=str(here / "Datasets" / "simuletic_uav_tank"))
    parser.add_argument("--dvids", default=str(here / "Datasets" / "dvids_military"))
    parser.add_argument("--openimages-tank", default=str(here / "Datasets" / "openimages_tank"))
    parser.add_argument(
        "--caps",
        default=V4_CAPS,
        help="Per-source train:val:test image caps (-1 = all); sources not listed are skipped. "
        f"kaal_v3 used: {V3_CAPS}",
    )
    parser.add_argument("--patch-frac", type=float, default=0.25)
    parser.add_argument(
        "--legacy-split",
        choices=["resplit", "original"],
        default="original",
        help="resplit = stratified 80/10/10 by content group (the original val/test "
        "folders contain no ground vehicles or soldiers; see module docstring)",
    )
    parser.add_argument(
        "--dup-dist",
        type=int,
        default=24,
        help="Legacy re-split: images within this many bits (256-bit dHash, flip-aware) "
        "are one photo (-1 = names only)",
    )
    parser.add_argument(
        "--cross-dup-dist",
        type=int,
        default=20,
        help="Near-duplicate grouping across ALL sources before splitting (dHash bits; "
        "-1 disables; kaal_v3 was built without it)",
    )
    parser.add_argument(
        "--eval-unique",
        action="store_true",
        help="Keep one image per near-duplicate group in val/test across sources "
        "(off for kaal_v4 reproducibility; recommended for the next version)",
    )
    parser.add_argument(
        "--keep-unlabeled-legacy",
        action="store_true",
        help="Keep legacy images without boxes (kaal_v3 behaviour)",
    )
    parser.add_argument(
        "--keep-openimages-person",
        action="store_true",
        help="Keep Open Images tank photos that contain people (people -> person)",
    )
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--min-box-px", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            parser.error(f"{out} exists (pass --overwrite to rebuild)")
        shutil.rmtree(out)

    loaders = {
        "legacy": lambda: legacy_samples(
            Path(args.legacy),
            args.patch_frac,
            args.seed,
            resplit=(0.8, 0.1, 0.1) if args.legacy_split == "resplit" else None,
            dup_dist=args.dup_dist,
            workers=args.workers,
            keep_unlabeled=args.keep_unlabeled_legacy,
        ),
        "visdrone": lambda: visdrone_samples(Path(args.visdrone)),
        "aircraft": lambda: aircraft_samples(Path(args.aircraft)),
        "uav_tank": lambda: uav_tank_samples(Path(args.uav_tank)),
        "dvids": lambda: dvids_samples(Path(args.dvids)),
        "openimages_tank": lambda: openimages_tank_samples(
            Path(args.openimages_tank), drop_person_images=not args.keep_openimages_person
        ),
    }
    caps = parse_caps(args.caps)
    unknown = set(caps) - set(loaders)
    if unknown:
        parser.error(f"unknown sources in --caps: {sorted(unknown)}")
    pool_all: List[Sample] = []
    for name, load in loaders.items():
        if name in caps:
            pool_all += load()
    groups_before = len({s.group for s in pool_all})
    if args.cross_dup_dist >= 0:
        # One group per near-identical photo across sources (e.g. the legacy
        # air_* images and the aircraft source share an upstream dataset).
        pool_all = merge_near_duplicate_groups(pool_all, args.cross_dup_dist, args.workers)
    pool_all, leaked_total = resolve_leakage(pool_all)
    eval_dupes = 0
    if args.eval_unique:
        pool_all, eval_dupes = keep_one_per_group_in_eval(pool_all)
    selected: List[Sample] = []
    for name in loaders:
        if name not in caps:
            continue
        for i, split in enumerate(SPLITS):
            pool = [s for s in pool_all if s.source == name and s.split == split]
            selected += rarity_sample(pool, caps[name][i], args.seed + i)

    stats = write_dataset(selected, out, args.max_side, args.min_box_px, args.workers)
    manifest = {
        "taxonomy": list(TAXONOMY),
        "args": vars(args),
        "cross_split_duplicates_removed": leaked_total,
        "groups_before_near_duplicate_merge": groups_before,
        "eval_duplicates_removed": eval_dupes,
        "licenses": {k: v for k, v in LICENSES.items() if k in caps},
        **stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Dataset written to %s", out)
    logger.info("Images: %s", json.dumps(stats["images"]))
    logger.info("Train instances: %s", json.dumps(stats["instances"]["train"]))


if __name__ == "__main__":
    main()
