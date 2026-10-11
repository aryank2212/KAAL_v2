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
             (the originals already carry Roboflow noise), and ``_patch``
             variants are kept for a fraction of training groups only, so
             augmented copies of one photo never straddle two splits.
  visdrone   VisDrone2019-DET (drone viewpoint). Pedestrians/people map to
             ``person`` — never ``soldier``: civilians must not be conflated
             with combatants. Ignored regions (category 0 / score 0) and
             "others" are dropped.
  aircraft   a2015003713 military-aircraft-detection-dataset (COCO json).
             Rotorcraft types map to ``helicopter``, everything else to
             ``aircraft``. Only images present on disk are used, so a partial
             download (``python -m MODEL.fetch_datasets aircraft``) works.
  uav_tank   Simuletic synthetic drone-view tanks -> ``tank``.

Usage:
    python -m MODEL.build_dataset --out MODEL/Datasets/kaal_v3
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
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

LICENSES = {
    "legacy": "KAAL project data (Roboflow exports; see original sources)",
    "visdrone": "VisDrone2019-DET — academic, non-commercial use",
    "aircraft": "a2015003713/military-aircraft-detection-dataset (Kaggle mirror; no license declared)",
    "uav_tank": "Simuletic UAV tank sample — CC BY-NC 4.0 per HF metadata (README says CC BY 4.0)",
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


def legacy_samples(root: Path, patch_frac: float, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    samples: List[Sample] = []
    dropped = 0
    split_dirs = {"train": "train", "val": "val", "test": "test"}
    for split, sub in split_dirs.items():
        img_dir, lbl_dir = root / sub / "images", root / sub / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            stem = img.stem
            if stem.endswith("_noisy"):
                continue
            is_patch = stem.endswith("_patch")
            group = stem[: -len("_patch")] if is_patch else stem
            if is_patch and (split != "train" or rng.random() >= patch_frac):
                continue
            boxes = parse_yolo_file(lbl_dir / f"{stem}.txt", LEGACY_MAP)
            if boxes is None:
                dropped += 1
                continue
            samples.append(
                Sample(
                    "legacy",
                    split,
                    img,
                    f"legacy:{group}",
                    (lambda b: lambda h, w: b)(boxes),
                    [b[0] for b in boxes],
                )
            )
    logger.info("legacy: %d samples, %d images dropped (unreliable class 0)", len(samples), dropped)
    return samples


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


def resolve_leakage(samples: List[Sample]) -> Tuple[List[Sample], int]:
    """Keep each group in a single split (priority test > val > train)."""
    rank = {"test": 0, "val": 1, "train": 2}
    home: Dict[str, str] = {}
    for s in samples:
        if s.group not in home or rank[s.split] < rank[home[s.group]]:
            home[s.group] = s.split
    kept = [s for s in samples if home[s.group] == s.split]
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
    parser = argparse.ArgumentParser(description="Build the cleaned KAAL v3 detection dataset.")
    parser.add_argument("--out", default=str(here / "Datasets" / "kaal_v3"))
    parser.add_argument(
        "--legacy", default=str(here / "runs" / "adversarial" / "augmented_dataset")
    )
    parser.add_argument("--visdrone", default=str(here / "Datasets" / "visdrone"))
    parser.add_argument("--aircraft", default=str(here / "Datasets" / "military_aircraft"))
    parser.add_argument("--uav-tank", default=str(here / "Datasets" / "simuletic_uav_tank"))
    parser.add_argument(
        "--caps",
        default="legacy=5000:400:400,visdrone=3000:400:400,aircraft=-1:-1:-1,uav_tank=-1:-1:-1",
        help="Per-source train:val:test image caps (-1 = all)",
    )
    parser.add_argument("--patch-frac", type=float, default=0.25)
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
        "legacy": lambda: legacy_samples(Path(args.legacy), args.patch_frac, args.seed),
        "visdrone": lambda: visdrone_samples(Path(args.visdrone)),
        "aircraft": lambda: aircraft_samples(Path(args.aircraft)),
        "uav_tank": lambda: uav_tank_samples(Path(args.uav_tank)),
    }
    caps = parse_caps(args.caps)
    selected: List[Sample] = []
    leaked_total = 0
    for name, load in loaders.items():
        if name not in caps:
            continue
        samples, leaked = resolve_leakage(load())
        leaked_total += leaked
        for i, split in enumerate(SPLITS):
            pool = [s for s in samples if s.split == split]
            selected += rarity_sample(pool, caps[name][i], args.seed + i)

    stats = write_dataset(selected, out, args.max_side, args.min_box_px, args.workers)
    manifest = {
        "taxonomy": list(TAXONOMY),
        "args": vars(args),
        "cross_split_duplicates_removed": leaked_total,
        "licenses": {k: v for k, v in LICENSES.items() if k in caps},
        **stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Dataset written to %s", out)
    logger.info("Images: %s", json.dumps(stats["images"]))
    logger.info("Train instances: %s", json.dumps(stats["instances"]["train"]))


if __name__ == "__main__":
    main()
