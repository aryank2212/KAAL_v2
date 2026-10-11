"""
build_ground_test.py — Independent ground-vehicle test set in the KAAL taxonomy.

The legacy KAAL data keeps every tank / APC / artillery / engineering vehicle
in its training folder, so ``kaal_v3`` cannot score those classes. This builds
a test-only set from public data that never enters training:

  orion_web  669 web images of named vehicle models, YOLO boxes
             (jonasrenault/orion v1.2.0; annotations MIT, images web-scraped —
             local evaluation only, never redistribute)
  russian    Russian-military-annotated v4 — T-64/72/80, BMP-1/2, BMD-2,
             BTR-70/80, MT-LB, BM-21 (Tuomo Hiippala, Univ. of Helsinki;
             CC BY 4.0). Roboflow export: augmented copies of one photo are
             collapsed to a single image.
  search2    TNO Search_2 — 44 natural scenes, one camouflaged vehicle each at
             2–5 km (Toet 2014; CC BY 4.0). 6144x4096 frames would shrink the
             target to a few pixels, so a 1024x1024 crop around it is used:
             the target keeps its native size, as through a narrow-FOV camera.

Light armoured cars (Orion "LAV", Search_2 "HVS"/"HVT": Humvee, VBL,
scout cars) fit no KAAL class; they are written as *ignore* regions
(``test/ignore/*.txt``) that ``compare_detectors`` neither rewards nor
penalises. Images that are near-duplicates of a ``kaal_v3`` training image
(flip-aware dHash) are excluded.

Download (≈257 MB) and build:
    gh release download v1.2.0 --repo jonasrenault/orion \\
        --pattern 'military-vehicles-dataset.tar.gz' --dir MODEL/Datasets/orion_ext
    gh release download v1.3.0 --repo jonasrenault/orion \\
        --pattern 'Russian-military-annotated*.zip' --pattern 'search_2.tar.gz' \\
        --dir MODEL/Datasets/orion_ext
    # extract into orion_ext/{google,russian,search2}, then:
    python -m MODEL.build_ground_test
    python -m MODEL.compare_detectors --data MODEL/Datasets/kaal_ground_test \\
        --eval-classes tank,armored_vehicle,artillery,engineering_vehicle --agnostic \\
        --detector rfdetr=... --out MODEL/runs/compare_ground
"""

import argparse
import csv
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .build_dataset import CLS, TAXONOMY, dhash, legacy_source, near_duplicate_pairs

logger = logging.getLogger(__name__)

IGNORE = -1
PxBox = Tuple[int, float, float, float, float]  # cls, x1, y1, x2, y2 (pixels)

ORION_CLASSES = ("AFV", "APC", "MEV", "LAV")  # obj.names order
ORION_TANK_MODELS = {
    "challenger tank", "leclerc tank", "Leopard tank", "M1 abrams", "T-72", "T-90",
    "VT-4", "ZTZ-96", "ZTZ-99",
}  # fmt: skip
RUSSIAN_MAP = {
    **{m: CLS["tank"] for m in ("t-64", "t-72", "t-80")},
    **{m: CLS["armored_vehicle"] for m in ("bmp-1", "bmp-2", "bmd-2", "btr-70", "btr-80", "mt-lb")},
    "bm-21": CLS["artillery"],  # multiple rocket launcher
}
SEARCH2_MAP = {
    **{m: CLS["tank"] for m in ("M1", "M60", "T72")},
    **{m: CLS["armored_vehicle"] for m in ("M3", "BMP", "BTR", "M113")},
    "HVS": IGNORE,
    "HVT": IGNORE,
}
LICENSES = {
    "orion_web": "jonasrenault/orion v1.2.0 — annotations MIT, images web-scraped (local eval only)",
    "russian": "Russian-military-annotated v4 (Tuomo Hiippala) — CC BY 4.0",
    "search2": "TNO Search_2 (Toet 2014, doi:10.6084/m9.figshare.1041463) — CC BY 4.0",
}


@dataclass
class TestImage:
    source: str
    name: str
    image: Path
    boxes: List[PxBox]
    crop: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


def orion_web_images(root: Path) -> List[TestImage]:
    out = []
    for label in sorted((root / "dataset" / "data").glob("*/*/*.txt")):
        folder = label.parent.name
        image = next(
            (p for p in label.parent.glob(f"{label.stem}.*") if p.suffix.lower() != ".txt"),
            None,
        )
        if image is None:
            continue
        img = cv2.imread(str(image))
        if img is None:  # e.g. WEBP without codec support
            continue
        h, w = img.shape[:2]
        boxes: List[PxBox] = []
        for line in label.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            kind = ORION_CLASSES[int(p[0])]
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            if kind == "AFV":
                cls = CLS["tank"] if folder in ORION_TANK_MODELS else CLS["armored_vehicle"]
            elif kind == "APC":
                cls = CLS["armored_vehicle"]
            elif kind == "MEV":
                cls = CLS["engineering_vehicle"]
            else:
                cls = IGNORE
            boxes.append(
                (cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            )
        out.append(TestImage("orion_web", image.stem, image, boxes))
    return out


def russian_images(root: Path) -> List[TestImage]:
    out, seen = [], set()
    for xml_path in sorted(root.glob("*/*.xml")):
        source = legacy_source(xml_path.stem)  # strips the Roboflow ".rf.<hash>"
        if source in seen:
            continue
        seen.add(source)
        tree = ET.parse(xml_path).getroot()
        boxes: List[PxBox] = []
        for obj in tree.iter("object"):
            name = (obj.findtext("name") or "").strip().lower()
            bb = obj.find("bndbox")
            if name not in RUSSIAN_MAP or bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((RUSSIAN_MAP[name], x1, y1, x2, y2))
        image = xml_path.with_suffix(".jpg")
        if boxes and image.exists():
            out.append(TestImage("russian", source, image, boxes))
    return out


def search2_images(root: Path, crop: int = 1024, seed: int = 0) -> List[TestImage]:
    rng = np.random.default_rng(seed)
    base = root / "search_2"
    out = []
    with open(base / "meta.csv", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            target = row["Target"].strip()
            if target not in SEARCH2_MAP:
                continue
            cx, cy, w, h = (float(row[k]) for k in ("X", "Y", "W", "H"))
            image = base / "images" / f"IMG{int(row['Image']):04d}.jpg"
            img_w, img_h = 6144, 4096
            # Random window that fully contains the target (deterministic per seed).
            x0 = int(rng.uniform(max(0, cx + w / 2 - crop), min(img_w - crop, cx - w / 2)))
            y0 = int(rng.uniform(max(0, cy + h / 2 - crop), min(img_h - crop, cy - h / 2)))
            box = (
                SEARCH2_MAP[target],
                cx - w / 2 - x0,
                cy - h / 2 - y0,
                cx + w / 2 - x0,
                cy + h / 2 - y0,
            )
            out.append(
                TestImage("search2", image.stem, image, [box], (x0, y0, x0 + crop, y0 + crop))
            )
    return out


def _load(t: TestImage) -> Optional[np.ndarray]:
    img = cv2.imread(str(t.image))
    if img is None:
        return None
    if t.crop:
        x1, y1, x2, y2 = t.crop
        img = img[y1:y2, x1:x2]
    return img


def _yolo_lines(boxes: List[PxBox], h: int, w: int) -> str:
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{max(c, 0)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return "".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build the independent KAAL ground-vehicle test set."
    )
    parser.add_argument("--src", default=str(here / "Datasets" / "orion_ext"))
    parser.add_argument("--out", default=str(here / "Datasets" / "kaal_ground_test"))
    parser.add_argument(
        "--train-ref", default=str(here / "Datasets" / "kaal_v3" / "train" / "images")
    )
    parser.add_argument("--dup-dist", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src, out = Path(args.src), Path(args.out)
    images = (
        orion_web_images(src / "google")
        + russian_images(src / "russian")
        + search2_images(src / "search2")
    )
    logger.info("candidates: %s", dict(Counter(t.source for t in images)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        crops = list(pool.map(_load, images))
    keep = [(t, img) for t, img in zip(images, crops) if img is not None]

    # Drop near-duplicates of training images, then within the test set.
    def hash_img(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (17, 16), interpolation=cv2.INTER_AREA
        )
        m = g[:, ::-1]
        return np.packbits(g[:, 1:] > g[:, :-1]), np.packbits(m[:, 1:] > m[:, :-1])

    test_h = [hash_img(img) for _, img in keep]
    train_paths = sorted(Path(args.train_ref).glob("*.jpg"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        train_h = [h for h in pool.map(dhash, train_paths) if h is not None]
    plain = np.stack([h[0] for h in test_h] + [h[0] for h in train_h])
    mirrored = np.stack([h[1] for h in test_h] + [h[1] for h in train_h])
    n_test = len(test_h)
    drop_train_dup, drop_self_dup = set(), set()
    for i, j in near_duplicate_pairs(plain, mirrored, args.dup_dist):
        if i < n_test <= j:
            drop_train_dup.add(i)
        elif j < n_test:
            drop_self_dup.add(j)
    removed = drop_train_dup | drop_self_dup

    if out.exists():
        shutil.rmtree(out)
    for sub in ("images", "labels", "ignore"):
        (out / "test" / sub).mkdir(parents=True)
    counts: Dict[str, Counter] = {}
    n_images: Counter = Counter()
    for i, (t, img) in enumerate(keep):
        if i in removed:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, args.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        boxes = [
            (c, x1 * scale, y1 * scale, x2 * scale, y2 * scale) for c, x1, y1, x2, y2 in t.boxes
        ]
        nh, nw = img.shape[:2]
        name = f"{t.source}_{t.name}".replace(" ", "_")
        cv2.imwrite(
            str(out / "test" / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        (out / "test" / "labels" / f"{name}.txt").write_text(
            _yolo_lines([b for b in boxes if b[0] != IGNORE], nh, nw)
        )
        ignore = _yolo_lines([b for b in boxes if b[0] == IGNORE], nh, nw)
        if ignore:
            (out / "test" / "ignore" / f"{name}.txt").write_text(ignore)
        n_images[t.source] += 1
        counts.setdefault(t.source, Counter()).update(
            "ignore" if b[0] == IGNORE else TAXONOMY[b[0]] for b in boxes
        )

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "test/images",  # test-only set; never train on it
                "val": "test/images",
                "test": "test/images",
                "nc": len(TAXONOMY),
                "names": dict(enumerate(TAXONOMY)),
            },
            sort_keys=False,
        )
    )
    manifest = {
        "images": dict(n_images),
        "instances": {s: dict(c) for s, c in counts.items()},
        "excluded_near_duplicates_of_train": len(drop_train_dup),
        "excluded_duplicates_within_test": len(drop_self_dup - drop_train_dup),
        "licenses": LICENSES,
        "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Ground test set written to %s: %s", out, json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
