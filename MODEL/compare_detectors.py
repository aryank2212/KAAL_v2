"""
compare_detectors.py — Head-to-head evaluation of KAAL detectors.

Runs every detector on the same test split twice — clean, and with
adversarial patches injected inside every ground-truth box — and reports:

  * COCO mAP@50 and mAP@50:95 (pycocotools), overall and per class
  * mAP drop under attack (robustness)
  * median inference latency (ms / image)
  * FFT defense: share of patched GT boxes flagged vs clean GT boxes flagged
    (model-free, identical for every detector — the Stream 2 safety net)

Example:
    python -m MODEL.compare_detectors \\
        --detector rfdetr=MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth \\
        --detector yolo26s=MODEL/runs/train/yolo26s/weights/best.pt

Writes ``runs/compare/report.json`` and ``runs/compare/report.md``.
"""

import argparse
import contextlib
import io
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

from .adversarial_patch import AdversarialPatchInjector
from .config import PatchType
from .detectors import load_detector
from .fft_scorer import DEFAULT_FFT_THRESHOLD, compute_region_scores

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def load_split(data_dir: Path, split: str, max_images: int) -> Tuple[List[Path], List[str]]:
    names = yaml.safe_load((data_dir / "data.yaml").read_text())["names"]
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    images = sorted((data_dir / split / "images").glob("*.jpg"))
    if 0 < max_images < len(images):
        keep = np.random.default_rng(0).choice(len(images), size=max_images, replace=False)
        images = sorted(images[i] for i in keep)
    return images, class_names


def imread(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    return image


def read_gt(label_path: Path, h: int, w: int) -> List[Tuple[int, List[float]]]:
    """YOLO label -> [(cls, [x, y, bw, bh] pixels)] (COCO bbox convention)."""
    out = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]))
    return out


def patched_copy(image: np.ndarray, gt: List[Tuple[int, List[float]]], seed: int) -> np.ndarray:
    """Inject a noise or pattern patch inside every GT box (deterministic per image)."""
    np.random.seed(seed)
    injector = AdversarialPatchInjector(
        patch_type=PatchType.NOISE if seed % 2 == 0 else PatchType.PATTERN,
        scale_range=(0.15, 0.4),
    )
    centers = [
        (int(x + bw / 2), int(y + bh / 2), int(bw), int(bh))
        for _, (x, y, bw, bh) in gt
        if bw >= 8 and bh >= 8
    ]
    patched, _ = injector.apply_patch(image, centers, prob=1.0)
    return patched


def coco_map(
    gts: List[List[Tuple[int, List[float]]]],
    dets: List[List[Dict[str, Any]]],
    sizes: List[Tuple[int, int]],
    class_names: List[str],
) -> Dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_json: Dict[str, Any] = {
        "images": [{"id": i, "height": h, "width": w} for i, (h, w) in enumerate(sizes)],
        "categories": [{"id": c, "name": n} for c, n in enumerate(class_names)],
        "annotations": [],
    }
    for img_id, boxes in enumerate(gts):
        for c, (x, y, bw, bh) in boxes:
            gt_json["annotations"].append(
                {
                    "id": len(gt_json["annotations"]) + 1,
                    "image_id": img_id,
                    "category_id": c,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
    dt_json = [
        {
            "image_id": img_id,
            "category_id": d["class_id"],
            "bbox": [
                d["bbox"][0],
                d["bbox"][1],
                d["bbox"][2] - d["bbox"][0],
                d["bbox"][3] - d["bbox"][1],
            ],
            "score": d["confidence"],
        }
        for img_id, ds in enumerate(dets)
        for d in ds
        if 0 <= d["class_id"] < len(class_names)
    ]
    if not dt_json:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_mAP50_95": {}}

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is chatty
        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_json)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 100, 300]  # dense drone scenes exceed 100 objects
        ev.evaluate()
        ev.accumulate()

    # precision: [IoU thresholds, recall points, classes, area ranges, maxDets];
    # read it directly so the 300-detection cap is used for every number.
    precision = ev.eval["precision"][:, :, :, 0, -1]

    def _mean(p: np.ndarray) -> float:
        p = p[p > -1]
        return float(p.mean()) if p.size else 0.0

    per_class = {}
    for k, name in enumerate(class_names):
        if (precision[:, :, k] > -1).any():
            per_class[name] = round(_mean(precision[:, :, k]), 4)
    return {
        "mAP50": round(_mean(precision[0]), 4),
        "mAP50_95": round(_mean(precision), 4),
        "per_class_mAP50_95": per_class,
    }


def _auroc(neg: List[float], pos: List[float]) -> float:
    if not neg or not pos:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1
    return float((ranks[len(neg) :].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def fft_flag_rates(images, gts, threshold: float) -> Dict[str, float]:
    """Model-free: does Stream 2 flag patched GT regions but not clean ones?"""
    clean_scores: List[float] = []
    patched_scores: List[float] = []
    for idx, (path, gt) in enumerate(zip(images, gts)):
        img = imread(path)
        boxes = [(x, y, x + bw, y + bh) for _, (x, y, bw, bh) in gt]
        clean = compute_region_scores(img, boxes)
        patched = compute_region_scores(patched_copy(img, gt, idx), boxes)
        for c, p in zip(clean, patched):
            if c is not None and p is not None:
                clean_scores.append(c)
                patched_scores.append(p)
    return {
        "threshold": threshold,
        "patched_box_flag_rate": round(float(np.mean(np.array(patched_scores) > threshold)), 4)
        if patched_scores
        else 0.0,
        "clean_box_false_flag_rate": round(float(np.mean(np.array(clean_scores) > threshold)), 4)
        if clean_scores
        else 0.0,
        "auroc": round(_auroc(clean_scores, patched_scores), 4),
        "boxes_scored": len(clean_scores),
    }


def evaluate(detector, images, gts, conf: float, attack: bool) -> Tuple[List, List[float]]:
    dets, latencies = [], []
    for idx, (path, gt) in enumerate(zip(images, gts)):
        img = imread(path)
        if attack:
            img = patched_copy(img, gt, idx)
        t0 = time.perf_counter()
        dets.append(detector.predict(img, conf=conf))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return dets, latencies[5:] or latencies  # drop warm-up iterations


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# KAAL detector comparison",
        "",
        f"Test split: `{report['data']}` ({report['n_images']} images). "
        "Attacked = noise/pattern patch inside every ground-truth box.",
        "",
        "| Detector | mAP50 | mAP50:95 | mAP50:95 attacked | Drop | Latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["detectors"].items():
        lines.append(
            f"| {name} | {r['clean']['mAP50']:.3f} | {r['clean']['mAP50_95']:.3f} | "
            f"{r['attacked']['mAP50_95']:.3f} | {r['robustness_drop']:+.3f} | {r['latency_ms']:.1f} |"
        )
    classes = sorted(
        {c for r in report["detectors"].values() for c in r["clean"]["per_class_mAP50_95"]}
    )
    lines += ["", "## Per-class mAP50:95 (clean)", "", "| Class | " + " | ".join(report["detectors"]) + " |",
              "|---|" + "---|" * len(report["detectors"])]  # fmt: skip
    for c in classes:
        vals = [
            f"{r['clean']['per_class_mAP50_95'].get(c, 0):.3f}"
            for r in report["detectors"].values()
        ]
        lines.append(f"| {c} | " + " | ".join(vals) + " |")
    f = report["fft_defense"]
    lines += [
        "",
        "## FFT defense (Stream 2, model-free)",
        "",
        f"At threshold {f['threshold']:.3f}: patched boxes flagged **{f['patched_box_flag_rate']:.1%}**, "
        f"clean boxes falsely flagged **{f['clean_box_false_flag_rate']:.1%}** "
        f"({f['boxes_scored']} boxes; threshold-free AUROC {f['auroc']:.3f}).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KAAL detectors on one test split.")
    parser.add_argument(
        "--detector", action="append", required=True, help="name=weights (repeatable)"
    )
    parser.add_argument("--data", default=str(HERE / "Datasets" / "kaal_v3"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=0, help="0 = whole split")
    parser.add_argument("--conf", type=float, default=0.01, help="Low threshold for mAP")
    parser.add_argument("--fft-threshold", type=float, default=DEFAULT_FFT_THRESHOLD)
    parser.add_argument("--out", default=str(HERE / "runs" / "compare"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data)
    images, class_names = load_split(data_dir, args.split, args.max_images)
    sizes, gts = [], []
    for p in images:
        h, w = imread(p).shape[:2]
        sizes.append((h, w))
        gts.append(read_gt(data_dir / args.split / "labels" / f"{p.stem}.txt", h, w))
    logger.info("Evaluating on %d %s images", len(images), args.split)

    report: Dict[str, Any] = {
        "data": str(data_dir),
        "split": args.split,
        "n_images": len(images),
        "detectors": {},
    }
    for spec in args.detector:
        name, weights = spec.split("=", 1)
        detector = load_detector(None, weights)
        clean_dets, lat = evaluate(detector, images, gts, args.conf, attack=False)
        attacked_dets, _ = evaluate(detector, images, gts, args.conf, attack=True)
        clean = coco_map(gts, clean_dets, sizes, class_names)
        attacked = coco_map(gts, attacked_dets, sizes, class_names)
        report["detectors"][name] = {
            "weights": weights,
            "clean": clean,
            "attacked": attacked,
            "robustness_drop": round(attacked["mAP50_95"] - clean["mAP50_95"], 4),
            "latency_ms": round(statistics.median(lat), 2),
        }
        logger.info(
            "%s: %s",
            name,
            json.dumps({k: v for k, v in report["detectors"][name].items() if k != "clean"}),
        )

    report["fft_defense"] = fft_flag_rates(images, gts, args.fft_threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(render_markdown(report))
    logger.info("Report written to %s", out / "report.md")


if __name__ == "__main__":
    main()
