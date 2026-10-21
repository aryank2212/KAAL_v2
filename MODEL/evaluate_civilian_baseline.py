"""
evaluate_civilian_baseline.py — evaluate the civilian-only baseline.

Runs the trained civilian detector over the held-out, sequence-level test
split of ``auair_civilian`` and reports what was actually measured:

* COCO mAP@50 and mAP@50:95, overall and per class
* overall precision and recall, and AP by object-size bucket
* latency p50 / p95, throughput and peak GPU memory
* the same numbers under four **controlled, non-adversarial image-quality
  changes** — JPEG compression, blur, dimming and low light — together with
  the clean false-positive rate for reference

The transformed numbers describe *these* transformations on *these* images.
They are not a claim about real-world conditions.

The test split is never used to choose anything: the model, the resolution and
the epoch count are fixed before this runs.

Results are saved as a KAAL Bench manifest (``MODEL.bench``) with the metric
values and the SHA-256 of every artefact, next to a plain JSON dump and a
Markdown summary.

    python -m MODEL.evaluate_civilian_baseline \\
        --weights MODEL/runs/train/auair-civilian-rfdetr-small-v1/checkpoint_best_total.pth \\
        --data MODEL/Datasets/auair_civilian --split test \\
        --out MODEL/runs/eval/auair-civilian-v1
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from MODEL.bench import capture_manifest, render_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

#: Normalised-area buckets, matching MODEL/audit_civilian_dataset.py.
SIZE_BUCKETS = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, 1.01),
)


def jpeg_compress(image: np.ndarray, quality: int = 40) -> np.ndarray:
    """Re-encode as JPEG at ``quality`` — a compression change, not an attack."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def gaussian_blur(image: np.ndarray, kernel: int = 9, sigma: float = 0.0) -> np.ndarray:
    """Gaussian blur — a focus / motion softness change."""
    k = max(3, kernel | 1)
    return cv2.GaussianBlur(image, (k, k), sigma)


def dim(image: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """Scale every channel down — a darker exposure."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def low_light(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Gamma darkening — a low-light capture."""
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


#: The controlled conditions, with the parameters used.
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "clean": {"description": "unmodified frame", "transform": None},
    "jpeg_q40": {
        "description": "JPEG re-encode at quality 40",
        "transform": lambda image: jpeg_compress(image, 40),
    },
    "blur_k9": {
        "description": "Gaussian blur, 9x9 kernel, sigma 0",
        "transform": lambda image: gaussian_blur(image, 9, 0.0),
    },
    "dim_x0.6": {
        "description": "brightness multiplied by 0.6",
        "transform": lambda image: dim(image, 0.6),
    },
    "low_light_gamma2.2": {
        "description": "gamma 2.2 darkening",
        "transform": lambda image: low_light(image, 2.2),
    },
}


def read_split(
    data_dir: Path, split: str
) -> Tuple[List[Path], List[str], Dict[str, List[Tuple[int, float, float, float, float]]]]:
    """Image paths, class names, and the ground truth of a split.

    Ground truth is ``(class_id, cx, cy, w, h)`` with normalised coordinates,
    read straight from the YOLO label files.
    """
    config = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    images = sorted((data_dir / split / "images").glob("*.jpg"))
    truth: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    for image in images:
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        boxes: List[Tuple[int, float, float, float, float]] = []
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                cx, cy, width, height = (float(value) for value in parts[1:])
                boxes.append((class_id, cx, cy, width, height))
        truth[image.name] = boxes
    return images, class_names, truth


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coco_evaluate(
    truth: Sequence[List[Tuple[int, float, float, float, float]]],
    detections: Sequence[List[Dict[str, Any]]],
    sizes: Sequence[Tuple[int, int]],
    class_names: Sequence[str],
    operating_conf: float,
) -> Dict[str, Any]:
    """COCO metrics: mAP, per-class AP, precision/recall and AP by area range.

    ``truth`` holds normalised ``(class_id, cx, cy, w, h)`` boxes.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{"id": index, "name": name} for index, name in enumerate(class_names)]
    ground_truth: Dict[str, Any] = {
        "images": [{"id": index, "height": h, "width": w} for index, (h, w) in enumerate(sizes)],
        "categories": categories,
        "annotations": [],
    }
    for image_id, boxes in enumerate(truth):
        height, width = sizes[image_id]
        for class_id, cx, cy, box_width, box_height in boxes:
            pixel_w = box_width * width
            pixel_h = box_height * height
            ground_truth["annotations"].append(
                {
                    "id": len(ground_truth["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [
                        (cx - box_width / 2) * width,
                        (cy - box_height / 2) * height,
                        pixel_w,
                        pixel_h,
                    ],
                    "area": pixel_w * pixel_h,
                    "iscrowd": 0,
                }
            )
    results = [
        {
            "image_id": image_id,
            "category_id": det["class_id"],
            "bbox": [
                det["bbox"][0],
                det["bbox"][1],
                det["bbox"][2] - det["bbox"][0],
                det["bbox"][3] - det["bbox"][1],
            ],
            "score": det["confidence"],
        }
        for image_id, dets in enumerate(detections)
        for det in dets
        if 0 <= det["class_id"] < len(class_names)
    ]

    missed_total = sum(len(boxes) for boxes in truth)
    empty: Dict[str, Any] = {
        "mAP50": None,
        "mAP50_95": None,
        "precision": None,  # undefined without detections
        "recall": 0.0 if missed_total else None,  # measured: nothing was detected
        "per_class_mAP50_95": {},
        "per_class_mAP50": {},
        "AP_by_area": {},
        "operating_confidence": operating_conf,
        "detections_at_operating_confidence": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_positives_per_image": 0.0,
        "missed_ground_truth": missed_total,
    }
    if not results:
        return empty

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is chatty
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 100, 300]  # dense scenes exceed 100 objects
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    # eval["precision"] is [IoU thresholds, recall points, classes, area ranges, maxDets]
    full = evaluator.eval["precision"]
    precision = full[:, :, :, 0, -1]  # all area ranges, maxDets=300
    area_precision = full[:, :, :, :, -1]  # [IoU, recall, classes, area ranges]

    def mean(values: np.ndarray) -> Optional[float]:
        usable = values[values > -1]
        return float(usable.mean()) if usable.size else None

    per_class_95: Dict[str, float] = {}
    per_class_50: Dict[str, float] = {}
    for index, class_id in enumerate(sorted(range(len(class_names)), key=lambda c: c)):
        if (precision[:, :, index] > -1).any():
            per_class_95[class_names[class_id]] = round(mean(precision[:, :, index]) or 0.0, 4)
            per_class_50[class_names[class_id]] = round(mean(precision[0, :, index]) or 0.0, 4)

    # COCO area ranges: all, small, medium, large.
    area_labels = ["all", "small", "medium", "large"]
    by_area = {
        area_labels[index]: round(mean(area_precision[:, :, :, index]) or 0.0, 4)
        for index in range(min(4, area_precision.shape[3]))
    }

    # Precision / recall at the operating threshold, matched greedily by IoU.
    true_positive = 0
    false_positive = 0
    for image_id, dets in enumerate(detections):
        height, width = sizes[image_id]
        matched: set = set()
        for det in dets:
            if det["confidence"] < operating_conf:
                continue
            if not (0 <= det["class_id"] < len(class_names)):
                false_positive += 1
                continue
            box = det["bbox"]
            best_iou = 0.0
            best_index = None
            for index, (class_id, cx, cy, box_width, box_height) in enumerate(truth[image_id]):
                if class_id != det["class_id"] or index in matched:
                    continue
                candidate = [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ]
                overlap = _iou(box, candidate)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None and best_iou >= 0.5:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
        # Ground-truth boxes nobody detected count as misses.
    missed = sum(len(boxes) for boxes in truth) - true_positive

    detections_at_threshold = sum(
        1 for dets in detections for det in dets if det["confidence"] >= operating_conf
    )
    return {
        "mAP50": round(mean(precision[0]) or 0.0, 4),
        "mAP50_95": round(mean(precision) or 0.0, 4),
        "precision": round(true_positive / detections_at_threshold, 4)
        if detections_at_threshold
        else None,
        "recall": round(true_positive / (true_positive + missed), 4)
        if (true_positive + missed)
        else None,
        "per_class_mAP50_95": per_class_95,
        "per_class_mAP50": per_class_50,
        "AP_by_area": by_area,
        "operating_confidence": operating_conf,
        "detections_at_operating_confidence": detections_at_threshold,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_positives_per_image": round(false_positive / max(1, len(detections)), 4),
        "missed_ground_truth": missed,
    }


def size_bucket_ap(
    truth: Sequence[List[Tuple[int, float, float, float, float]]],
    detections: Sequence[List[Dict[str, Any]]],
    sizes: Sequence[Tuple[int, int]],
    class_names: Sequence[str],
    operating_conf: float,
) -> Dict[str, Any]:
    """AP and recall per normalised-area bucket, matched greedily at IoU 0.5.

    COCO's own area ranges are absolute pixels; on a 1920x1080 frame they do
    not line up with what "small" means here, so the buckets used by the
    dataset audit are reported instead. A detection is counted in the bucket
    of its own box; a ground-truth box is counted in the bucket of its own
    box.
    """
    buckets: Dict[str, Counter] = {name: Counter() for name, _low, _high in SIZE_BUCKETS}
    for image_id, dets in enumerate(detections):
        height, width = sizes[image_id]
        truth_buckets = [
            size_bucket(box_width * box_height)
            for (_class_id, _cx, _cy, box_width, box_height) in truth[image_id]
        ]
        used: set = set()
        for det in dets:
            if det["confidence"] < operating_conf:
                continue
            det_bucket = _bucket_of(det, (height, width))
            best_iou = 0.0
            best_index = None
            for index, (class_id, cx, cy, box_width, box_height) in enumerate(truth[image_id]):
                if class_id != det["class_id"] or index in used:
                    continue
                candidate = [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ]
                overlap = _iou(det["bbox"], candidate)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None and best_iou >= 0.5:
                used.add(best_index)
                buckets[det_bucket]["tp"] += 1
                buckets[truth_buckets[best_index]]["matched_gt"] += 1
            else:
                buckets[det_bucket]["fp"] += 1
        for index, bucket in enumerate(truth_buckets):
            buckets[bucket]["gt"] += 1
            if index not in used:
                buckets[bucket]["missed"] += 1
    out: Dict[str, Any] = {}
    for name, _low, _high in SIZE_BUCKETS:
        entry = buckets[name]
        detected = entry["tp"] + entry["fp"]
        out[name] = {
            "ground_truth": entry["gt"],
            "true_positives": entry["tp"],
            "false_positives": entry["fp"],
            "missed": entry["missed"],
            "precision": round(entry["tp"] / detected, 4) if detected else None,
            "recall": round(entry["matched_gt"] / entry["gt"], 4) if entry["gt"] else None,
        }
    return out


def _bucket_of(det: Dict[str, Any], size: Tuple[int, int]) -> str:
    height, width = size
    x1, y1, x2, y2 = det["bbox"]
    area = ((x2 - x1) / width) * ((y2 - y1) / height)
    return size_bucket(area)


def size_bucket(area: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= area < high:
            return name
    return SIZE_BUCKETS[-1][0]


def run_condition(
    detector: Any,
    images: Sequence[Path],
    transform: Optional[Callable[[np.ndarray], np.ndarray]],
    conf: float,
    warmup: int = 5,
) -> Tuple[List[List[Dict[str, Any]]], List[float], List[Tuple[int, int]]]:
    """Run the detector over every image, returning detections, latencies, sizes."""
    detections: List[List[Dict[str, Any]]] = []
    latencies: List[float] = []
    sizes: List[Tuple[int, int]] = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unreadable image: {path}")
        if transform is not None:
            image = transform(image)
        sizes.append((image.shape[0], image.shape[1]))
        start = time.perf_counter()
        detections.append(detector.predict(image, conf=conf))
        latencies.append((time.perf_counter() - start) * 1000.0)
    usable = latencies[warmup:] or latencies
    return detections, usable, sizes


def cuda_available() -> bool:
    """True when torch with a CUDA device is importable and present."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def reset_peak_memory() -> None:
    """Reset the CUDA peak-memory counter, if there is one."""
    if not cuda_available():
        return
    import torch

    torch.cuda.reset_peak_memory_stats()


def peak_memory_mib() -> Optional[float]:
    """Peak CUDA memory in MiB since the last reset, or None without a GPU."""
    if not cuda_available():
        return None
    import torch

    return round(torch.cuda.max_memory_allocated() / 1024**2, 1)


def percentiles(values: Sequence[float], points: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{point}": None for point in points}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory with data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.01, help="Detection threshold for mAP")
    parser.add_argument(
        "--operating-conf",
        type=float,
        default=0.25,
        help="Threshold for precision/recall and FP rates",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 = the whole split; a positive value samples deterministically",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated condition names to run",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    images, class_names, truth_by_name = read_split(args.data, args.split)
    if args.max_images and args.max_images < len(images):
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(len(images), size=args.max_images, replace=False).tolist())
        images = [images[index] for index in keep]
    truth = [truth_by_name[image.name] for image in images]
    logger.info("evaluating %d images from %s", len(images), args.split)

    from MODEL.detectors import RFDETRDetector

    detector = RFDETRDetector.from_weights(str(args.weights))
    logger.info("class names: %s", detector.names)

    reset_peak_memory()
    results: Dict[str, Any] = {}
    for name in [item.strip() for item in args.conditions.split(",") if item.strip()]:
        if name not in CONDITIONS:
            raise SystemExit(f"Unknown condition: {name}. Known: {sorted(CONDITIONS)}")
        spec = CONDITIONS[name]
        logger.info("condition %s — %s", name, spec["description"])
        transform = spec["transform"]
        reset_peak_memory()
        start = time.perf_counter()
        detections, latencies, sizes = run_condition(detector, images, transform, args.conf)
        wall = time.perf_counter() - start
        metrics = coco_evaluate(truth, detections, sizes, class_names, args.operating_conf)
        metrics["size_buckets"] = size_bucket_ap(
            truth, detections, sizes, class_names, args.operating_conf
        )
        metrics["description"] = spec["description"]
        metrics["latency_ms"] = {
            **percentiles(latencies, (50, 95)),
            "mean": float(statistics.fmean(latencies)) if latencies else None,
            "images_measured": len(latencies),
        }
        metrics["throughput_images_per_second"] = round(len(images) / wall, 3) if wall > 0 else None
        metrics["peak_gpu_memory_mib"] = peak_memory_mib()
        metrics["condition"] = name
        metrics["images_evaluated"] = len(images)
        results[name] = metrics
        logger.info(
            "%s: mAP50=%s mAP50:95=%s p95=%s ms",
            name,
            metrics["mAP50"],
            metrics["mAP50_95"],
            metrics["latency_ms"]["p95"],
        )

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "eval.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", json_path)

    bench_metrics: Dict[str, Optional[float]] = {}
    for name, entry in results.items():
        bench_metrics[f"{name}/mAP50"] = entry.get("mAP50")
        bench_metrics[f"{name}/mAP50_95"] = entry.get("mAP50_95")
        bench_metrics[f"{name}/precision"] = entry.get("precision")
        bench_metrics[f"{name}/recall"] = entry.get("recall")
        bench_metrics[f"{name}/false_positives_per_image"] = entry.get("false_positives_per_image")
        bench_metrics[f"{name}/latency_p50_ms"] = entry["latency_ms"].get("p50")
        bench_metrics[f"{name}/latency_p95_ms"] = entry["latency_ms"].get("p95")
    manifest = capture_manifest(
        run_id=args.out.name,
        model_name="auair-civilian-rfdetr-small-v1",
        dataset_id="auair_civilian",
        split=args.split,
        seed=42,
        evaluation_version="1",
        model_weights=args.weights,
        dataset_manifest=args.data / "manifest.json",
        metrics=bench_metrics,
        artifacts=[json_path],
        repository=args.data.parent.parent.parent,
    )
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s", manifest_path)

    report_path = args.out / "report.md"
    report_path.write_text(render_markdown(manifest), encoding="utf-8")
    logger.info("wrote %s", report_path)

    summary_path = args.out / "eval.md"
    summary_path.write_text(render_evaluation_markdown(results, args), encoding="utf-8")
    logger.info("wrote %s", summary_path)


def render_evaluation_markdown(results: Dict[str, Any], args: argparse.Namespace) -> str:
    """A concise, deterministic summary of the evaluation."""
    clean = results.get("clean", {})
    lines = [
        "# Civilian baseline evaluation",
        "",
        f"- **Weights:** `{args.weights}`",
        f"- **Dataset:** `{args.data}` — split `{args.split}`",
        f"- **Images evaluated:** {clean.get('images_evaluated', 'see eval.json')}",
        f"- **mAP threshold:** {args.conf} · **operating threshold:** {args.operating_conf}",
        "",
        "> The transformed rows are **controlled image-quality changes**, not real-world",
        "> conditions and not adversarial attacks. They say what this model does to these",
        "> specific images under these specific transformations.",
        "",
        "| Condition | mAP@50 | mAP@50:95 | Precision | Recall | FP/image | p50 ms | p95 ms | img/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in results.items():
        latency = entry["latency_ms"]
        lines.append(
            f"| {name} | {entry.get('mAP50')} | {entry.get('mAP50_95')} | "
            f"{entry.get('precision')} | {entry.get('recall')} | "
            f"{entry.get('false_positives_per_image')} | "
            f"{latency.get('p50')} | {latency.get('p95')} | "
            f"{entry.get('throughput_images_per_second')} |"
        )
    lines.extend(
        [
            "",
            "## Per-class mAP@50:95 (clean)",
            "",
            "| Class | mAP@50:95 | mAP@50 |",
            "|---|---|---|",
        ]
    )
    per_class_95 = clean.get("per_class_mAP50_95", {})
    per_class_50 = clean.get("per_class_mAP50", {})
    for name in sorted(per_class_95):
        lines.append(f"| {name} | {per_class_95[name]} | {per_class_50.get(name)} |")
    lines.extend(
        [
            "",
            "## Object size (clean)",
            "",
            "| Bucket | ground truth | true positives | missed | precision | recall |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for name, entry in clean.get("size_buckets", {}).items():
        lines.append(
            f"| {name} | {entry['ground_truth']} | {entry['true_positives']} | "
            f"{entry['missed']} | {entry['precision']} | {entry['recall']} |"
        )
    if clean.get("peak_gpu_memory_mib") is not None:
        lines.extend(["", f"- Peak GPU memory: **{clean['peak_gpu_memory_mib']} MiB**"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
