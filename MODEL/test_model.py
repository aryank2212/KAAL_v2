#!/usr/bin/env python3
"""
KAAL Model Test Script
======================
Tests YOLO models on the aircraft test set and generates a comprehensive report.
Runs inference on all test images, compares with ground-truth labels,
and produces per-class and overall metrics.
"""

import os
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_BASE = os.path.join(_MODEL_DIR, "Datasets", "aircraft.v1i.yolov8")

TEST_IMAGES_DIR = os.path.join(_DATASET_BASE, "test", "images")
TEST_LABELS_DIR = os.path.join(_DATASET_BASE, "test", "labels")
DATA_YAML = os.path.join(_DATASET_BASE, "data.yaml")

# Models to test (paths relative to MODEL dir or absolute)
MODELS = {
    "YOLOv8m (base)": os.path.join(_MODEL_DIR, "yolov8m.pt"),
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

CLASS_NAMES = [
    "A10",
    "A400M",
    "AG600",
    "AV8B",
    "B1",
    "B2",
    "B52",
    "Be200",
    "C130",
    "C17",
    "C5",
    "E2",
    "EF2000",
    "F117",
    "F14",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "F4",
    "J20",
    "JAS39",
    "MQ9",
    "Mig31",
    "Mirage2000",
    "RQ4",
    "Rafale",
    "SR71",
    "Su34",
    "Su57",
    "Tornado",
    "Tu160",
    "Tu95",
    "U2",
    "US2",
    "V22",
    "Vulcan",
    "XB70",
    "YF23",
]


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-10)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized [cx, cy, w, h] → pixel [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_labels(label_path, img_w=640, img_h=640):
    """Load YOLO-format ground truth labels and convert to xyxy."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            xyxy = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            boxes.append({"cls_id": cls_id, "bbox": xyxy})
    return boxes


def test_model(model_path, model_name):
    """Run inference + evaluation for a single model."""
    logger.info("\n%s", "=" * 60)
    logger.info("  TESTING: %s", model_name)
    logger.info("  Weights: %s", model_path)
    logger.info("%s", "=" * 60)

    if not os.path.exists(model_path):
        logger.error("  [ERROR] Weights file not found: %s", model_path)
        return None

    model = YOLO(model_path)

    # ── 1. Run built-in YOLO validation on test split ────────────
    logger.info("\n[1/2] Running YOLO built-in validation on test split...")
    try:
        val_results = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            workers=0,
            plots=False,
            verbose=False,
        )
        builtin_map50 = val_results.box.map50
        builtin_map5095 = val_results.box.map
        builtin_prec = val_results.box.mp
        builtin_rec = val_results.box.mr
    except Exception as e:
        logger.warning("  [WARN] Built-in val failed: %s", e)
        builtin_map50 = builtin_map5095 = builtin_prec = builtin_rec = 0.0

    # ── 2. Run manual per-image inference with GT comparison ─────
    logger.info("[2/2] Running per-image inference with GT comparison...")
    images_dir = Path(TEST_IMAGES_DIR)
    labels_dir = Path(TEST_LABELS_DIR)
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    total_gt = 0
    total_det = 0
    tp = 0
    fp = 0
    fn = 0
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_det_count = defaultdict(int)
    class_gt_count = defaultdict(int)
    confidences = []

    t_start = time.time()

    for img_path in image_files:
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
            workers=0,
        )

        # Get image dimensions from results
        orig_shape = results[0].orig_shape  # (h, w)
        img_h, img_w = orig_shape

        # Load GT
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_labels(str(label_path), img_w, img_h)
        total_gt += len(gt_boxes)

        for gt in gt_boxes:
            cls_name = (
                CLASS_NAMES[gt["cls_id"]]
                if gt["cls_id"] < len(CLASS_NAMES)
                else f"cls_{gt['cls_id']}"
            )
            class_gt_count[cls_name] += 1

        # Parse detections
        dets = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu())
                conf = float(box.conf[0].cpu())
                cls_name = model.names.get(cls_id, f"cls_{cls_id}")
                dets.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf, "bbox": xyxy})
                confidences.append(conf)
                class_det_count[cls_name] += 1

        total_det += len(dets)

        # Match detections to GT (greedy IoU matching)
        gt_matched = [False] * len(gt_boxes)
        det_matched = [False] * len(dets)

        for di, det in enumerate(dets):
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= IOU_THRESHOLD and best_gi >= 0:
                gt_matched[best_gi] = True
                det_matched[di] = True
                tp += 1
                cls_name = (
                    CLASS_NAMES[gt_boxes[best_gi]["cls_id"]]
                    if gt_boxes[best_gi]["cls_id"] < len(CLASS_NAMES)
                    else f"cls_{gt_boxes[best_gi]['cls_id']}"
                )
                class_tp[cls_name] += 1
            else:
                fp += 1
                class_fp[det["cls_name"]] += 1

        for gi, matched in enumerate(gt_matched):
            if not matched:
                fn += 1
                cls_name = (
                    CLASS_NAMES[gt_boxes[gi]["cls_id"]]
                    if gt_boxes[gi]["cls_id"] < len(CLASS_NAMES)
                    else f"cls_{gt_boxes[gi]['cls_id']}"
                )
                class_fn[cls_name] += 1

    t_elapsed = time.time() - t_start
    fps = len(image_files) / t_elapsed if t_elapsed > 0 else 0

    # ── Build Report ─────────────────────────────────────────────
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    avg_conf = np.mean(confidences) if confidences else 0

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           KAAL MODEL TEST REPORT                             ║
╠══════════════════════════════════════════════════════════════╣
║  Model:      {model_name:<46s} ║
║  Weights:    {os.path.basename(model_path):<46s} ║
║  Test Set:   {len(image_files)} images, {total_gt} ground truth objects{" " * (24 - len(str(len(image_files))) - len(str(total_gt)))}║
║  Device:     {"CUDA" if torch.cuda.is_available() else "CPU":<46s} ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  BUILT-IN YOLO METRICS (test split)                          ║
║  ─────────────────────────────────                           ║
║    mAP@0.5:          {builtin_map50:<8.4f}                              ║
║    mAP@0.5:0.95:     {builtin_map5095:<8.4f}                              ║
║    Precision:         {builtin_prec:<8.4f}                              ║
║    Recall:            {builtin_rec:<8.4f}                              ║
║                                                              ║
║  MANUAL EVALUATION (IoU ≥ {IOU_THRESHOLD})                            ║
║  ──────────────────────────────                              ║
║    True  Positives:   {tp:<8d}                              ║
║    False Positives:   {fp:<8d}                              ║
║    False Negatives:   {fn:<8d}                              ║
║    Precision:         {precision:<8.4f}                              ║
║    Recall:            {recall:<8.4f}                              ║
║    F1 Score:          {f1:<8.4f}                              ║
║    Avg Confidence:    {avg_conf:<8.4f}                              ║
║                                                              ║
║  PERFORMANCE                                                 ║
║  ───────────                                                 ║
║    Total Inference:   {t_elapsed:<8.1f} seconds                         ║
║    Speed:             {fps:<8.1f} FPS                                ║
║    Total Detections:  {total_det:<8d}                              ║
╠══════════════════════════════════════════════════════════════╣
║  PER-CLASS BREAKDOWN (Top classes by GT count)               ║
╠══════════════════════════════════════════════════════════════╣
"""
    # Sort classes by GT count
    sorted_classes = sorted(class_gt_count.items(), key=lambda x: x[1], reverse=True)
    report += f"║  {'Class':<14s} {'GT':>5s} {'Det':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>7s} {'Rec':>7s}  ║\n"
    report += f"║  {'─' * 13:<14s}{'─' * 5:>5s} {'─' * 5:>5s} {'─' * 5:>5s} {'─' * 5:>5s} {'─' * 5:>5s} {'─' * 7:>7s} {'─' * 7:>7s}  ║\n"
    for cls_name, gt_c in sorted_classes[:20]:
        tp_c = class_tp.get(cls_name, 0)
        fp_c = class_fp.get(cls_name, 0)
        fn_c = class_fn.get(cls_name, 0)
        det_c = class_det_count.get(cls_name, 0)
        p_c = tp_c / (tp_c + fp_c + 1e-10)
        r_c = tp_c / (tp_c + fn_c + 1e-10)
        report += f"║  {cls_name:<14s}{gt_c:>5d} {det_c:>5d} {tp_c:>5d} {fp_c:>5d} {fn_c:>5d} {p_c:>7.3f} {r_c:>7.3f}  ║\n"

    report += """╚══════════════════════════════════════════════════════════════╝
"""

    logger.info("%s", report)

    # Save report
    report_path = os.path.join(
        os.path.dirname(model_path) if "runs" in model_path else _MODEL_DIR,
        f"test_report_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.txt",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("  Report saved to: %s", report_path)

    return {
        "model": model_name,
        "map50": builtin_map50,
        "map5095": builtin_map5095,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fps": fps,
    }


def main():
    logger.info("=" * 60)
    logger.info("   KAAL — YOLOv8 Aircraft Detection Model Testing")
    logger.info("=" * 60)
    logger.info("  Test Images: %s", TEST_IMAGES_DIR)
    logger.info("  PyTorch:     %s", torch.__version__)
    logger.info(
        "  CUDA:        %s%s",
        torch.cuda.is_available(),
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "",
    )

    all_results = []

    for name, path in MODELS.items():
        try:
            result = test_model(path, name)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error("\n  [ERROR] Failed to test %s: %s", name, e)
            import traceback

            traceback.print_exc()

    # ── Comparison Summary ───────────────────────────────────────
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("   MODEL COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info("  %-28s %8s %8s %8s %8s %8s", "Model", "mAP@50", "Prec", "Recall", "F1", "FPS")
        logger.info(
            "  %-28s %8s %8s %8s %8s %8s", "─" * 28, "─" * 8, "─" * 8, "─" * 8, "─" * 8, "─" * 8
        )
        for r in all_results:
            logger.info(
                "  %-28s %8.4f %8.4f %8.4f %8.4f %8.1f",
                r["model"],
                r["map50"],
                r["precision"],
                r["recall"],
                r["f1"],
                r["fps"],
            )

    logger.info("\n  Testing complete!")


if __name__ == "__main__":
    main()
