"""
evaluator.py — Evaluation metrics for adversarial robustness.

Computes:
  - mAP on clean dataset
  - mAP on adversarial dataset (attack without defense)
  - mAP with dual-stream defense enabled
  - Per-class confusion matrix
  - Formatted metrics report
"""

import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from ultralytics import YOLO

from .config import Config
from .adversarial_patch import AdversarialPatchInjector, parse_yolo_labels
from .dual_stream_defense import DualStreamDefense

logger = logging.getLogger(__name__)

# Base seed for patch injection; identical across attacked/defended passes.
ATTACK_SEED = 1234


class RobustnessEvaluator:
    """
    Evaluates a YOLO model's robustness against adversarial patches
    under three conditions: clean, attacked, and defended.
    """

    def __init__(self, model: YOLO, config: Config):
        self.model = model
        self.config = config
        self.injector = AdversarialPatchInjector(
            patch_type=config.patch_type,
            scale_range=config.patch_scale_range,
            alpha=config.patch_alpha,
            patch_image_path=config.patch_image_path,
            model=model if config.patch_gradient else None,
            use_perspective=config.patch_perspective,
            use_lighting=config.patch_lighting,
            grad_steps=config.patch_opt_steps,
            grad_lr=config.patch_opt_lr,
        )
        self.defense = DualStreamDefense(
            model=model,
            fft_threshold=config.fft_threshold,
            high_freq_ratio=config.fft_high_freq_ratio,
            conf_threshold=config.conf_threshold,
            region_mode=config.fft_region_mode,
            region_min_size=config.fft_region_min_size,
            overlap_iou=config.fft_overlap_iou,
            calibration_profile=config.calibration_profile,
        )
        self.results: Dict[str, Any] = {}

    # ── Main Evaluation Entry Point ───────────────────────────────────────

    def run_full_evaluation(self, val_images_dir: str, val_labels_dir: str):
        """
        Execute all three evaluation passes and store results.

        Args:
            val_images_dir: Directory of validation images.
            val_labels_dir: Directory of YOLO-format label files.
        """
        logger.info("\n%s", "=" * 60)
        logger.info("  Robustness Evaluation")
        logger.info("%s\n", "=" * 60)

        # Collect image/label pairs
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = sorted(
            [f for f in os.listdir(val_images_dir) if Path(f).suffix.lower() in img_extensions]
        )

        if not image_files:
            logger.warning("No validation images found!")
            return

        logger.info(f"Found {len(image_files)} validation images")

        # ── Pass 1: Clean mAP ────────────────────────────────────────────
        logger.info("── Pass 1: Clean Dataset (no attack) ──")
        clean_metrics = self._evaluate_pass(
            image_files,
            val_images_dir,
            val_labels_dir,
            apply_attack=False,
            use_defense=False,
        )
        self.results["clean"] = clean_metrics
        logger.info("   mAP@0.5: %.4f\n", clean_metrics["mAP50"])

        # ── Pass 2: Adversarial mAP (no defense) ─────────────────────────
        logger.info("── Pass 2: Adversarial Attack (no defense) ──")
        adv_metrics = self._evaluate_pass(
            image_files,
            val_images_dir,
            val_labels_dir,
            apply_attack=True,
            use_defense=False,
        )
        self.results["adversarial"] = adv_metrics
        logger.info("   mAP@0.5: %.4f\n", adv_metrics["mAP50"])

        # ── Pass 3: Adversarial + Dual-Stream Defense ─────────────────────
        logger.info("── Pass 3: Adversarial Attack + Dual-Stream Defense ──")
        defended_metrics = self._evaluate_pass(
            image_files,
            val_images_dir,
            val_labels_dir,
            apply_attack=True,
            use_defense=True,
        )
        self.results["defended"] = defended_metrics
        logger.info("   mAP@0.5: %.4f", defended_metrics["mAP50"])
        logger.info(
            "   Adversarial images detected: %d/%d\n",
            defended_metrics["adversarial_detected"],
            len(image_files),
        )

        # ── Print Summary ─────────────────────────────────────────────────
        self.print_report()

    # ── Evaluation Pass ───────────────────────────────────────────────────

    def _evaluate_pass(
        self,
        image_files: List[str],
        images_dir: str,
        labels_dir: str,
        apply_attack: bool,
        use_defense: bool,
    ) -> Dict[str, Any]:
        """
        Single evaluation pass over the validation set.

        Args:
            image_files:   List of image filenames.
            images_dir:    Path to image directory.
            labels_dir:    Path to label directory.
            apply_attack:  Whether to inject adversarial patches.
            use_defense:   Whether to use dual-stream defense.

        Returns:
            Dict with mAP50, per-class metrics, confusion counts, etc.
        """
        all_gt_boxes = []  # Ground truth: list of (class_id, bbox) per image
        all_pred_boxes = []  # Predictions: list of (class_id, confidence, bbox) per image
        adversarial_detected = 0
        total_anomaly_scores = []
        total_objects_flagged = 0

        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, Path(img_name).stem + ".txt")

            image = cv2.imread(img_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            # Parse ground truth
            class_ids, bboxes = parse_yolo_labels(label_path, img_w, img_h)

            # Convert ground truth to (x1,y1,x2,y2)
            gt_boxes = []
            for cls_id, (cx, cy, w, h) in zip(class_ids, bboxes):
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                gt_boxes.append((cls_id, [x1, y1, x2, y2]))
            all_gt_boxes.append(gt_boxes)

            # Apply adversarial attack if requested
            if apply_attack and bboxes:
                # Seed per image so the attacked and defended passes see the
                # *same* patches — otherwise their mAP gap is just patch noise.
                np.random.seed(ATTACK_SEED + idx)
                image, _ = self.injector.apply_patch(image, bboxes, prob=1.0)

            # Run inference
            if use_defense:
                result = self.defense.predict(image)
                detections = result["detections"]
                total_anomaly_scores.append(result["anomaly_score"])
                total_objects_flagged += result.get("objects_flagged", 0)
                if result["is_adversarial"]:
                    adversarial_detected += 1
            else:
                detections = self.defense.yolo_predict(image)

            # Collect predictions
            pred_boxes = []
            for det in detections:
                pred_boxes.append(
                    (
                        det["class_id"],
                        det["confidence"],
                        det["bbox"],
                    )
                )
            all_pred_boxes.append(pred_boxes)

        # Compute mAP
        mAP50 = self._compute_mAP(all_gt_boxes, all_pred_boxes, iou_threshold=0.5)

        # Confusion matrix counts
        confusion = self._compute_confusion(all_gt_boxes, all_pred_boxes)

        return {
            "mAP50": mAP50,
            "confusion": confusion,
            "adversarial_detected": adversarial_detected,
            "anomaly_scores": total_anomaly_scores,
            "objects_flagged": total_objects_flagged,
            "num_images": len(image_files),
        }

    # ── mAP Computation ──────────────────────────────────────────────────

    @staticmethod
    def _compute_mAP(
        all_gt: List[List[Tuple]],
        all_preds: List[List[Tuple]],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Compute mean Average Precision at a given IoU threshold.

        Uses the standard VOC-style per-class AP computation.
        """
        # Collect all predictions and ground truths by class
        class_preds = defaultdict(list)  # class_id → [(confidence, img_idx, bbox)]
        class_gt = defaultdict(list)  # class_id → [(img_idx, bbox, matched)]
        gt_count: Dict[int, int] = defaultdict(int)  # class_id → total ground truth count

        for img_idx, gt_boxes in enumerate(all_gt):
            for cls_id, bbox in gt_boxes:
                class_gt[cls_id].append({"img_idx": img_idx, "bbox": bbox, "matched": False})
                gt_count[cls_id] += 1

        for img_idx, pred_boxes in enumerate(all_preds):
            for cls_id, conf, bbox in pred_boxes:
                class_preds[cls_id].append({"conf": conf, "img_idx": img_idx, "bbox": bbox})

        all_classes = set(list(gt_count.keys()) + list(class_preds.keys()))
        if not all_classes:
            return 0.0

        aps = []
        for cls_id in all_classes:
            preds = class_preds[cls_id]
            gts = class_gt[cls_id]
            n_gt = gt_count[cls_id]

            if n_gt == 0:
                continue

            # Sort predictions by confidence (descending)
            preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

            # Reset matched flags
            for g in gts:
                g["matched"] = False

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, pred in enumerate(preds):
                best_iou = 0.0
                best_gt_idx = -1

                for j, gt in enumerate(gts):
                    if gt["img_idx"] != pred["img_idx"]:
                        continue
                    iou = _compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if (
                    best_iou >= iou_threshold
                    and best_gt_idx >= 0
                    and not gts[best_gt_idx]["matched"]
                ):
                    tp[i] = 1
                    gts[best_gt_idx]["matched"] = True
                else:
                    fp[i] = 1

            # Compute precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall = tp_cumsum / n_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # AP via 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                prec_at_recall = precision[recall >= t]
                if len(prec_at_recall) > 0:
                    ap += np.max(prec_at_recall)
            ap /= 11.0
            aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0

    @staticmethod
    def _compute_confusion(
        all_gt: List[List[Tuple]],
        all_preds: List[List[Tuple]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, int]:
        """Compute overall TP, FP, FN counts across all classes."""
        tp, fp, fn = 0, 0, 0

        for gt_boxes, pred_boxes in zip(all_gt, all_preds):
            matched_gt = set()
            for cls_id, conf, bbox in pred_boxes:
                best_iou = 0.0
                best_idx = -1
                for j, (gt_cls, gt_bbox) in enumerate(gt_boxes):
                    if gt_cls != cls_id or j in matched_gt:
                        continue
                    iou = _compute_iou(bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_iou >= iou_threshold and best_idx >= 0:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

        return {"TP": tp, "FP": fp, "FN": fn}

    # ── Report ────────────────────────────────────────────────────────────

    def print_report(self) -> str:
        """Build and log a formatted evaluation report; returns the report text."""
        report = []
        report.append(f"\n{'=' * 60}")
        report.append("  ROBUSTNESS EVALUATION REPORT")
        report.append(f"{'=' * 60}\n")

        headers = ["Scenario", "mAP@0.5", "TP", "FP", "FN"]
        rows = []

        for scenario, label in [
            ("clean", "Clean (baseline)"),
            ("adversarial", "Adversarial (no defense)"),
            ("defended", "Adversarial + Defense"),
        ]:
            if scenario in self.results:
                m = self.results[scenario]
                c = m["confusion"]
                rows.append(
                    [
                        label,
                        f"{m['mAP50']:.4f}",
                        str(c["TP"]),
                        str(c["FP"]),
                        str(c["FN"]),
                    ]
                )

        if rows:
            col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            separator = "-+-".join("-" * w for w in col_widths)

            report.append(f"  {header_line}")
            report.append(f"  {separator}")
            for row in rows:
                report.append(f"  {' | '.join(v.ljust(w) for v, w in zip(row, col_widths))}")

        # Defense-specific stats
        if "defended" in self.results:
            d = self.results["defended"]
            report.append(
                f"\n  Adversarial images flagged: {d['adversarial_detected']}/{d['num_images']}"
            )
            if d["anomaly_scores"]:
                report.append(f"  Mean anomaly score: {np.mean(d['anomaly_scores']):.4f}")
                report.append(f"  Max anomaly score:  {np.max(d['anomaly_scores']):.4f}")

        # mAP drop analysis
        if "clean" in self.results and "adversarial" in self.results:
            drop = self.results["clean"]["mAP50"] - self.results["adversarial"]["mAP50"]
            report.append(f"\n  mAP drop from attack: {drop:+.4f}")

        if "adversarial" in self.results and "defended" in self.results:
            # The dual-stream defense *flags* tampered inputs; it does not alter
            # detections, so defended mAP equals attacked mAP on identical
            # patches. The defense metric is the flag rate reported above.
            delta = self.results["defended"]["mAP50"] - self.results["adversarial"]["mAP50"]
            report.append(
                f"  Defended vs attacked mAP: {delta:+.4f} (expected 0 — the defense flags, "
                "it does not correct; see flag rate)"
            )

        report.append(f"\n{'=' * 60}\n")

        text = "\n".join(report)
        logger.info("%s", text)
        return text

    def save_report(self, path: str):
        """Save evaluation results to a text file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.print_report())
        logger.info("Report saved to: %s", path)


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-10)
