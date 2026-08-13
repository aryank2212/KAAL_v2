"""
visualizer.py — Visualization utilities for adversarial robustness evaluation.

Generates:
  - Before/after adversarial patch comparison images
  - Bounding box overlays with confidence scores
  - Adversarial detection flags
  - Sample prediction grids
  - Confusion matrix heatmap
"""

import os
import cv2
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .config import Config
from .adversarial_patch import AdversarialPatchInjector, parse_yolo_labels
from .dual_stream_defense import DualStreamDefense

logger = logging.getLogger(__name__)


# ── Color Palette ─────────────────────────────────────────────────────────
# Distinct colors for up to 20 classes (BGR)
COLORS = [
    (255, 77, 77),
    (77, 255, 77),
    (77, 77, 255),
    (255, 255, 77),
    (255, 77, 255),
    (77, 255, 255),
    (255, 165, 0),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (214, 39, 40),
    (44, 160, 44),
    (31, 119, 180),
    (255, 127, 14),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
]


class ResultVisualizer:
    """
    Creates publication-quality visualizations for adversarial robustness analysis.
    """

    def __init__(self, config: Config, defense: DualStreamDefense):
        self.config = config
        self.defense = defense
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.injector = AdversarialPatchInjector(
            patch_type=config.patch_type,
            scale_range=config.patch_scale_range,
            alpha=config.patch_alpha,
            patch_image_path=config.patch_image_path,
            use_perspective=config.patch_perspective,
            use_lighting=config.patch_lighting,
            grad_steps=config.patch_opt_steps,
            grad_lr=config.patch_opt_lr,
        )

    # ── Before / After Adversarial Comparison ─────────────────────────────

    def show_adversarial_comparison(
        self,
        images_dir: str,
        labels_dir: str,
        num_samples: int = 4,
    ) -> str:
        """
        Generate side-by-side comparison images showing clean vs. patched.

        Returns:
            Path to the saved comparison image.
        """
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in os.listdir(images_dir) if Path(f).suffix.lower() in img_extensions
        ]
        # Select a subset
        np.random.shuffle(image_files)
        selected = image_files[:num_samples]

        panels = []
        for img_name in selected:
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, Path(img_name).stem + ".txt")

            image = cv2.imread(img_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            # Parse bboxes for patch placement
            _, bboxes = parse_yolo_labels(label_path, img_w, img_h)

            # Create adversarial version
            patched, _ = self.injector.apply_patch(image, bboxes, prob=1.0)

            # Resize both to a common height for display
            display_h = 320
            scale = display_h / img_h
            display_w = int(img_w * scale)
            clean_resized = cv2.resize(image, (display_w, display_h))
            patched_resized = cv2.resize(patched, (display_w, display_h))

            # Add labels
            _add_label(clean_resized, "CLEAN", (0, 180, 0))
            _add_label(patched_resized, "ADVERSARIAL", (0, 0, 220))

            # Side by side
            panel = np.hstack([clean_resized, patched_resized])
            panels.append(panel)

        if not panels:
            logger.warning("No images to compare")
            return ""

        # Stack vertically
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                p = np.hstack([p, pad])
            padded.append(p)

        grid = np.vstack(padded)
        output_path = os.path.join(self.output_dir, "adversarial_comparison.png")
        cv2.imwrite(output_path, grid)
        logger.info(f"Adversarial comparison saved to: {output_path}")
        return output_path

    # ── Prediction Visualization ──────────────────────────────────────────

    def draw_predictions(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        ground_truth: Optional[List[Tuple[int, List[float]]]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, confidence scores, and adversarial flags on an image.

        Args:
            image:        BGR uint8 image.
            result:       Output from DualStreamDefense.predict().
            ground_truth: Optional list of (class_id, [x1,y1,x2,y2]) for GT overlay.

        Returns:
            Annotated image.
        """
        canvas = image.copy()

        # Draw ground truth boxes (dashed, thin, gray)
        if ground_truth:
            for cls_id, bbox in ground_truth:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                _draw_dashed_rect(canvas, (x1, y1), (x2, y2), (180, 180, 180), thickness=1)
                cv2.putText(
                    canvas,
                    f"GT:{cls_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                )

        # Draw predicted boxes
        for det in result["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_id = det["class_id"]
            conf = det["confidence"]
            name = det["class_name"]
            # Per-region adversarial flag overrides the class color (red box)
            if det.get("is_adversarial"):
                color = (0, 0, 255)
            else:
                color = COLORS[cls_id % len(COLORS)]

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            if det.get("region_scored") and det.get("anomaly_score") is not None:
                label += f" fft={det['anomaly_score']:.2f}"
            _draw_label_bg(canvas, label, (x1, y1), color)

        # Adversarial detection overlay
        if result["is_adversarial"]:
            _draw_adversarial_banner(canvas, result["anomaly_score"])

        # Anomaly score indicator (bottom-right)
        score_text = f"FFT Score: {result['anomaly_score']:.3f}"
        score_color = (0, 0, 255) if result["is_adversarial"] else (0, 200, 0)
        h, w = canvas.shape[:2]
        cv2.putText(
            canvas, score_text, (w - 220, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 2
        )

        return canvas

    # ── Sample Prediction Grid ────────────────────────────────────────────

    def save_sample_predictions(
        self,
        images_dir: str,
        labels_dir: str,
        num_samples: int = 8,
        attack: bool = True,
    ) -> str:
        """
        Generate a grid of sample predictions with dual-stream defense annotations.

        Returns:
            Path to saved grid image.
        """
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in os.listdir(images_dir) if Path(f).suffix.lower() in img_extensions
        ]
        np.random.shuffle(image_files)
        selected = image_files[:num_samples]

        panels = []
        for img_name in selected:
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, Path(img_name).stem + ".txt")

            image = cv2.imread(img_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            # Parse GT
            class_ids, bboxes = parse_yolo_labels(label_path, img_w, img_h)
            gt_boxes = []
            for cls_id, (cx, cy, w, h) in zip(class_ids, bboxes):
                gt_boxes.append((cls_id, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))

            # Optionally attack
            if attack and bboxes:
                image, _ = self.injector.apply_patch(image, bboxes, prob=1.0)

            # Run dual-stream defense
            result = self.defense.predict(image)

            # Draw predictions
            annotated = self.draw_predictions(image, result, ground_truth=gt_boxes)

            # Resize for grid
            display_h = 320
            scale = display_h / annotated.shape[0]
            display_w = int(annotated.shape[1] * scale)
            annotated = cv2.resize(annotated, (display_w, display_h))
            panels.append(annotated)

        if not panels:
            logger.warning("No samples to visualize")
            return ""

        # Arrange in a 2-column grid
        grid = _make_grid(panels, ncols=2)
        suffix = "attacked" if attack else "clean"
        output_path = os.path.join(self.output_dir, f"sample_predictions_{suffix}.png")
        cv2.imwrite(output_path, grid)
        logger.info(f"Sample predictions saved to: {output_path}")
        return output_path

    # ── Confusion Matrix Visualization ────────────────────────────────────

    def save_confusion_matrix(self, confusion: Dict[str, int]) -> str:
        """
        Generate and save a simple confusion matrix visualization.

        Args:
            confusion: Dict with keys TP, FP, FN.

        Returns:
            Path to saved image.
        """
        tp, fp, fn = confusion["TP"], confusion["FP"], confusion["FN"]

        # Create a simple visual
        canvas = np.ones((300, 450, 3), dtype=np.uint8) * 40  # Dark background

        # Title
        cv2.putText(
            canvas,
            "Detection Confusion Matrix",
            (50, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Grid
        y_start = 70
        cell_w, cell_h = 150, 80

        # Headers
        cv2.putText(
            canvas, "Predicted", (180, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        cv2.putText(
            canvas, "Pos", (175, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        cv2.putText(
            canvas, "Neg", (325, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        cv2.putText(
            canvas, "Actual", (20, y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

        # Cells
        cells = [
            (150, y_start + 50, tp, "TP", (0, 150, 0)),  # True Positive — green
            (300, y_start + 50, fn, "FN", (0, 0, 180)),  # False Negative — red
            (150, y_start + 130, fp, "FP", (0, 100, 200)),  # False Positive — orange
        ]
        for x, y, val, label, color in cells:
            cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), color, -1)
            cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), (255, 255, 255), 1)
            text = f"{label}: {val}"
            cv2.putText(
                canvas, text, (x + 20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # Precision / Recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        cv2.putText(
            canvas,
            f"Precision: {precision:.3f}",
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            canvas,
            f"Recall: {recall:.3f}",
            (250, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        cv2.imwrite(output_path, canvas)
        logger.info(f"Confusion matrix saved to: {output_path}")
        return output_path


# ── Helper Functions ──────────────────────────────────────────────────────


def _add_label(image: np.ndarray, text: str, color: Tuple[int, int, int]):
    """Add a colored label banner at the top of an image."""
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), color, -1)
    cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _draw_label_bg(
    image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]
):
    """Draw text with a filled background rectangle."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = origin
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(image, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _draw_dashed_rect(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 8,
):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]:
        _draw_dashed_line(image, edge[0], edge[1], color, thickness, dash_length)


def _draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 8,
):
    """Draw a dashed line between two points."""
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist < 1:
        return
    n_dashes = int(dist / dash_length)
    for i in range(0, n_dashes, 2):
        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * i / n_dashes),
            int(pt1[1] + (pt2[1] - pt1[1]) * i / n_dashes),
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * min(i + 1, n_dashes) / n_dashes),
            int(pt1[1] + (pt2[1] - pt1[1]) * min(i + 1, n_dashes) / n_dashes),
        )
        cv2.line(image, start, end, color, thickness)


def _draw_adversarial_banner(image: np.ndarray, score: float):
    """Draw a red warning banner on an image flagged as adversarial."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(
        image,
        f"ADVERSARIAL DETECTED (score: {score:.3f})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def _make_grid(panels: List[np.ndarray], ncols: int = 2, pad: int = 4) -> np.ndarray:
    """Arrange panels in a grid with padding."""
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Ensure all panels have the same width
    max_w = max(p.shape[1] for p in panels)
    max_h = max(p.shape[0] for p in panels)

    rows = []
    for i in range(0, len(panels), ncols):
        row_panels = panels[i : i + ncols]
        # Pad each panel to max dimensions
        padded = []
        for p in row_panels:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[: p.shape[0], : p.shape[1]] = p
            padded.append(canvas)
        # Pad row if incomplete
        while len(padded) < ncols:
            padded.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
        row = np.hstack(padded)
        rows.append(row)

    grid = np.vstack(rows)
    return grid
