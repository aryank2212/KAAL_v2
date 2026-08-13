"""
ablation.py — Ablation study harness for the FFT dual-stream defense (Phase 1).

Sweeps the four key hyperparameters and records detection metrics:

  1. high_freq_ratio  →  how sensitive the FFT band definition is
  2. fft_threshold     →  precision / recall / F1 of the detector
  3. patch_scale_range →  detection rate vs patch size (attack strength)
  4. patch_alpha       →  detection rate vs patch opacity

Metrics are computed by planting patches on clean images and measuring
whether the defense flags them (TP) or not (FN), plus false alarms on
clean images (FP / TN). Curves are written to CSV + JSON + PNG so they can
become paper figures directly.
"""

import os
import json
import csv
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Sequence

from .config import PatchType
from .fft_scorer import compute_fft_anomaly_score
from .adversarial_patch import AdversarialPatchInjector, parse_yolo_labels

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Sweeps defense/attack hyperparameters and logs detection-rate curves.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str = "runs/ablation",
        model: Optional[object] = None,
        n_images: int = 60,
        seed: int = 42,
    ):
        """
        Args:
            images_dir: Validation images (clean).
            labels_dir: YOLO label files for those images (bboxes for patches).
            output_dir: Where CSV/JSON/PNG results are written.
            model:      Optional detector for gradient patches (not required).
            n_images:   Number of images to sample per sweep point.
            seed:       RNG seed for reproducibility.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.model = model
        self.n_images = n_images
        self.seed = seed
        os.makedirs(self.output_dir, exist_ok=True)

        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted(
            f for f in os.listdir(images_dir) if Path(f).suffix.lower() in img_extensions
        )[:n_images]
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

    # ── Data Preparation ──────────────────────────────────────────────────

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Load image + pixel-coordinate (cx, cy, w, h) bboxes for sample idx."""
        fname = self.image_files[idx]
        img = cv2.imread(os.path.join(self.images_dir, fname))
        if img is None:
            raise ValueError(f"Could not read {fname}")
        h, w = img.shape[:2]
        label_path = os.path.join(self.labels_dir, Path(fname).stem + ".txt")
        _, bboxes = parse_yolo_labels(label_path, w, h)
        return img, bboxes

    def _score_clean_and_patched(
        self,
        high_freq_ratio: float,
        patch_type: PatchType,
        scale_range: Tuple[float, float],
        alpha: float,
        use_perspective: bool,
        use_lighting: bool,
    ) -> Tuple[List[float], List[float]]:
        """
        Return (clean_scores, patched_scores) across the sampled images.

        A patched image counts as "detected" if its FFT score exceeds the
        sweep's threshold — the caller decides the threshold.
        """
        injector = AdversarialPatchInjector(
            patch_type=patch_type,
            scale_range=scale_range,
            alpha=alpha,
            model=self.model,
            use_perspective=use_perspective,
            use_lighting=use_lighting,
        )
        clean_scores, patched_scores = [], []
        for idx in range(len(self.image_files)):
            img, bboxes = self._load_sample(idx)
            clean_scores.append(compute_fft_anomaly_score(img, high_freq_ratio))
            if not bboxes:
                continue
            attacked, _ = injector.apply_patch(img.copy(), bboxes, prob=1.0)
            patched_scores.append(compute_fft_anomaly_score(attacked, high_freq_ratio))
        return clean_scores, patched_scores

    # ── Metric Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _detection_metrics(
        clean_scores: List[float], patched_scores: List[float], threshold: float
    ) -> Dict[str, Any]:
        """Precision / recall / F1 / TPR / FPR at a given threshold."""
        fp = sum(1 for s in clean_scores if s > threshold)
        tn = len(clean_scores) - fp
        tp = sum(1 for s in patched_scores if s > threshold)
        fn = len(patched_scores) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "threshold": float(threshold),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tpr": round(recall, 4),
            "fpr": round(fp / len(clean_scores) if clean_scores else 0.0, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    # ── Sweeps ────────────────────────────────────────────────────────────

    def sweep_high_freq_ratio(
        self,
        values: Sequence[float] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35),
        threshold: float = 0.3,
        scale_range: Tuple[float, float] = (0.1, 0.4),
        alpha: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """Sweep the FFT high-frequency band definition."""
        rows = []
        for ratio in values:
            clean, patched = self._score_clean_and_patched(
                high_freq_ratio=ratio,
                patch_type=PatchType.NOISE,
                scale_range=scale_range,
                alpha=alpha,
                use_perspective=False,
                use_lighting=False,
            )
            m = self._detection_metrics(clean, patched, threshold)
            m["param"] = "high_freq_ratio"
            m["value"] = ratio
            rows.append(m)
            logger.info("high_freq_ratio=%.2f -> %s", ratio, m)
        self._write_results("sweep_high_freq_ratio", rows, "high_freq_ratio")
        return rows

    def sweep_threshold(
        self,
        values: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
        high_freq_ratio: float = 0.25,
        scale_range: Tuple[float, float] = (0.1, 0.4),
        alpha: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """Sweep the anomaly threshold; report precision/recall/F1 at each."""
        clean, patched = self._score_clean_and_patched(
            high_freq_ratio=high_freq_ratio,
            patch_type=PatchType.NOISE,
            scale_range=scale_range,
            alpha=alpha,
            use_perspective=False,
            use_lighting=False,
        )
        rows = []
        for thr in values:
            m = self._detection_metrics(clean, patched, thr)
            m["param"] = "fft_threshold"
            m["value"] = thr
            rows.append(m)
            logger.info("fft_threshold=%.2f -> %s", thr, m)
        self._write_results("sweep_fft_threshold", rows, "fft_threshold")
        return rows

    def sweep_patch_scale_range(
        self,
        values: Sequence[Tuple[float, float]] = (
            (0.05, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
        ),
        high_freq_ratio: float = 0.25,
        threshold: float = 0.3,
        alpha: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """Sweep patch size (fraction of bbox); detection rate vs size."""
        rows = []
        for scale_range in values:
            clean, patched = self._score_clean_and_patched(
                high_freq_ratio=high_freq_ratio,
                patch_type=PatchType.NOISE,
                scale_range=scale_range,
                alpha=alpha,
                use_perspective=False,
                use_lighting=False,
            )
            m = self._detection_metrics(clean, patched, threshold)
            m["param"] = "patch_scale_range"
            m["value"] = list(scale_range)
            m["scale_max"] = scale_range[1]
            rows.append(m)
            logger.info("patch_scale_range=%s -> %s", scale_range, m)
        self._write_results("sweep_patch_scale_range", rows, "patch_scale_range")
        return rows

    def sweep_patch_alpha(
        self,
        values: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        high_freq_ratio: float = 0.25,
        threshold: float = 0.3,
        scale_range: Tuple[float, float] = (0.1, 0.4),
    ) -> List[Dict[str, Any]]:
        """Sweep patch opacity; detection rate vs opacity."""
        rows = []
        for alpha in values:
            clean, patched = self._score_clean_and_patched(
                high_freq_ratio=high_freq_ratio,
                patch_type=PatchType.NOISE,
                scale_range=scale_range,
                alpha=alpha,
                use_perspective=False,
                use_lighting=False,
            )
            m = self._detection_metrics(clean, patched, threshold)
            m["param"] = "patch_alpha"
            m["value"] = alpha
            rows.append(m)
            logger.info("patch_alpha=%.2f -> %s", alpha, m)
        self._write_results("sweep_patch_alpha", rows, "patch_alpha")
        return rows

    # ── Output ───────────────────────────────────────────────────────────

    def _write_results(self, name: str, rows: List[Dict], x_key: str) -> str:
        """Write CSV + JSON for one sweep; returns the CSV path."""
        if not rows:
            return ""
        base = os.path.join(self.output_dir, name)
        csv_path = base + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        json_path = base + ".json"
        with open(json_path, "w") as f:
            json.dump({"name": name, "x_key": x_key, "rows": rows}, f, indent=2)

        self._plot_curve(base + ".png", rows, x_key)
        logger.info("Sweep results: %s.csv / %s.json / %s.png", base, base, base)
        return csv_path

    @staticmethod
    def _plot_curve(path: str, rows: List[Dict], x_key: str) -> None:
        """Best-effort matplotlib figure; skipped if matplotlib is absent."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            logger.info("matplotlib not available — skipping curve plot %s", path)
            return

        x = [r["value"] for r in rows]
        plt.figure(figsize=(7, 5))
        for metric in ("precision", "recall", "f1", "tpr"):
            y = [r.get(metric, 0.0) for r in rows]
            plt.plot([str(v) for v in x], y, marker="o", label=metric)
        plt.xlabel(x_key)
        plt.ylabel("score")
        plt.title(f"Ablation sweep: {x_key}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 ablation sweeps for the FFT dual-stream defense."
    )
    parser.add_argument("--images", required=True, help="Clean validation images dir")
    parser.add_argument("--labels", required=True, help="YOLO labels dir")
    parser.add_argument("--output", default="runs/ablation", help="Output directory")
    parser.add_argument("--n-images", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    study = AblationStudy(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        n_images=args.n_images,
        seed=args.seed,
    )
    study.sweep_high_freq_ratio(threshold=args.threshold)
    study.sweep_threshold()
    study.sweep_patch_scale_range(threshold=args.threshold)
    study.sweep_patch_alpha(threshold=args.threshold)
    logger.info("Ablation study complete. Results in %s", args.output)


if __name__ == "__main__":
    main()
