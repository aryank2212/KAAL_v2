"""
dual_stream_defense.py — FFT-based dual-stream adversarial defense.

Stream 1: Object detection (YOLO or RF-DETR, see ``MODEL/detectors.py``).
Stream 2: Frequency-domain anomaly detection using 2D FFT.
Stream 3 (optional, video): temporal-consistency cue that catches
          low-frequency perturbations the FFT cannot see.

Fusion: If the high-frequency anomaly score exceeds a threshold, the image
(or an individual detection, in per-region mode) is flagged as adversarially
perturbed and predictions are annotated with a warning. When a temporal cue
is attached, a frame is also flagged if its residual against the learned
background exceeds the cue's threshold.

Per-region mode (Phase 1): each detection is scored on its own bbox crop so
individual objects are flagged rather than entire frames.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .fft_scorer import DEFAULT_FFT_THRESHOLD, compute_fft_anomaly_score
from .calibration import load_calibrated_threshold
from .detectors import as_detector
from .temporal_scorer import StreamingTemporalCue


class DualStreamDefense:
    """
    Inference pipeline combining object detection with FFT-based adversarial
    patch detection (and an optional temporal cue for video streams).
    """

    def __init__(
        self,
        model: Any,
        fft_threshold: Optional[float] = None,
        high_freq_ratio: float = 0.25,
        conf_threshold: float = 0.25,
        region_mode: bool = True,
        region_min_size: int = 16,
        overlap_iou: float = 0.5,
        calibration_profile: Optional[str] = None,
        temporal_cue: Optional[StreamingTemporalCue] = None,
    ):
        """
        Args:
            model:               Stream 1 detector: a ``MODEL.detectors`` detector
                                 (YOLO / RF-DETR) or a raw Ultralytics model.
            fft_threshold:       Anomaly score above which a region/image is
                                 flagged adversarial. None → use calibration
                                 profile (if given) or DEFAULT_FFT_THRESHOLD.
            high_freq_ratio:     Fraction of the frequency spectrum considered
                                 "high-frequency".
            conf_threshold:      Confidence threshold for detector predictions.
            region_mode:         Score each detection on its own bbox crop
                                 (per-object flags) instead of whole frames only.
            region_min_size:     Skip FFT on detections smaller than this many
                                 pixels (width or height) — too noisy to score.
            overlap_iou:         Detections whose IoU with an already-scored box
                                 exceeds this reuse its score (dedupe overlaps).
            calibration_profile: Path to a JSON calibration profile whose
                                 threshold replaces the hardcoded default.
            temporal_cue:        Optional streaming temporal cue. Only meaningful
                                 when ``predict`` is fed consecutive frames of one
                                 (static) scene; call ``reset_temporal`` on cuts.
        """
        self.model = model
        self.detector = as_detector(model)
        self.temporal_cue = temporal_cue
        self.high_freq_ratio = high_freq_ratio
        self.conf_threshold = conf_threshold
        self.region_mode = region_mode
        self.region_min_size = region_min_size
        self.overlap_iou = overlap_iou

        if fft_threshold is None:
            fft_threshold = load_calibrated_threshold(
                profile_path=calibration_profile,
                fallback=DEFAULT_FFT_THRESHOLD,
                high_freq_ratio=high_freq_ratio,
            )
        self.fft_threshold = fft_threshold

    # ── Stream 1: Object detection ────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run the Stream 1 detector on a single BGR image."""
        return self.detector.predict(image, conf=self.conf_threshold)

    # Backwards-compatible name used by the evaluator / benchmark harnesses.
    yolo_predict = detect

    def reset_temporal(self) -> None:
        """Forget the temporal background (call on scene cuts / camera moves)."""
        if self.temporal_cue is not None:
            self.temporal_cue.reset()

    # ── Stream 2: FFT Anomaly Detection ───────────────────────────────────

    def compute_anomaly_score(self, image: np.ndarray) -> float:
        """
        Compute a high-frequency anomaly score using 2D FFT (whole image).

        Delegates to the standalone scorer so the calibrator and ablation
        harness measure exactly what the defense uses at inference time.
        """
        return compute_fft_anomaly_score(image, self.high_freq_ratio)

    def compute_anomaly_score_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute anomaly score for a specific region (bounding box).

        Args:
            image: BGR uint8 image.
            bbox:  (x1, y1, x2, y2) pixel coordinates.

        Returns:
            Regional anomaly score.
        """
        return self.compute_anomaly_score(self._crop_bbox(image, bbox))

    @staticmethod
    def _crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Clip a bbox to the image and extract the crop (empty → zeros)."""
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 - x1 < 1 or y2 - y1 < 1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return image[y1:y2, x1:x2]

    # ── Fusion Logic ──────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Full dual-stream prediction with adversarial detection.

        Args:
            image: BGR uint8 image.

        Returns:
            Dict with keys:
              - detections: list of detection dicts, each augmented with
                            `anomaly_score` (float or None for unscored) and
                            `is_adversarial` (bool).
              - anomaly_score: whole-image FFT score.
              - is_adversarial: True if the image OR any object is flagged.
              - objects_flagged: number of individually-flagged detections.
              - temporal: temporal-cue result dict (``None`` without a cue).
              - defense_action: str describing what the defense did.
        """
        # Stream 1: object detection
        detections = self.detect(image)

        # Stream 2: whole-image FFT score
        anomaly_score = self.compute_anomaly_score(image)

        # Per-region scoring of individual detections
        objects_flagged = 0
        if self.region_mode and detections:
            self._score_detections(image, detections)

            for det in detections:
                if det.get("is_adversarial"):
                    objects_flagged += 1

        # Stream 3: temporal consistency (video only)
        temporal = self.temporal_cue.observe(image) if self.temporal_cue is not None else None
        temporal_flag = bool(temporal and temporal["flagged"])

        is_adversarial = anomaly_score > self.fft_threshold or objects_flagged > 0 or temporal_flag

        if is_adversarial:
            temporal_note = (
                f", temporal={temporal['score']:.4f} > {temporal['threshold']:.4f}"
                if temporal_flag and temporal is not None
                else ""
            )
            defense_action = (
                f"⚠ ADVERSARIAL DETECTED (score={anomaly_score:.3f}, "
                f"threshold={self.fft_threshold:.3f}, objects_flagged={objects_flagged}"
                f"{temporal_note}). Predictions may be unreliable."
            )
        else:
            defense_action = (
                f"✓ Clean image (score={anomaly_score:.3f} ≤ "
                f"threshold={self.fft_threshold:.3f}). "
                f"Trusting detector predictions."
            )

        return {
            "detections": detections,
            "anomaly_score": anomaly_score,
            "is_adversarial": is_adversarial,
            "objects_flagged": objects_flagged,
            "temporal": temporal,
            "defense_action": defense_action,
        }

    def _score_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """
        Score each detection on its own bbox crop, handling edge cases:
          - Tiny boxes (< region_min_size px) are marked unscored (None, False).
          - Overlapping boxes reuse the score of the highest-confidence box
            they overlap (IoU >= overlap_iou) to avoid double-flagging.
        """
        # Sort by confidence (descending) so strong boxes are scored first
        ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        scored_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []

        for det in ordered:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            w = x2 - x1
            h = y2 - y1

            if w < self.region_min_size or h < self.region_min_size:
                det["anomaly_score"] = None
                det["is_adversarial"] = False
                det["region_scored"] = False
                continue

            # Reuse score from an overlapping already-scored box
            reused = None
            for (bx1, by1, bx2, by2), score in scored_boxes:
                if _iou((x1, y1, x2, y2), (bx1, by1, bx2, by2)) >= self.overlap_iou:
                    reused = score
                    break

            if reused is not None:
                det["anomaly_score"] = reused
                det["is_adversarial"] = reused > self.fft_threshold
                det["region_scored"] = True
            else:
                score = self.compute_anomaly_score_region(image, (x1, y1, x2, y2))
                det["anomaly_score"] = score
                det["is_adversarial"] = score > self.fft_threshold
                det["region_scored"] = True
                scored_boxes.append(((x1, y1, x2, y2), score))

    # ── Batch Inference ───────────────────────────────────────────────────

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Run dual-stream prediction on a batch of images."""
        return [self.predict(img) for img in images]


def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-10)
