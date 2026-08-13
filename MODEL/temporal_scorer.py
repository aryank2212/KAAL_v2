"""
temporal_scorer.py — Model-free temporal-consistency anomaly cue (Phase 3/4).

The FFT detector is blind to low-frequency perturbations because they add
almost no high-frequency spectral energy. This module provides a *second,
non-frequency cue*: it tracks a running exponential-moving-average (EMA)
background model of a scene and scores each new frame by its block-wise
residual against that model.

A persistent perturbation (e.g. `add_low_freq_noise` or a smoothed universal
patch) shifts the whole scene and keeps the residual high, even though the
spectral content looks clean. Natural per-frame sensor noise and slow drift,
by contrast, average out and produce a low residual.

Typical use (matches Phase 1 flow):

    detector = TemporalConsistencyDetector()
    threshold = detector.calibrate(clean_frames)     # percentile of clean residuals
    for frame in stream:
        score = detector.score(frame)                # 0..1 residual, vs background
        detector.update(frame)                       # fold frame into background
        flagged = score > threshold

Kept independent of the detector model so it can run before/alongside YOLO
without pulling in torch/Ultralytics.
"""

import cv2
import numpy as np
from typing import Sequence

from .calibration import FFTCalibrator


class TemporalConsistencyDetector:
    """EMA background model + block-wise residual scoring for one scene."""

    def __init__(
        self,
        grid: int = 32,
        alpha: float = 0.05,
    ) -> None:
        """
        Args:
            grid:  Downsample each frame to ``grid x grid`` cells before
                   differencing (block averaging -> robust to pixel noise and
                   tiny motion). Larger = more spatially sensitive.
            alpha: EMA update factor in [0, 1]. Smaller = slower background
                   adaptation -> persistent perturbations stay visible longer.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if grid < 4:
            raise ValueError(f"grid must be >= 4, got {grid}")
        self.grid = grid
        self.alpha = float(alpha)
        self._background: np.ndarray | None = None

    # ── low-level helpers ────────────────────────────────────────────────

    def _to_cells(self, image: np.ndarray) -> np.ndarray:
        """Grayscale float32 frame downsampled to ``grid x grid`` cells."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        small = cv2.resize(
            gray.astype(np.float32), (self.grid, self.grid), interpolation=cv2.INTER_AREA
        )
        return small

    # ── public API ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget the background model (e.g. when the scene changes)."""
        self._background = None

    @property
    def initialized(self) -> bool:
        return self._background is not None

    def score(self, image: np.ndarray) -> float:
        """
        Residual of ``image`` against the current background model, in [0, 1].

        Returns 0.0 when no background has been learned yet.
        """
        if self._background is None:
            return 0.0
        cells = self._to_cells(image)
        residual = float(np.mean(np.abs(cells - self._background)) / 255.0)
        return float(np.clip(residual, 0.0, 1.0))

    def update(self, image: np.ndarray) -> None:
        """Fold ``image`` into the EMA background model."""
        cells = self._to_cells(image)
        if self._background is None:
            self._background = cells.astype(np.float32)
        else:
            self._background = (1.0 - self.alpha) * self._background + self.alpha * cells

    def score_and_update(self, image: np.ndarray) -> float:
        """Score against the current model, then fold the frame in."""
        score = self.score(image)
        self.update(image)
        return score

    # ── calibration ──────────────────────────────────────────────────────

    def calibrate(
        self,
        clean_frames: Sequence[np.ndarray],
        percentile: float = 0.99,
    ) -> float:
        """
        Learn the background from ``clean_frames`` and return a threshold.

        Each clean frame is scored (residual vs. the model so far) and then
        folded in; the threshold is the ``percentile``-th percentile of those
        residuals, so a well-behaved clean scene rarely trips it.
        """
        scores = [self.score_and_update(frame) for frame in clean_frames]
        return float(FFTCalibrator.compute_threshold(scores, "percentile", percentile))

    def detection_rate(
        self,
        frames: Sequence[np.ndarray],
        threshold: float,
    ) -> float:
        """Fraction of ``frames`` whose residual exceeds ``threshold``."""
        if not frames:
            return 0.0
        return sum(1 for frame in frames if self.score_and_update(frame) > threshold) / len(frames)
