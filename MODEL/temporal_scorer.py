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
from typing import Any, Dict, List, Optional, Sequence

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


class StreamingTemporalCue:
    """
    Online, *localized* temporal cue for live video from a static camera.

    The frame-mean residual of :class:`TemporalConsistencyDetector` dilutes a
    patch that covers a few percent of the frame to almost nothing, so the
    live cue works per grid cell instead:

    * **Per-cell residuals, illumination-normalized** — the signed residual
      against the EMA background has its median removed, so a global
      brightness change (auto-exposure, clouds) does not light up every cell,
      while a spatially varying perturbation still does.
    * **Warm-up calibration** — without a fixed threshold, the cell residuals
      of the first ``warmup_frames`` frames (assumed clean) set the cell
      threshold at their ``percentile`` times ``margin`` (floor ``min_threshold``).
    * **Persistence** — a frame is flagged only when some cell has stayed above
      threshold for ``persist_frames`` consecutive frames, so an object moving
      through the scene does not trigger it but a patch fixed on a target does.
    * **Masked background updates** — hot cells are not folded into the
      background (a persistent patch stays visible); the rest keeps adapting.
    * **Re-baselining** — a cell hot for ``max_flagged_frames`` is absorbed into
      the background (e.g. a vehicle that parked); if most of the frame is hot
      that long, the camera moved and the cue relearns from scratch.

    Limitation: models a *static or slowly changing* scene; disable it (or
    call :meth:`reset` often) on a moving camera. A flag means "something
    changed persistently at this spot" — an alert for review, not a verdict.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        warmup_frames: int = 30,
        percentile: float = 0.999,
        margin: float = 1.5,
        min_threshold: float = 0.01,
        persist_frames: int = 5,
        max_flagged_frames: int = 150,
        global_frac: float = 0.5,
        grid: int = 32,
        alpha: float = 0.05,
    ) -> None:
        if warmup_frames < 2:
            raise ValueError(f"warmup_frames must be >= 2, got {warmup_frames}")
        if persist_frames < 1:
            raise ValueError(f"persist_frames must be >= 1, got {persist_frames}")
        self.detector = TemporalConsistencyDetector(grid=grid, alpha=alpha)
        self.fixed_threshold = threshold
        self.threshold: Optional[float] = threshold
        self.warmup_frames = warmup_frames
        self.percentile = percentile
        self.margin = margin
        self.min_threshold = min_threshold
        self.persist_frames = persist_frames
        self.max_flagged_frames = max_flagged_frames
        self.global_frac = global_frac
        self._warmup: List[np.ndarray] = []
        self._hot_count = np.zeros((grid, grid), dtype=np.int32)
        self.rebaselines = 0

    def reset(self) -> None:
        """Forget the background and (unless fixed) the learned threshold."""
        self.detector.reset()
        self.threshold = self.fixed_threshold
        self._warmup = []
        self._hot_count[:] = 0

    @property
    def ready(self) -> bool:
        return self.detector.initialized and self.threshold is not None

    def _cell_residuals(self, cells: np.ndarray) -> np.ndarray:
        assert self.detector._background is not None
        signed = cells - self.detector._background
        signed -= np.median(signed)  # remove global illumination shift
        return np.abs(signed) / 255.0

    def _masked_update(self, cells: np.ndarray, hot: np.ndarray) -> None:
        bg = self.detector._background
        assert bg is not None
        blended = (1.0 - self.detector.alpha) * bg + self.detector.alpha * cells
        self.detector._background = np.where(hot, bg, blended).astype(np.float32)

    def observe(self, frame: np.ndarray) -> Dict[str, Any]:
        """Score one frame; returns ``{score, threshold, flagged, ready, hot_cells}``."""
        cells = self.detector._to_cells(frame)
        if not self.detector.initialized:
            self.detector._background = cells.astype(np.float32)
            return self._result(0.0, False, 0)

        residuals = self._cell_residuals(cells)
        score = float(residuals.max())
        if self.threshold is None:  # still calibrating
            self._warmup.append(residuals.ravel())
            self._masked_update(cells, np.zeros_like(residuals, dtype=bool))
            if len(self._warmup) >= self.warmup_frames:
                base = float(np.percentile(np.concatenate(self._warmup), self.percentile * 100.0))
                self.threshold = max(base * self.margin, self.min_threshold)
                self._warmup = []
            return self._result(score, False, 0)

        hot = residuals > self.threshold
        self._hot_count = np.where(hot, self._hot_count + 1, 0)
        stale = self._hot_count >= self.max_flagged_frames
        if stale.any():
            self.rebaselines += 1
            if stale.mean() >= self.global_frac:  # the whole view changed: relearn
                self.reset()
                self.detector._background = cells.astype(np.float32)
                return self._result(score, False, 0)
            # Absorb the long-lived local change (e.g. a vehicle that parked).
            bg = self.detector._background
            assert bg is not None
            self.detector._background = np.where(stale, cells, bg).astype(np.float32)
            self._hot_count[stale] = 0
            hot &= ~stale

        flagged = bool((self._hot_count >= self.persist_frames).any())
        self._masked_update(cells, hot)
        return self._result(score, flagged, int(hot.sum()))

    def _result(self, score: float, flagged: bool, hot_cells: int) -> Dict[str, Any]:
        return {
            "score": round(float(score), 5),
            "threshold": None if self.threshold is None else round(float(self.threshold), 5),
            "flagged": bool(flagged),
            "ready": self.ready,
            "hot_cells": hot_cells,
        }
