"""
fft_scorer.py — Standalone 2D-FFT high-frequency anomaly scoring.

Kept independent of the detection model so it can be used by the
calibrator, the ablation harness, and the dual-stream defense without
pulling in Ultralytics / torch.

The score is the fraction of (AC) spectral *power* that lies in the
high-frequency band. Natural images follow a roughly 1/f^2 power law, so
almost all of their energy sits at low frequencies; adversarial patches
inject sharp, high-frequency structure, so a high ratio is an adversarial
signal.

Design notes (why it is computed this way):
  * Frequencies are measured in cycles/pixel (``np.fft.fftfreq``) on both
    axes, so the high-frequency band is the same physical band for any crop
    shape. The previous version sized a circular mask from ``min(h, w)`` in
    index space, which made elongated boxes score ~0.9 regardless of content.
  * The crop is mean-subtracted and Hann-windowed before the FFT so the
    implicit periodic tiling at the crop border does not leak a spurious
    high-frequency cross into the spectrum.
  * Power (|F|^2) is used instead of ``log1p(|F|)``. The log compressed the
    spectrum so much that the ratio tracked the *pixel area* of the band
    (white noise and a smooth image scored within 0.03 of each other).
"""

from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np

# Fallback anomaly threshold used when no calibration profile is supplied.
# ~p93 of clean ground-truth boxes on kaal_v3 val (5.4% false flags, 29% of
# patched boxes flagged on test; MODEL/RESULTS.md §0). Always prefer a value
# calibrated for the actual camera: ``python -m MODEL.calibration --labels``.
DEFAULT_FFT_THRESHOLD = 0.08

# Smallest side (pixels) that still yields a meaningful spectrum.
MIN_FFT_SIDE = 8


@lru_cache(maxsize=128)
def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


@lru_cache(maxsize=128)
def _band_weights(h: int, w: int, high_freq_ratio: float) -> tuple:
    """Boolean high-band mask + rfft column weights for an ``h x w`` input."""
    fy = np.fft.fftfreq(h)[:, None]  # cycles/pixel, [-0.5, 0.5)
    fx = np.fft.rfftfreq(w)[None, :]  # cycles/pixel, [0, 0.5]
    radius = np.sqrt(fy**2 + fx**2)
    cutoff = 0.5 * (1.0 - high_freq_ratio)
    high = radius > cutoff

    # rfft2 stores only half the (conjugate-symmetric) spectrum: interior
    # columns stand for two bins, the DC and (even-w) Nyquist columns for one.
    weights = np.full(fx.shape, 2.0, dtype=np.float64)
    weights[0, 0] = 1.0
    if w % 2 == 0:
        weights[0, -1] = 1.0
    return high, weights


def compute_fft_anomaly_score(
    image: Optional[np.ndarray],
    high_freq_ratio: float = 0.25,
) -> float:
    """
    Compute a high-frequency anomaly score using 2D FFT.

    Args:
        image:           BGR uint8 image (grayscale is auto-converted).
        high_freq_ratio: Width of the high-frequency band as a fraction of the
                         Nyquist range: frequencies above
                         ``0.5 * (1 - high_freq_ratio)`` cycles/pixel count as
                         "high". Larger values widen the band.

    Returns:
        Anomaly score in [0, 1]: share of AC power in the high band.
        Higher = more likely adversarial.
    """
    if image is None or image.size == 0:
        return 0.0
    if not 0.0 < high_freq_ratio < 1.0:
        raise ValueError(f"high_freq_ratio must be in (0, 1), got {high_freq_ratio}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape
    if min(h, w) < MIN_FFT_SIDE:
        return 0.0  # Region too small for a meaningful FFT

    signal = (gray - gray.mean()) * _hann2d(h, w)
    power = np.abs(np.fft.rfft2(signal)) ** 2
    high, weights = _band_weights(h, w, float(high_freq_ratio))

    weighted = power * weights
    weighted[0, 0] = 0.0  # DC carries no texture information
    total = float(weighted.sum())
    if total < 1e-6:
        return 0.0  # Constant region

    return float(weighted[high].sum() / total)


def compute_region_scores(
    image: np.ndarray,
    boxes: list,
    high_freq_ratio: float = 0.25,
    min_size: int = 16,
) -> list:
    """
    Compute per-region FFT anomaly scores for a list of boxes.

    Args:
        image:           BGR uint8 image.
        boxes:           List of (x1, y1, x2, y2) pixel-coordinate boxes.
        high_freq_ratio: FFT high-frequency band fraction.
        min_size:        Minimum box width AND height in pixels to score.
                         Boxes smaller than this return None (unscored).

    Returns:
        Parallel list of scores; None marks an unscored (too-small) box.
    """
    scores: List[Optional[float]] = []
    h_img, w_img = image.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 - x1 < min_size or y2 - y1 < min_size:
            scores.append(None)
            continue

        region = image[y1:y2, x1:x2]
        scores.append(compute_fft_anomaly_score(region, high_freq_ratio))

    return scores
