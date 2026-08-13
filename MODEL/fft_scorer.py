"""
fft_scorer.py — Standalone 2D-FFT high-frequency anomaly scoring.

Kept independent of the detection model so it can be used by the
calibrator, the ablation harness, and the dual-stream defense without
pulling in Ultralytics / torch.

The score is the fraction of spectral energy in the high-frequency band.
Adversarial patches introduce sharp, high-frequency patterns that are
unusual in natural images, so a high ratio is a strong adversarial signal.
"""

import cv2
import numpy as np
from typing import List, Optional


def compute_fft_anomaly_score(
    image: np.ndarray,
    high_freq_ratio: float = 0.25,
) -> float:
    """
    Compute a high-frequency anomaly score using 2D FFT.

    Args:
        image:           BGR uint8 image (grayscale is auto-converted).
        high_freq_ratio: Fraction of the spectrum considered "high-frequency".
                         Larger values make the detector more sensitive to
                         mid-frequency content.

    Returns:
        Anomaly score in [0, 1]. Higher = more likely adversarial.
    """
    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    if min(gray.shape) < 8:
        return 0.0  # Region too small for a meaningful FFT

    # 2D FFT and shift zero-frequency to center
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Magnitude spectrum (log scale for numerical stability)
    magnitude = np.log1p(np.abs(f_shift))

    total_energy = np.sum(magnitude)
    if total_energy < 1e-10:
        return 0.0

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * (1 - high_freq_ratio) / 2)

    Y, X = np.ogrid[:h, :w]
    low_freq_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius**2

    low_freq_energy = np.sum(magnitude[low_freq_mask])
    high_freq_energy = total_energy - low_freq_energy

    return float(high_freq_energy / total_energy)


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
