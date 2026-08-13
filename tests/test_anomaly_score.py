"""
test_anomaly_score.py — FFT anomaly scoring unit tests.

Verifies the Phase 1 claim that clean (smooth) images score LOW and
adversarially-patched images score HIGH, plus edge-case handling.
"""

import numpy as np
import cv2
import pytest

from MODEL.fft_scorer import compute_fft_anomaly_score, compute_region_scores


def _smooth(rng, h=160, w=160):
    return cv2.GaussianBlur(rng.randint(0, 255, (h, w, 3), dtype=np.uint8), (31, 31), 0)


class TestCleanVsPatched:
    """Core discriminator property: patches raise the high-freq score."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_patch_scores_higher_than_clean(self, seed):
        rng = np.random.RandomState(seed)
        clean = _smooth(rng)
        clean_score = compute_fft_anomaly_score(clean, 0.25)

        patched = clean.copy()
        ps = np.random.RandomState(99).randint(30, 60)
        x = np.random.RandomState(1).randint(0, clean.shape[1] - ps)
        y = np.random.RandomState(2).randint(0, clean.shape[0] - ps)
        patched[y : y + ps, x : x + ps] = np.random.RandomState(3).randint(
            0, 256, (ps, ps, 3), dtype=np.uint8
        )
        patched_score = compute_fft_anomaly_score(patched, 0.25)

        assert patched_score > clean_score

    def test_detection_rate_high(self):
        """Most (>=8/10) patched images should exceed all clean scores."""
        rng = np.random.RandomState(10)
        clean_scores, patched_scores = [], []
        for _ in range(10):
            clean = _smooth(rng)
            clean_scores.append(compute_fft_anomaly_score(clean, 0.25))
            patched = clean.copy()
            ps = 50
            x = rng.randint(0, clean.shape[1] - ps)
            y = rng.randint(0, clean.shape[0] - ps)
            patched[y : y + ps, x : x + ps] = rng.randint(0, 256, (ps, ps, 3), dtype=np.uint8)
            patched_scores.append(compute_fft_anomaly_score(patched, 0.25))

        threshold = max(clean_scores)
        hits = sum(s > threshold for s in patched_scores)
        assert hits >= 8


class TestEdgeCases:
    def test_empty_image_returns_zero(self):
        assert compute_fft_anomaly_score(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
        assert compute_fft_anomaly_score(None) == 0.0

    def test_tiny_image_returns_zero(self):
        assert compute_fft_anomaly_score(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0

    def test_constant_image_scores_zero(self):
        const = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert compute_fft_anomaly_score(const) == 0.0

    def test_grayscale_input_accepted(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        assert compute_fft_anomaly_score(gray) == 0.0


class TestRegionScores:
    def test_tiny_box_unscored(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [(10, 10, 20, 20), (5, 5, 8, 8)]
        scores = compute_region_scores(img, boxes, min_size=16)
        assert scores[0] is None
        assert scores[1] is None

    def test_out_of_bounds_box_clipped(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        scores = compute_region_scores(img, [(-50, -50, 200, 200)], min_size=16)
        assert scores[0] is not None

    def test_returns_parallel_list(self):
        img = np.random.RandomState(0).randint(0, 255, (100, 100, 3), dtype=np.uint8)
        boxes = [(10, 10, 60, 60), (20, 20, 80, 80)]
        scores = compute_region_scores(img, boxes, min_size=16)
        assert len(scores) == len(boxes)
