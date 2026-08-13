"""
test_temporal_scorer.py — Unit tests for MODEL/temporal_scorer.py (Phase 3/4).
"""

import cv2
import numpy as np
import pytest

from MODEL.temporal_scorer import TemporalConsistencyDetector
from MODEL.failure_analysis import _smooth_gradient


def _clean_frames(n=10, seed=0):
    rng = np.random.RandomState(seed)
    frames = []
    base = _smooth_gradient(rng).astype(np.float32)
    for i in range(n):
        frame = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _smooth_offset(shape, seed=0, sigma=9, amp=20):
    """A smooth, low-frequency perturbation (the FFT-blind attack)."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, amp, shape[:2]).astype(np.float32)
    noise = np.stack([noise] * 3, axis=-1)
    return cv2.GaussianBlur(noise, (sigma, sigma), 0).astype(np.float32)


class TestScoring:
    def test_scores_bounded(self):
        det = TemporalConsistencyDetector()
        frames = _clean_frames(5)
        det.calibrate(frames)
        s = det.score(frames[0])
        assert 0.0 <= s <= 1.0

    def test_not_initialized_returns_zero(self):
        det = TemporalConsistencyDetector()
        assert det.score(_clean_frames(1)[0]) == 0.0
        assert not det.initialized

    def test_clean_vs_attack_residual_gap(self):
        """A persistent low-freq perturbation must lift the residual above clean."""
        clean = _clean_frames(12)
        det = TemporalConsistencyDetector()
        det.calibrate(clean)
        clean_scores = [det.score_and_update(f) for f in clean]

        base = clean[-1].astype(np.float32)
        offset = _smooth_offset(base.shape)
        attacked = [
            np.clip(
                base + offset + np.random.RandomState(i).normal(0, 2, base.shape), 0, 255
            ).astype(np.uint8)
            for i in range(6)
        ]
        attack_scores = [det.score_and_update(f) for f in attacked]
        assert np.mean(attack_scores) > np.mean(clean_scores)


class TestCalibration:
    def test_threshold_sane(self):
        det = TemporalConsistencyDetector()
        frames = _clean_frames(8)
        t = det.calibrate(frames)
        assert 0.0 < t < 1.0
        assert det.initialized

    def test_clean_fpr_low(self):
        frames = _clean_frames(20)
        det = TemporalConsistencyDetector()
        t = det.calibrate(frames)
        rate = det.detection_rate(frames, t)
        assert rate < 0.1

    def test_detects_low_freq_attack(self):
        """Low-frequency attacks are exactly what the FFT misses — the temporal cue should catch them."""
        clean = _clean_frames(20)
        det = TemporalConsistencyDetector()
        t = det.calibrate(clean)

        base = clean[-1].astype(np.float32)
        offset = _smooth_offset(base.shape, seed=1)
        attacked = [
            np.clip(
                base + offset + np.random.RandomState(i).normal(0, 2, base.shape), 0, 255
            ).astype(np.uint8)
            for i in range(10)
        ]
        rate = det.detection_rate(attacked, t)
        assert rate >= 0.5


class TestValidation:
    def test_bad_alpha_rejected(self):
        with pytest.raises(ValueError):
            TemporalConsistencyDetector(alpha=1.0)

    def test_bad_grid_rejected(self):
        with pytest.raises(ValueError):
            TemporalConsistencyDetector(grid=2)

    def test_reset_forgets_background(self):
        det = TemporalConsistencyDetector()
        det.calibrate(_clean_frames(4))
        assert det.initialized
        det.reset()
        assert not det.initialized
        assert det.score(_clean_frames(1)[0]) == 0.0
