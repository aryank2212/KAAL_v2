"""
test_augmentations.py — Augmentation function unit tests.

Verifies:
  - Noise stays within the valid uint8 range (no wraparound regression).
  - Blur does not crash on small images.
  - All attack types preserve shape/dtype.
  - Low-frequency perturbation has lower HF energy than Gaussian noise.
"""

import numpy as np
import cv2
import pytest

from MODEL.augmentations import (
    add_patch,
    add_noise,
    blur,
    occlude,
    add_low_freq_noise,
    apply_random_attack,
    apply_all_attacks,
)
from MODEL.fft_scorer import compute_fft_anomaly_score


def _img(h=160, w=160):
    return cv2.GaussianBlur(
        np.random.RandomState(0).randint(0, 255, (h, w, 3), dtype=np.uint8), (15, 15), 0
    )


class TestAddNoise:
    def test_in_valid_range(self):
        out = add_noise(_img())
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255
        assert out.shape == _img().shape

    def test_changes_pixels(self):
        img = _img()
        assert np.any(add_noise(img) != img)


class TestBlur:
    def test_normal_image(self):
        out = blur(_img())
        assert out.shape == _img().shape and out.dtype == np.uint8

    def test_small_image_no_crash(self):
        out = blur(_img(h=8, w=8))
        assert out.shape == (8, 8, 3)


class TestShapePreservation:
    @pytest.mark.parametrize("fn", [add_patch, add_noise, occlude, add_low_freq_noise])
    def test_all_attacks_preserve_shape(self, fn):
        img = _img()
        out = fn(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_apply_all_attacks_combined(self):
        img = _img()
        out, name = apply_all_attacks(img)
        assert out.shape == img.shape
        assert name == "combined"


class TestAttackDispatch:
    @pytest.mark.parametrize(
        "key,expected",
        [
            ("patch", "patch"),
            ("noise", "noise+blur"),
            ("occlude", "occlusion"),
            ("lowfreq", "low-freq"),
        ],
    )
    def test_apply_random_attack_selects_requested(self, key, expected):
        out, name = apply_random_attack(_img(), attack_types=key)
        assert name == expected
        assert out.shape == _img().shape

    def test_apply_random_attack_all(self):
        out, name = apply_random_attack(_img(), attack_types="all")
        assert name in ("patch", "noise+blur", "occlusion", "low-freq")


class TestLowFreq:
    def test_low_freq_less_anomalous_than_noise(self):
        """Phase 1 failure-case test: smooth perturbation should evade FFT."""
        rng = np.random.RandomState(5)
        for _ in range(5):
            img = cv2.GaussianBlur(rng.randint(0, 255, (120, 120, 3), dtype=np.uint8), (21, 21), 0)
            s_clean = compute_fft_anomaly_score(img, 0.25)
            s_noise = compute_fft_anomaly_score(add_noise(img), 0.25)
            s_low = compute_fft_anomaly_score(add_low_freq_noise(img), 0.25)
            # Gaussian noise raises the HF ratio more than the smooth field
            assert (s_noise - s_clean) > (s_low - s_clean)
