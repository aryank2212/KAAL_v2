"""
test_calibration.py — Adaptive threshold calibration unit tests.

Verifies score collection, percentile/sigma threshold derivation, JSON
profile round-trips, and the calibrated-threshold loader fallbacks.
"""

import pytest

from MODEL.fft_scorer import DEFAULT_FFT_THRESHOLD
from MODEL.calibration import (
    FFTCalibrator,
    CalibrationProfile,
    load_calibrated_threshold,
)


class TestThresholdComputation:
    def test_percentile_method(self):
        scores = [float(i) for i in range(1, 101)]  # 1..100
        thr = FFTCalibrator.compute_threshold(scores, "percentile", 0.99)
        assert thr == pytest.approx(99.01)

    def test_sigma_method(self):
        import numpy as np

        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        thr = FFTCalibrator.compute_threshold(scores, "sigma", sigma=1.0)
        # Calibration uses the population std of the collected distribution
        expected = np.mean(scores) + 1.0 * np.std(scores)
        assert thr == pytest.approx(expected)

    def test_empty_scores_falls_back(self):
        assert FFTCalibrator.compute_threshold([], "percentile") == DEFAULT_FFT_THRESHOLD


class TestProfileBuild:
    def test_collect_scores_and_build(self, clean_image_dir):
        cal = FFTCalibrator(environment_id="cam01")
        scores = cal.collect_scores(clean_image_dir, n=12)
        assert len(scores) == 12
        profile = cal.build_profile(scores, "percentile", 0.99)
        assert profile.count == 12
        assert profile.p99 >= profile.p95 >= profile.p50
        assert profile.threshold >= profile.p99 - 1e-9
        assert profile.environment_id == "cam01"

    def test_one_shot_calibrate_from_dir(self, clean_image_dir):
        cal = FFTCalibrator(environment_id="cam02")
        profile = cal.calibrate_from_dir(clean_image_dir, n=6, method="sigma", sigma=3.0)
        assert profile.count == 6
        assert profile.threshold > profile.mean


class TestSerialization:
    def test_save_load_round_trip(self, tmp_path):
        prof = CalibrationProfile(
            environment_id="camA",
            scores=[0.1, 0.2, 0.3],
            threshold=0.28,
            method="percentile",
            count=3,
            mean=0.2,
            std=0.0816,
            p50=0.2,
            p95=0.29,
            p99=0.298,
            p999=0.2998,
        )
        path = str(tmp_path / "profile.json")
        FFTCalibrator.save_profile(prof, path)
        loaded = FFTCalibrator.load_profile(path)
        assert loaded.environment_id == "camA"
        assert loaded.scores == pytest.approx([0.1, 0.2, 0.3])
        assert loaded.threshold == pytest.approx(0.28)


class TestLoader:
    def test_none_path_returns_fallback(self):
        assert load_calibrated_threshold(None, 0.42) == 0.42

    def test_missing_file_returns_fallback(self):
        assert load_calibrated_threshold("/does/not/exist.json", 0.35) == 0.35

    def test_loads_profile_threshold(self, tmp_path):
        prof = CalibrationProfile(environment_id="camB", threshold=0.175, method="percentile")
        path = str(tmp_path / "p.json")
        FFTCalibrator.save_profile(prof, path)
        assert load_calibrated_threshold(path, 0.3) == pytest.approx(0.175)

    def test_corrupt_file_returns_fallback(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{not json")
        assert load_calibrated_threshold(path, 0.3) == 0.3


def test_collect_box_scores_scores_each_gt_box(tmp_path):
    import cv2
    import numpy as np

    from MODEL.calibration import FFTCalibrator

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.RandomState(0)
    img = cv2.GaussianBlur(rng.randint(0, 255, (200, 200, 3), dtype=np.uint8), (15, 15), 0)
    cv2.imwrite(str(images / "a.png"), img)
    # two scoreable boxes + one too small to score
    (labels / "a.txt").write_text("0 0.3 0.3 0.3 0.3\n1 0.7 0.7 0.3 0.3\n0 0.5 0.5 0.02 0.02\n")
    scores = FFTCalibrator().collect_box_scores(str(images), str(labels))
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
