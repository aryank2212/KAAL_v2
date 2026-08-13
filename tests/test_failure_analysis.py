"""
test_failure_analysis.py — Unit tests for MODEL/failure_analysis.py (Phase 3).
"""

import numpy as np
import pytest

import MODEL.failure_analysis as fa
from MODEL.failure_analysis import (
    run_failure_analysis,
    generate_textures,
    _smooth_gradient,
    _foliage,
    _urban,
    _rain,
)


class TestTextureGenerators:
    @pytest.mark.parametrize("gen", [_smooth_gradient, _foliage, _urban, _rain])
    def test_shapes_and_dtype(self, gen):
        rng = np.random.RandomState(0)
        img = gen(rng)
        assert img.shape == (fa.TEXTURE_SIZE, fa.TEXTURE_SIZE, 3)
        assert img.dtype == np.uint8

    def test_deterministic(self):
        a = generate_textures(5, seed=3)
        b = generate_textures(5, seed=3)
        assert set(a.keys()) == {"sky", "foliage", "urban", "rain"}
        for name in a:
            for x, y in zip(a[name], b[name]):
                np.testing.assert_array_equal(x, y)


class TestRunFailureAnalysis:
    def test_report_structure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        report = run_failure_analysis(n=8, seed=0)
        assert set(report["fpr_by_texture"].keys()) == {"sky", "foliage", "urban", "rain"}
        for fpr in report["fpr_by_texture"].values():
            assert 0.0 <= fpr <= 1.0
        for rate in report["attack_detection_rate"].values():
            assert 0.0 <= rate <= 1.0
        assert report["calibrated_threshold"] > 0
        assert set(report["temporal_cue"]) == {
            "threshold",
            "clean_fpr",
            "low_freq_noise",
            "universal_low_freq",
        }
        assert 0.0 <= report["temporal_cue"]["clean_fpr"] <= 1.0
        assert (tmp_path / "runs" / "failure_analysis" / "report.json").exists()
        assert (tmp_path / "runs" / "failure_analysis" / "report.md").exists()

    def test_region_fpr_measured(self, tmp_path, monkeypatch):
        """ROI-crop scoring must be reported alongside whole-frame FPR."""
        monkeypatch.chdir(tmp_path)
        report = run_failure_analysis(n=8, seed=0, region=64)
        assert set(report["fpr_by_texture_region"].keys()) == {
            "sky",
            "foliage",
            "urban",
            "rain",
        }
        for fpr in report["fpr_by_texture_region"].values():
            assert 0.0 <= fpr <= 1.0

    def test_low_freq_attack_underperforms_high_freq(self, tmp_path, monkeypatch):
        """The whole point: smoothed perturbations evade the FFT detector."""
        monkeypatch.chdir(tmp_path)
        report = run_failure_analysis(n=16, seed=1)
        rates = report["attack_detection_rate"]
        assert rates["low_freq_noise"] < rates["gaussian_noise_high_freq"]
        assert "failure_modes"  # low-freq failure should be documented
