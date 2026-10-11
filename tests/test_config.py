"""
test_config.py — Config validation unit tests.

Verifies sensible defaults and that validation catches bad values.
"""

import pytest

from MODEL.config import Config, DefenseMode, PatchType
from MODEL.fft_scorer import DEFAULT_FFT_THRESHOLD


class TestDefaults:
    def test_default_values(self):
        c = Config()
        assert c.patch_scale_range == (0.05, 0.5)
        assert c.patch_alpha == 0.8
        assert c.fft_threshold is None
        assert c.fft_default_threshold == DEFAULT_FFT_THRESHOLD
        assert c.fft_high_freq_ratio == 0.25
        assert c.fft_region_mode is True
        assert c.fft_region_min_size == 16
        assert c.patch_perspective is True
        assert c.patch_lighting is True
        assert c.defense_mode == DefenseMode.DUAL_STREAM

    def test_enum_values(self):
        assert [d.value for d in DefenseMode] == [
            "none",
            "adversarial_training",
            "dual_stream",
        ]
        assert [p.value for p in PatchType] == [
            "noise",
            "pattern",
            "loaded",
            "gradient",
        ]


class TestValidation:
    def test_patch_prob_out_of_range(self):
        with pytest.raises(AssertionError):
            Config(patch_prob=1.5)
        with pytest.raises(AssertionError):
            Config(patch_prob=-0.1)

    def test_patch_alpha_zero(self):
        with pytest.raises(AssertionError):
            Config(patch_alpha=0.0)

    def test_fft_threshold_invalid(self):
        with pytest.raises(AssertionError):
            Config(fft_threshold=1.2)
        with pytest.raises(AssertionError):
            Config(fft_threshold=0.0)

    def test_fft_threshold_none_allowed(self):
        assert Config(fft_threshold=None).fft_threshold is None

    def test_inverted_scale_range(self):
        with pytest.raises(AssertionError):
            Config(patch_scale_range=(0.5, 0.05))

    def test_bad_calibration_method(self):
        with pytest.raises(AssertionError):
            Config(calibration_method="median")

    def test_bad_defense_mode(self):
        with pytest.raises(ValueError):
            Config(defense_mode=DefenseMode("nope"))  # invalid enum value


class TestValidOverrides:
    def test_custom_values_accepted(self):
        c = Config(
            fft_threshold=0.4,
            fft_high_freq_ratio=0.3,
            patch_scale_range=(0.05, 0.5),
            calibration_method="sigma",
            calibration_sigma=2.0,
        )
        assert c.fft_threshold == 0.4
        assert c.fft_high_freq_ratio == 0.3
        assert c.calibration_method == "sigma"
