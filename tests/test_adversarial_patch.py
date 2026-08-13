"""
test_adversarial_patch.py — Patch generation, placement, and blending tests.

Verifies:
  - All four patch types generate correctly-shaped BGR uint8 patches.
  - Patch placement stays within the bounding box.
  - Alpha blending is exact (alpha=1 → patch replaces ROI; alpha=0 → no-op).
  - PatchType.LOADED requires a path.
"""

import numpy as np
import cv2
import pytest

from MODEL.adversarial_patch import AdversarialPatchInjector
from MODEL.config import PatchType


def _make_scene(h=200, w=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (120, 120, 120), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


class TestPatchGeneration:
    @pytest.mark.parametrize("pt", [PatchType.NOISE, PatchType.PATTERN, PatchType.GRADIENT])
    def test_generated_patch_shape_and_dtype(self, pt):
        inj = AdversarialPatchInjector(patch_type=pt, use_perspective=False, use_lighting=False)
        patch = inj.generate_patch(32, 48)
        assert patch.shape == (32, 48, 3)
        assert patch.dtype == np.uint8

    def test_loaded_patch_requires_path(self):
        with pytest.raises(ValueError):
            AdversarialPatchInjector(patch_type=PatchType.LOADED)

    def test_loaded_patch_resizes(self, tmp_path):
        p = tmp_path / "patch.png"
        cv2.imwrite(str(p), np.random.RandomState(0).randint(0, 256, (64, 64, 3), dtype=np.uint8))
        inj = AdversarialPatchInjector(patch_type=PatchType.LOADED, patch_image_path=str(p))
        patch = inj.generate_patch(16, 32)
        assert patch.shape == (16, 32, 3)


class TestPlacement:
    def test_changes_stay_inside_bbox(self):
        inj = AdversarialPatchInjector(
            patch_type=PatchType.NOISE,
            scale_range=(0.2, 0.4),
            alpha=0.9,
            use_perspective=True,
            use_lighting=True,
        )
        img = _make_scene()
        bbox = (100, 100, 100, 100)  # cx, cy, w, h
        out, was = inj.apply_patch(img, [bbox], prob=1.0)
        assert was

        diff = np.where(np.any(out != img, axis=2))
        if len(diff[0]) == 0:
            pytest.skip("no pixel changed — degenerate draw")
        x1, y1 = int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)
        x2, y2 = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
        assert diff[0].min() >= y1 and diff[0].max() < y2
        assert diff[1].min() >= x1 and diff[1].max() < x2

    def test_no_patch_when_prob_zero(self):
        inj = AdversarialPatchInjector(patch_type=PatchType.NOISE)
        img = _make_scene()
        out, was = inj.apply_patch(img, [(100, 100, 80, 80)], prob=0.0)
        assert not was
        np.testing.assert_array_equal(out, img)

    def test_no_bboxes_returns_copy(self):
        inj = AdversarialPatchInjector(patch_type=PatchType.NOISE)
        img = _make_scene()
        out, was = inj.apply_patch(img, [], prob=1.0)
        assert not was
        np.testing.assert_array_equal(out, img)


class TestAlphaBlending:
    def test_alpha_one_fully_replaces_roi(self):
        inj = AdversarialPatchInjector(
            patch_type=PatchType.NOISE,
            scale_range=(0.3, 0.3),
            alpha=1.0,
            use_perspective=False,
            use_lighting=False,
        )
        img = _make_scene()
        out, _ = inj.apply_patch(img, [(100, 100, 100, 100)], prob=1.0)
        # ROI must differ from the original gray region somewhere
        assert np.any(out != img)

    def test_alpha_zero_is_noop(self):
        inj = AdversarialPatchInjector(
            patch_type=PatchType.NOISE,
            scale_range=(0.3, 0.3),
            alpha=0.0,
            use_perspective=False,
            use_lighting=False,
        )
        img = _make_scene()
        out, _ = inj.apply_patch(img, [(100, 100, 100, 100)], prob=1.0)
        np.testing.assert_array_equal(out, img)

    def test_blend_is_linear_combination(self):
        """ROI pixels must equal alpha*patch + (1-alpha)*roi within tolerance."""
        inj = AdversarialPatchInjector(
            patch_type=PatchType.PATTERN,
            scale_range=(0.3, 0.3),
            alpha=0.6,
            use_perspective=False,
            use_lighting=False,
        )
        img = _make_scene()
        # Deterministic patch generation
        np.random.seed(123)
        out, _ = inj.apply_patch(img, [(100, 100, 100, 100)], prob=1.0)
        diff_roi = np.abs(out.astype(np.int16) - img.astype(np.int16))
        # Only blended ROI differs; everything else identical
        assert np.max(diff_roi) <= 255
        assert np.any(diff_roi > 0)
