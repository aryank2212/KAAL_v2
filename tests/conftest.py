"""
conftest.py — Shared fixtures and environment setup for the KAAL test suite.

If real `ultralytics` is not installed, a minimal stub module is registered in
`sys.modules` so model-free unit tests can import MODEL modules that reference
`from ultralytics import YOLO` (the defense never calls it unless a model is
passed). Integration tests supply their own FakeYOLO.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── ultralytics stub (used only when the real package is absent) ─────────


def _ensure_ultralytics_stub():
    """Register a minimal `ultralytics` module if the real one is missing."""
    if "ultralytics" in sys.modules:
        return
    try:
        import ultralytics  # noqa: F401

        return
    except ImportError:
        pass

    class _YOLO:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
            self.names = {}

        def predict(self, *args, **kwargs):
            raise NotImplementedError("Stub YOLO has no real inference — use a FakeYOLO fixture.")

    module = types.ModuleType("ultralytics")
    module.YOLO = _YOLO
    sys.modules["ultralytics"] = module


_ensure_ultralytics_stub()


# ── Shared fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def smooth_image():
    """A smooth, low-frequency BGR image (mimics clean natural imagery)."""
    rng = np.random.RandomState(0)
    return cv2_gaussian_blur(rng.randint(0, 255, (160, 160, 3), dtype=np.uint8), ksize=31)


@pytest.fixture
def clean_image_dir(tmp_path):
    """Directory of N smooth clean images for calibration/ablation tests."""
    rng = np.random.RandomState(7)
    d = tmp_path / "clean"
    d.mkdir()
    for i in range(12):
        img = cv2_gaussian_blur(rng.randint(0, 255, (120, 120, 3), dtype=np.uint8), ksize=25)
        import cv2

        cv2.imwrite(str(d / f"img{i:02d}.png"), img)
    return str(d)


def cv2_gaussian_blur(img, ksize):
    """Gaussian blur helper that imports cv2 lazily."""
    import cv2

    return cv2.GaussianBlur(img, (ksize, ksize), 0)
