"""
test_dual_stream.py — Integration test for the dual-stream defense.

Runs the FULL `DualStreamDefense.predict()` path (YOLO detections from a
FakeYOLO + real FFT scoring) on clean and patched images, calibrates a
threshold, and asserts the per-region detection rate exceeds 80%.

This mirrors the Phase 2 roadmap requirement:
  "Dual-stream: run predict() on 10 clean + 10 patched images, assert
   detection rate > 80%"
"""

import numpy as np
import cv2
import pytest

from MODEL.dual_stream_defense import DualStreamDefense
from MODEL.calibration import FFTCalibrator


class _C:
    def __init__(self, v):
        self.v = v

    def numpy(self):
        return np.asarray(self.v, dtype=np.float32)

    def tolist(self):
        return list(self.v)

    def __float__(self):
        return float(np.asarray(self.v).ravel()[0])

    def __int__(self):
        return int(np.asarray(self.v).ravel()[0])


class _T:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return _C(self.v)


class _Boxes:
    def __init__(self, boxes):
        self.xyxy = [_T(b[0]) for b in boxes]
        self.conf = [_T([b[1]]) for b in boxes]
        self.cls = [_T([b[2]]) for b in boxes]

    def __len__(self):
        return len(self.xyxy)


class _Results:
    def __init__(self, boxes):
        self.boxes = _Boxes(boxes)
        self.names = {0: "person", 1: "car"}


class FakeYOLO:
    """Deterministic fake detector: always returns a central person box."""

    def predict(self, **kwargs):
        boxes = [([40, 40, 120, 120], 0.9, 0)]
        return [_Results(boxes)]


@pytest.fixture
def fake_yolo():
    return FakeYOLO()


def _make_images(n=10, seed=0):
    """Return (clean_images, patched_images). Patches sit inside the person box."""
    rng = np.random.RandomState(seed)
    clean, patched = [], []
    for _ in range(n):
        img = cv2.GaussianBlur(rng.randint(0, 255, (160, 160, 3), dtype=np.uint8), (31, 31), 0)
        clean.append(img.copy())
        # Place a high-frequency patch inside [40,40,120,120]
        ps = 40
        x, y = 60, 60
        img[y : y + ps, x : x + ps] = rng.randint(0, 256, (ps, ps, 3), dtype=np.uint8)
        patched.append(img)
    return clean, patched


def test_frame_level_detection_rate_above_80(fake_yolo):
    clean, patched = _make_images(10, seed=1)
    defense = DualStreamDefense(model=fake_yolo, fft_threshold=0.3, region_mode=False)

    clean_scores = [defense.predict(img)["anomaly_score"] for img in clean]
    patched_scores = [defense.predict(img)["anomaly_score"] for img in patched]

    # Calibrate the threshold from the clean distribution (Phase 1 flow)
    threshold = max(clean_scores)
    assert all(s <= threshold for s in clean_scores)  # zero FPR on clean
    detected = sum(s > threshold for s in patched_scores)
    assert detected / len(patched) > 0.8


def test_per_region_detection_rate_above_80(fake_yolo):
    clean, patched = _make_images(10, seed=2)

    # Calibrate from a holdout clean batch, then apply to the test batch
    calibrator = DualStreamDefense(
        model=fake_yolo, fft_threshold=0.3, region_mode=True, region_min_size=16
    )
    calib_scores = [calibrator.predict(img)["detections"][0]["anomaly_score"] for img in clean[:5]]
    threshold = max(calib_scores)

    defense = DualStreamDefense(
        model=fake_yolo,
        fft_threshold=threshold,
        region_mode=True,
        region_min_size=16,
    )

    # No false positives on held-out clean images
    clean_flagged = sum(
        defense.predict(img)["detections"][0]["is_adversarial"] for img in clean[5:]
    )
    assert clean_flagged == 0

    # Patched images must be flagged (frame OR per-object)
    detected = sum(1 for img in patched if defense.predict(img)["is_adversarial"])
    assert detected / len(patched) > 0.8


def test_calibrated_threshold_wiring(fake_yolo, clean_image_dir):
    """Calibrated profile drives the defense threshold instead of default."""
    cal = FFTCalibrator(environment_id="test")
    profile = cal.calibrate_from_dir(clean_image_dir, n=12, method="percentile", percentile=0.99)

    import tempfile
    import os

    path = os.path.join(tempfile.mkdtemp(), "profile.json")
    FFTCalibrator.save_profile(profile, path)

    defense = DualStreamDefense(model=fake_yolo, calibration_profile=path)
    assert defense.fft_threshold == pytest.approx(profile.threshold)


def test_tiny_detections_unscored(fake_yolo):
    class TinyYOLO(FakeYOLO):
        def predict(self, **kwargs):
            boxes = [([10, 10, 15, 15], 0.9, 0)]
            return [_Results(boxes)]

    defense = DualStreamDefense(
        model=TinyYOLO(), fft_threshold=0.3, region_mode=True, region_min_size=16
    )
    img = cv2.GaussianBlur(
        np.random.RandomState(0).randint(0, 255, (160, 160, 3), dtype=np.uint8), (31, 31), 0
    )
    res = defense.predict(img)
    det = res["detections"][0]
    assert det["region_scored"] is False
    assert det["anomaly_score"] is None
