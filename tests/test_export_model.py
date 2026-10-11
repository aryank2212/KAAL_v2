"""
test_export_model.py — The export parity check compares like with like.
"""

import cv2
import numpy as np
import pytest

from MODEL.export_model import parity, sha256

pytest.importorskip("pycocotools")


class _GroundTruthDetector:
    """Returns the ground-truth box (perfect) or a shifted copy (degraded)."""

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift

    def predict(self, image, conf=0.25):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = 0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h
        d = self.shift * w
        return [
            {
                "bbox": [x1 + d, y1, x2 + d, y2],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "aircraft",
            }
        ]


def _dataset(tmp_path, n=6):
    (tmp_path / "data.yaml").write_text("names:\n  0: aircraft\n  1: tank\n")
    for sub in ("images", "labels"):
        (tmp_path / "val" / sub).mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(
            str(tmp_path / "val" / "images" / f"img_{i}.jpg"), np.zeros((80, 120, 3), np.uint8)
        )
        (tmp_path / "val" / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    return tmp_path


def test_identical_detectors_have_zero_delta(tmp_path):
    data = _dataset(tmp_path)
    out = parity(_GroundTruthDetector(), _GroundTruthDetector(), data, "val", 0)
    assert out["n_images"] == 6
    assert out["pytorch"]["mAP50_95"] == pytest.approx(1.0)
    assert out["map50_95_delta"] == pytest.approx(0.0)


def test_degraded_export_shows_negative_delta(tmp_path):
    data = _dataset(tmp_path)
    out = parity(_GroundTruthDetector(), _GroundTruthDetector(shift=0.2), data, "val", 0)
    assert out["map50_95_delta"] < -0.1


def test_sha256_streams_whole_file(tmp_path):
    import hashlib

    f = tmp_path / "w.bin"
    payload = b"kaal" * 600_000  # > 1 MiB, so more than one read chunk
    f.write_bytes(payload)
    assert sha256(f) == hashlib.sha256(payload).hexdigest()
