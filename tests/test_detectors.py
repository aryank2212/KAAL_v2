"""
test_detectors.py — Detector adapter contract (no torch needed).
"""

import numpy as np
import pytest

from MODEL.detectors import (
    production_weights,
    RFDETRDetector,
    UltralyticsDetector,
    as_detector,
    infer_backend,
)


class _FakeRFDETRModel:
    """Mimics rfdetr.RFDETR: RGB input, supervision-style Detections output."""

    class_names = ["aircraft", "tank"]

    def __init__(self):
        self.seen = None

    def predict(self, image, threshold=0.5):
        self.seen = (image.copy(), threshold)

        class _Dets:
            xyxy = np.array([[10.0, 20.0, 50.0, 60.0]])
            confidence = np.array([0.8])
            class_id = np.array([1])

        return _Dets()


class TestInferBackend:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("MODEL/yolov8m.pt", "yolo"),
            ("runs/train/x/checkpoint_best_total.pth", "rfdetr"),
            ("last.ckpt", "rfdetr"),
            (None, "yolo"),
        ],
    )
    def test_suffix_mapping(self, path, expected):
        assert infer_backend(path) == expected


class TestRFDETRDetector:
    def test_converts_bgr_to_rgb_and_normalizes_output(self):
        model = _FakeRFDETRModel()
        det = RFDETRDetector(model)
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[..., 0] = 255  # pure blue in BGR
        out = det.predict(bgr, conf=0.3)

        rgb_seen, threshold = model.seen
        assert threshold == 0.3
        assert rgb_seen[0, 0].tolist() == [0, 0, 255]  # blue moved to the last channel
        assert out == [
            {
                "bbox": [10.0, 20.0, 50.0, 60.0],
                "confidence": 0.8,
                "class_id": 1,
                "class_name": "tank",
            }
        ]

    def test_as_detector_wraps_bare_rfdetr_model(self):
        assert isinstance(as_detector(_FakeRFDETRModel()), RFDETRDetector)


class TestAsDetector:
    def test_wraps_ultralytics_like_object(self):
        class _Model:
            names = {0: "tank"}

            def predict(self, **kwargs):
                return []

        det = as_detector(_Model())
        assert isinstance(det, UltralyticsDetector)
        assert det.predict(np.zeros((8, 8, 3), dtype=np.uint8)) == []

    def test_passthrough_and_rejects_unknown(self):
        det = RFDETRDetector(_FakeRFDETRModel())
        assert as_detector(det) is det
        with pytest.raises(TypeError):
            as_detector(object())


class TestProductionWeights:
    def test_resolves_production_model_on_disk(self, tmp_path, monkeypatch):
        import MODEL.detectors as det

        monkeypatch.setattr(det, "REPO_ROOT", tmp_path)
        (tmp_path / "w").mkdir()
        (tmp_path / "w" / "best.pth").write_bytes(b"x")
        reg = tmp_path / "registry.yaml"
        reg.write_text(
            "models:\n"
            "  - {id: old, status: baseline, weights: w/old.pt}\n"
            "  - {id: new, status: production, weights: w/best.pth}\n"
        )
        assert det.production_weights(reg) == tmp_path / "w" / "best.pth"

    def test_missing_weights_or_registry_give_none(self, tmp_path):
        reg = tmp_path / "registry.yaml"
        reg.write_text("models:\n  - {id: new, status: production, weights: nope.pth}\n")
        assert production_weights(reg) is None
        assert production_weights(tmp_path / "absent.yaml") is None
