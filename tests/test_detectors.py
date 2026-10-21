"""
test_detectors.py — Detector adapter contract (no torch needed).
"""

import numpy as np
import pytest

from MODEL.detectors import (
    ONNXDetector,
    RFDETRDetector,
    UltralyticsDetector,
    as_detector,
    infer_backend,
    production_weights,
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


class _FakeRFDETRModelWithLabels:
    """RF-DETR with the extra no-object logit slot, as a fine-tuned model has."""

    class_names = ["aircraft", "tank"]

    def __init__(self):
        self.seen = None

    def predict(self, image, threshold=0.5):
        self.seen = (image.copy(), threshold)

        class _Dets:
            xyxy = np.array([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 8.0, 8.0]])
            confidence = np.array([0.8, 0.4])
            class_id = np.array([1, 2])  # 2 == num_classes == the no-object slot
            data = {"class_name": np.array(["tank", "__background__"], dtype=object)}

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

    def test_uses_rfdetr_class_name_instead_of_inventing_one(self):
        """The no-object slot must not be reported under a class that does not exist."""
        det = RFDETRDetector(_FakeRFDETRModelWithLabels())
        out = det.predict(np.zeros((64, 64, 3), np.uint8), conf=0.3)
        # The background query is not an object, so it is not a detection.
        assert [d["class_name"] for d in out] == ["tank"]
        assert [d["class_id"] for d in out] == [1]

    def test_falls_back_to_class_names_when_rfdetr_gives_no_labels(self):
        det = RFDETRDetector(_FakeRFDETRModel())
        out = det.predict(np.zeros((64, 64, 3), np.uint8), conf=0.3)
        assert [d["class_name"] for d in out] == ["tank"]


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

    def test_falls_back_to_onnx_when_only_it_is_on_disk(self, tmp_path, monkeypatch):
        import MODEL.detectors as det

        monkeypatch.setattr(det, "REPO_ROOT", tmp_path)
        reg = tmp_path / "registry.yaml"
        reg.write_text(
            "models:\n  - {id: new, status: production, weights: w/best.pth, onnx: e/best.onnx}\n"
        )
        (tmp_path / "e").mkdir()
        (tmp_path / "e" / "best.onnx").write_bytes(b"x")
        assert det.production_weights(reg) == tmp_path / "e" / "best.onnx"
        (tmp_path / "w").mkdir()
        (tmp_path / "w" / "best.pth").write_bytes(b"x")
        assert det.production_weights(reg) == tmp_path / "w" / "best.pth"  # .pth preferred

    def test_model_entry_by_id_or_production(self, tmp_path):
        from MODEL.detectors import model_entry

        reg = tmp_path / "registry.yaml"
        reg.write_text(
            "models:\n"
            "  - {id: old, status: superseded, weights: a.pth}\n"
            "  - {id: new, status: production, weights: b.pth}\n"
        )
        assert model_entry(None, reg)["id"] == "new"
        assert model_entry("old", reg)["id"] == "old"
        assert model_entry("missing", reg) is None


class _FakeIO:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


class _FakeSession:
    """Mimics an onnxruntime session for an RF-DETR graph with 3 queries, 2 classes (+1 slot)."""

    def __init__(self, boxes, logits):
        self._outputs = [boxes, logits]

    def get_inputs(self):
        return [_FakeIO("input", [1, 3, 64, 64])]

    def get_outputs(self):
        return [_FakeIO("dets"), _FakeIO("labels")]

    def run(self, _names, feed):
        assert feed["input"].shape == (1, 3, 64, 64) and feed["input"].dtype == np.float32
        return self._outputs


class TestONNXDetector:
    def test_postprocess_matches_rfdetr_semantics(self):
        boxes = np.array(
            [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]], np.float32
        )
        logits = np.full((1, 3, 3), -10.0, np.float32)
        logits[0, 0, 1] = 3.0  # query 0 -> class 1, p ~ 0.95
        logits[0, 1, 0] = 0.0  # query 1 -> class 0, p = 0.5
        logits[0, 2, 2] = 5.0  # query 2 -> slot 2 (not a named class) -> dropped
        det = ONNXDetector(_FakeSession(boxes, logits), {0: "aircraft", 1: "tank"})
        out = det.predict(np.zeros((100, 200, 3), np.uint8), conf=0.4)

        assert [d["class_name"] for d in out] == ["tank", "aircraft"]  # sorted by score
        assert out[0]["bbox"] == pytest.approx(
            [80.0, 30.0, 120.0, 70.0]
        )  # cx,cy,w,h * (w=200, h=100)
        assert out[0]["confidence"] == pytest.approx(1 / (1 + np.exp(-3.0)))
        assert det.predict(np.zeros((100, 200, 3), np.uint8), conf=0.6)[0]["class_name"] == "tank"
        assert len(det.predict(np.zeros((100, 200, 3), np.uint8), conf=0.6)) == 1

    def test_onnx_suffix_selects_onnx_backend(self):
        assert infer_backend("model.onnx") == "onnx"
