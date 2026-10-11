"""
detectors.py — One interface over the object detectors KAAL can run.

Every backend returns plain detection dicts so the dual-stream defense, the
backend server and the benchmark scripts never touch framework-specific
result objects:

    {"bbox": [x1, y1, x2, y2], "confidence": float,
     "class_id": int, "class_name": str}

Backends
  yolo    Ultralytics YOLO (.pt weights) — YOLOv8 / YOLO11 / YOLO26.
  rfdetr  Roboflow RF-DETR (ICLR 2026, Apache-2.0) — a real-time DETR with a
          DINOv2 backbone. NMS-free, and on COCO / RF100-VL more accurate than
          YOLO at equal latency. Loads ``checkpoint_best_total.pth`` files
          written by ``python -m MODEL.train_detector --arch rfdetr-*``.

RF-DETR is the production detector (Docs/DECISIONS.md, ADR-001); YOLO is an
optional backend. ``production_weights()`` resolves the weights of the model
marked ``status: production`` in ``MODEL/datasets_registry.yaml``.

Heavy frameworks are imported lazily so model-free code (FFT scoring,
calibration, tests) never pays for torch.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = Path(__file__).resolve().parent / "datasets_registry.yaml"

RFDETR_SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


class Detector(Protocol):
    names: Dict[int, str]

    def predict(self, image: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        """Detect objects in a BGR uint8 image."""
        ...


class UltralyticsDetector:
    """Wraps an Ultralytics model object (or anything with its result API)."""

    def __init__(self, model: Any, iou: float = 0.45, agnostic_nms: bool = True) -> None:
        self.model = model
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.names: Dict[int, str] = dict(getattr(model, "names", {}) or {})

    @classmethod
    def from_weights(cls, weights: str, **kwargs: Any) -> "UltralyticsDetector":
        from ultralytics import YOLO

        return cls(YOLO(weights), **kwargs)

    def predict(self, image: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=self.iou,
            agnostic_nms=self.agnostic_nms,  # merge part-boxes across class labels
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = getattr(result, "names", None) or self.names
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu())
                detections.append(
                    {
                        "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                        "confidence": float(boxes.conf[i].cpu()),
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                    }
                )
        return detections


class RFDETRDetector:
    """Wraps a Roboflow RF-DETR model (``rfdetr`` package)."""

    def __init__(self, model: Any, optimize: bool = False) -> None:
        self.model = model
        self.names: Dict[int, str] = dict(enumerate(model.class_names))
        if optimize:
            # Traces / compiles the graph for faster repeated inference.
            model.optimize_for_inference()

    @classmethod
    def from_weights(
        cls, weights: Optional[str] = None, size: str = "small", optimize: bool = False
    ) -> "RFDETRDetector":
        import rfdetr

        if weights:
            model = rfdetr.RFDETR.from_checkpoint(weights)
        else:  # COCO-pretrained starter weights
            model = getattr(rfdetr, RFDETR_SIZES[size])()
        return cls(model, optimize=optimize)

    def predict(self, image: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RF-DETR expects RGB
        dets = self.model.predict(rgb, threshold=conf)
        out = []
        for box, score, cls_id in zip(dets.xyxy, dets.confidence, dets.class_id):
            cls_id = int(cls_id)
            out.append(
                {
                    "bbox": [float(v) for v in box],
                    "confidence": float(score),
                    "class_id": cls_id,
                    "class_name": self.names.get(cls_id, str(cls_id)),
                }
            )
        return out


def production_weights(registry: Path = REGISTRY) -> Optional[Path]:
    """Weights of the registry model with ``status: production``, if they exist on disk."""
    try:
        models = (yaml.safe_load(registry.read_text()) or {}).get("models", [])
    except (OSError, yaml.YAMLError):
        return None
    for model in models:
        if model.get("status") == "production" and model.get("weights"):
            path = REPO_ROOT / model["weights"]
            return path if path.exists() else None
    return None


def infer_backend(weights: Optional[str]) -> str:
    """Guess the backend from a weights path (``.pt`` -> yolo, ``.pth/.ckpt`` -> rfdetr)."""
    suffix = Path(weights).suffix.lower() if weights else ""
    return "rfdetr" if suffix in (".pth", ".ckpt") else "yolo"


def load_detector(
    backend: Optional[str] = None,
    weights: Optional[str] = None,
    rfdetr_size: str = "small",
    optimize: bool = False,
) -> Detector:
    """Build a detector. ``backend`` is "yolo", "rfdetr" or None (inferred from weights)."""
    backend = (backend or infer_backend(weights)).lower()
    if backend == "yolo":
        if not weights:
            raise ValueError("YOLO backend needs a weights path (e.g. MODEL/yolov8m.pt)")
        return UltralyticsDetector.from_weights(weights)
    if backend == "rfdetr":
        return RFDETRDetector.from_weights(weights, size=rfdetr_size, optimize=optimize)
    raise ValueError(f"Unknown detector backend: {backend!r} (expected 'yolo' or 'rfdetr')")


def as_detector(model: Any) -> Detector:
    """Accept either a ``Detector`` or a raw Ultralytics-style model object."""
    if isinstance(model, (UltralyticsDetector, RFDETRDetector)):
        return model
    if hasattr(model, "predict") and not hasattr(model, "class_names"):
        return UltralyticsDetector(model)
    if hasattr(model, "class_names"):  # bare rfdetr model
        return RFDETRDetector(model)
    raise TypeError(f"Unsupported detector object: {type(model).__name__}")
