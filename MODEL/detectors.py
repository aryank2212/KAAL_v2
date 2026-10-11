"""
detectors.py — One interface over the object detectors KAAL can run.

Every backend returns plain detection dicts so the dual-stream defense, the
backend server and the benchmark scripts never touch framework-specific
result objects:

    {"bbox": [x1, y1, x2, y2], "confidence": float,
     "class_id": int, "class_name": str}

Backends
  onnx    RF-DETR exported with ``python -m MODEL.export_model`` (.onnx +
          .onnx.json class names), run by ONNX Runtime — no torch needed;
          mirrors RF-DETR's own pre/post-processing (parity-tested).
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


class ONNXDetector:
    """RF-DETR exported to ONNX, run with ONNX Runtime.

    Reproduces ``RFDETR.predict``: bilinear resize to the graph's input size,
    ImageNet normalisation, sigmoid over the class logits, top-``num_select``
    over (query, class) pairs, centre-size boxes scaled to the frame.
    """

    MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, session: Any, names: Dict[int, str], num_select: int = 300) -> None:
        self.session = session
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self.output_names = [o.name for o in session.get_outputs()]
        self.names = names
        self.num_select = num_select

    @classmethod
    def from_weights(cls, weights: str, providers: Optional[List[str]] = None) -> "ONNXDetector":
        import json

        import onnxruntime as ort

        meta_path = Path(f"{weights}.json")
        names: Dict[int, str] = {}
        if meta_path.exists():
            names = dict(enumerate(json.loads(meta_path.read_text())["class_names"]))
        session = ort.InferenceSession(
            weights, providers=providers or ort.get_available_providers()
        )
        return cls(session, names)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return ((x - self.MEANS) / self.STDS).transpose(2, 0, 1)[None].astype(np.float32)

    def postprocess(
        self, boxes: np.ndarray, logits: np.ndarray, image_hw: tuple, conf: float
    ) -> List[Dict[str, Any]]:
        """``boxes`` (Q, 4) normalised cx, cy, w, h; ``logits`` (Q, C)."""
        h0, w0 = image_hw
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        flat = prob.ravel()
        k = min(self.num_select, flat.size)
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top], kind="stable")]
        queries, labels, scores = top // prob.shape[1], top % prob.shape[1], flat[top]
        cx, cy, bw, bh = boxes[queries].T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        xyxy = np.clip(xyxy * np.array([w0, h0, w0, h0], dtype=np.float32), 0, [w0, h0, w0, h0])
        out = []
        for box, score, label in zip(xyxy, scores, labels):
            if score <= conf or (self.names and int(label) not in self.names):
                continue
            out.append(
                {
                    "bbox": [float(v) for v in box],
                    "confidence": float(score),
                    "class_id": int(label),
                    "class_name": self.names.get(int(label), str(int(label))),
                }
            )
        return out

    def predict(self, image: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        outputs = dict(
            zip(
                self.output_names, self.session.run(None, {self.input_name: self.preprocess(image)})
            )
        )
        boxes = outputs.get("dets", next(iter(outputs.values())))
        logits = outputs.get("labels", list(outputs.values())[-1])
        return self.postprocess(boxes[0], logits[0], image.shape[:2], conf)


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
    """Guess the backend from a weights path (.pth/.ckpt -> rfdetr, .onnx -> onnx, else yolo)."""
    suffix = Path(weights).suffix.lower() if weights else ""
    if suffix == ".onnx":
        return "onnx"
    return "rfdetr" if suffix in (".pth", ".ckpt") else "yolo"


def load_detector(
    backend: Optional[str] = None,
    weights: Optional[str] = None,
    rfdetr_size: str = "small",
    optimize: bool = False,
) -> Detector:
    """Build a detector. ``backend`` is "rfdetr", "onnx", "yolo" or None (inferred from weights)."""
    backend = (backend or infer_backend(weights)).lower()
    if backend == "yolo":
        if not weights:
            raise ValueError("YOLO backend needs a weights path (e.g. MODEL/yolov8m.pt)")
        return UltralyticsDetector.from_weights(weights)
    if backend == "rfdetr":
        return RFDETRDetector.from_weights(weights, size=rfdetr_size, optimize=optimize)
    if backend == "onnx":
        if not weights:
            raise ValueError(
                "ONNX backend needs an exported .onnx file (python -m MODEL.export_model)"
            )
        return ONNXDetector.from_weights(weights)
    raise ValueError(f"Unknown detector backend: {backend!r} (expected 'rfdetr', 'onnx' or 'yolo')")


def as_detector(model: Any) -> Detector:
    """Accept either a ``Detector`` or a raw Ultralytics-style model object."""
    if isinstance(model, (UltralyticsDetector, RFDETRDetector, ONNXDetector)):
        return model
    if hasattr(model, "predict") and not hasattr(model, "class_names"):
        return UltralyticsDetector(model)
    if hasattr(model, "class_names"):  # bare rfdetr model
        return RFDETRDetector(model)
    raise TypeError(f"Unsupported detector object: {type(model).__name__}")
