"""
export_model.py — Export the production RF-DETR for deployment and prove parity.

Writes the model in a deployment format via RF-DETR's exporter and, for
ONNX, a ``<model>.onnx.json`` sidecar (class names, input size, source
weights, SHA-256 of both files) that ``MODEL.detectors.ONNXDetector`` reads.
With ``--parity N`` the exported ONNX graph and the PyTorch checkpoint are
both scored on N images of a dataset split; the export fails (exit 1) if
mAP50:95 differs by more than ``--max-map-drop``.

    python -m MODEL.export_model                                  # production weights -> ONNX
    python -m MODEL.export_model --parity 200 --data MODEL/Datasets/kaal_v4
    python -m MODEL.export_model --format tensorrt                # on a machine with TensorRT

Formats are passed straight to ``RFDETR.export`` (onnx, tensorrt, openvino,
coreml, tflite, …); only ONNX gets the sidecar and the parity check here.
Serve the ONNX model with ``KAAL_MODEL_PATH=<file>.onnx`` (backend ``onnx``).
"""

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parity(
    torch_detector: Any, onnx_detector: Any, data: Path, split: str, n: int
) -> Dict[str, Any]:
    """Score both detectors on the same images with the same COCO maths."""
    from .compare_detectors import coco_map, imread, load_split, read_gt

    images, class_names = load_split(data, split, n)
    sizes, gts = [], []
    results: Dict[str, Any] = {}
    frames = []
    for p in images:
        img = imread(p)
        frames.append(img)
        sizes.append(img.shape[:2])
        gts.append(read_gt(data / split / "labels" / f"{p.stem}.txt", *img.shape[:2]))
    for name, det in (("pytorch", torch_detector), ("onnx", onnx_detector)):
        dets, lat = [], []
        for img in frames:
            t0 = time.perf_counter()
            dets.append(det.predict(img, conf=0.01))
            lat.append((time.perf_counter() - t0) * 1000.0)
        m = coco_map(gts, dets, sizes, class_names)
        results[name] = {
            "mAP50": m["mAP50"],
            "mAP50_95": m["mAP50_95"],
            "latency_ms": round(statistics.median(lat[5:] or lat), 2),
        }
    results["n_images"] = len(images)
    results["map50_95_delta"] = round(
        results["onnx"]["mAP50_95"] - results["pytorch"]["mAP50_95"], 4
    )
    return results


def main() -> None:
    from .detectors import ONNXDetector, RFDETRDetector, production_weights

    parser = argparse.ArgumentParser(description="Export the KAAL RF-DETR and check parity.")
    parser.add_argument(
        "--weights", default=None, help="RF-DETR checkpoint (default: registry production)"
    )
    parser.add_argument("--format", default="onnx")
    parser.add_argument(
        "--out", default=None, help="Output dir (default: MODEL/runs/export/<run name>)"
    )
    parser.add_argument(
        "--parity", type=int, default=0, help="Images for the ONNX parity check (0 = skip)"
    )
    parser.add_argument("--data", default=str(HERE / "Datasets" / "kaal_v4"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-map-drop", type=float, default=0.01)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    weights = Path(args.weights) if args.weights else production_weights()
    if weights is None or not weights.exists():
        parser.error(
            "no --weights given and no production weights on disk (MODEL/datasets_registry.yaml)"
        )
    out = Path(args.out) if args.out else HERE / "runs" / "export" / weights.parent.name
    out.mkdir(parents=True, exist_ok=True)

    torch_detector = RFDETRDetector.from_weights(str(weights))
    model = torch_detector.model
    exported = Path(model.export(output_dir=str(out), format=args.format, verbose=False))
    logger.info("Exported %s -> %s", weights, exported)
    if args.format != "onnx":
        return

    target = out / f"{weights.parent.name}.onnx"
    if exported != target:
        exported.replace(target)
    onnx_detector = ONNXDetector.from_weights(str(target))
    meta: Dict[str, Any] = {
        "class_names": list(model.class_names),
        "input_hw": list(onnx_detector.input_hw),
        "source_weights": str(weights),
        "source_sha256": sha256(weights),
        "onnx_sha256": sha256(target),
        "exporter": "rfdetr RFDETR.export(format='onnx')",
    }
    if args.parity:
        onnx_detector.names = dict(enumerate(meta["class_names"]))
        meta["parity"] = parity(
            torch_detector, onnx_detector, Path(args.data), args.split, args.parity
        )
        logger.info("Parity: %s", json.dumps(meta["parity"]))
    Path(f"{target}.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s and %s.json", target, target)
    if args.parity and abs(meta["parity"]["map50_95_delta"]) > args.max_map_drop:
        logger.error(
            "ONNX parity failed: |ΔmAP50:95| %.4f > %.4f",
            abs(meta["parity"]["map50_95_delta"]),
            args.max_map_drop,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
