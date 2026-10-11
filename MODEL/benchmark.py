"""
benchmark.py — Comparison benchmark: attack x defense x (mAP, latency, detectability).

Phase 3 deliverable. Builds the roadmap comparison matrix:

    | Defense       | Clean mAP | Patch mAP | Physical mAP | Low-Freq mAP | Universal mAP | Latency (ms) |
    |---------------|-----------|-----------|--------------|--------------|---------------|--------------|
    | None          | ...       | ...       | ...          | ...          | ...           | ...          |
    | JPEG          | ...       | ...       | ...          | ...          | ...           | ...          |
    | Blur          | ...       | ...       | ...          | ...          | ...           | ...          |
    | Bit-depth     | ...       | ...       | ...          | ...          | ...           | ...          |
    | Feature sqz.  | ...       | ...       | ...          | ...          | ...           | ...          |
    | FFT (ours)    | ...       | ...       | ...          | ...          | ...           | ...          |
    | FFT + JPEG    | ...       | ...       | ...          | ...          | ...           | ...          |

Design notes
------------
- Preprocessing defenses operate on BGR uint8 numpy images and are
  model-free, so the harness runs without torch or a detector. They mirror
  the torch-tensor versions in `MODEL/defenses/preprocess.py` (which are the
  canonical in-training implementations); these numpy equivalents exist for
  benchmarking only.
- Attacks are numpy-based: random patch, physical-world patch
  (perspective + lighting), low-frequency perturbation, and an
  image-agnostic universal perturbation. PGD/gradient-optimized patches are
  covered by `AdversarialPatchInjector(PatchType.GRADIENT)` and the tensor
  `pgd_attack()` in `MODEL/attacks/`; the matrix here measures the
  model-free attack families so it runs in CI.
- Two metric families are produced:

  1. **FFT detectability** (model-free, always): for every (defense, attack)
     pair, the rate at which the FFT anomaly scorer still flags the image
     *after* the defense is applied, plus the clean false-positive rate.
     For preprocessing defenses a low rate means the defense neutralised the
     attack's frequency signature (good); for the "fft" rows it is the
     detector's own true/false-positive rate.

  2. **mAP matrix** (requires a detector): VOC-style mAP@0.5 per
     (defense, attack) plus median defense latency. The "fft" and "fft_jpeg"
     rows implement the dual-stream suppression policy: when the FFT flags a
     frame, detections on that frame are dropped (fail-safe alert) instead
     of trusting a tampered image.

- A detector is any object implementing
  `predict(image) -> List[Dict]` with keys `bbox` (xyxy), `confidence`,
  `class_id` — the same contract as `DualStreamDefense.yolo_predict`.
"""

import csv
import json
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence, Callable

import cv2
import numpy as np

from .augmentations import add_low_freq_noise
from .adversarial_patch import AdversarialPatchInjector, parse_yolo_labels
from .calibration import FFTCalibrator
from .config import PatchType
from .fft_scorer import compute_fft_anomaly_score

logger = logging.getLogger(__name__)

# ── Defense registry (numpy BGR uint8 -> BGR uint8) ──────────────────────


def _jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def _gaussian_blur(image: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    k = max(3, kernel | 1)  # odd, at least 3
    return cv2.GaussianBlur(image, (k, k), sigma)


def _bit_depth(image: np.ndarray, bits: int) -> np.ndarray:
    levels = max(2, min(8, bits))
    steps = float(2**levels - 1)
    quantized = np.round(image.astype(np.float32) * steps / 255.0) * (255.0 / steps)
    return quantized.astype(np.uint8)


def _feature_squeeze(image: np.ndarray, bits: int, kernel: int) -> np.ndarray:
    return cv2.medianBlur(_bit_depth(image, bits), max(3, kernel | 1))


DEFENSE_TRANSFORMS: Dict[str, Callable[..., np.ndarray]] = {
    "none": lambda image, **_: image,
    "jpeg": lambda image, jpeg_quality, **_: _jpeg_compress(image, jpeg_quality),
    "blur": lambda image, blur_kernel, blur_sigma, **_: _gaussian_blur(
        image, blur_kernel, blur_sigma
    ),
    "bitdepth": lambda image, bit_depth, **_: _bit_depth(image, bit_depth),
    "featuresqueeze": lambda image, bit_depth, blur_kernel, **_: _feature_squeeze(
        image, bit_depth, blur_kernel
    ),
}

DEFAULT_DEFENSES: Sequence[str] = (
    "none",
    "jpeg",
    "blur",
    "bitdepth",
    "featuresqueeze",
    "fft",
    "fft_jpeg",
)

# Defenses that are detection policies rather than image transforms.
POLICY_DEFENSES: Sequence[str] = ("fft", "fft_jpeg")


def apply_defense(
    defense: str,
    image: np.ndarray,
    jpeg_quality: int = 30,
    blur_kernel: int = 5,
    blur_sigma: float = 1.0,
    bit_depth: int = 4,
) -> np.ndarray:
    """
    Apply a preprocessing defense to a BGR uint8 image.

    Policy defenses ("fft", "fft_jpeg") are handled by the runner, not here;
    they return the image unchanged.
    """
    if defense in POLICY_DEFENSES:
        return image
    if defense not in DEFENSE_TRANSFORMS:
        raise ValueError(f"Unknown defense: {defense!r}")
    return DEFENSE_TRANSFORMS[defense](
        image,
        jpeg_quality=jpeg_quality,
        blur_kernel=blur_kernel,
        blur_sigma=blur_sigma,
        bit_depth=bit_depth,
    )


def measure_defense_latency(
    defense: str,
    image: np.ndarray,
    n_repeats: int = 20,
    **params: Any,
) -> float:
    """Median wall-clock latency (ms) of a preprocessing defense."""
    if defense in POLICY_DEFENSES:
        return 0.0
    elapsed = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        apply_defense(defense, image, **params)
        elapsed.append((time.perf_counter() - start) * 1e3)
    return float(statistics.median(elapsed))


# ── Attacks (numpy BGR uint8 -> BGR uint8) ───────────────────────────────


def generate_universal_perturbation(
    images: Sequence[np.ndarray],
    epsilon: int = 40,
    seed: int = 0,
    low_freq: bool = True,
) -> np.ndarray:
    """
    Build one image-agnostic universal adversarial perturbation.

    The default (``low_freq=True``) is a smoothed random field clipped to
    ``+-epsilon``. Smoothing keeps spectral energy in the low-frequency band,
    which is exactly the FFT detector's failure case (documented in
    `docs/failure-analysis.md`). ``low_freq=False`` yields a sharp noise
    pattern that the FFT detector catches easily.
    """
    if not images:
        raise ValueError("generate_universal_perturbation requires at least one image")
    h, w = images[0].shape[:2]
    rng = np.random.RandomState(seed)
    pert: np.ndarray = rng.uniform(-epsilon, epsilon, (h, w, 3)).astype(np.float32)
    if low_freq:
        pert = cv2.GaussianBlur(pert, (0, 0), sigmaX=max(h, w) / 25.0)
        pert = cv2.resize(
            cv2.resize(pert, (max(4, w // 8), max(4, h // 8)), interpolation=cv2.INTER_CUBIC),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
    peak = float(np.abs(pert).max())
    if peak > 0:
        pert = pert / peak * epsilon
    return pert


def apply_attack(
    attack: str,
    image: np.ndarray,
    rng: np.random.RandomState,
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    universal: Optional[np.ndarray] = None,
    scale_range: Tuple[float, float] = (0.1, 0.4),
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Apply a numpy attack to a BGR uint8 image.

    Args:
        attack: One of "none", "patch", "physical", "lowfreq", "universal".
        bboxes: Object boxes as (cx, cy, w, h) pixel tuples (patch attacks).
        universal: Precomputed perturbation from `generate_universal_perturbation`.
    """
    if attack in ("none", ""):
        return image
    if attack == "lowfreq":
        return add_low_freq_noise(image)
    if attack == "universal":
        if universal is None:
            raise ValueError("'universal' attack requires a precomputed perturbation")
        pert = np.resize(universal, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + pert, 0, 255).astype(np.uint8)
    if attack in ("patch", "physical"):
        physical = attack == "physical"
        injector = AdversarialPatchInjector(
            patch_type=PatchType.NOISE,
            scale_range=scale_range,
            alpha=alpha,
            use_perspective=physical,
            use_lighting=physical,
        )
        if bboxes:
            np.random.seed(int(rng.randint(0, 2**32 - 1)))
            attacked, _ = injector.apply_patch(image.copy(), bboxes, prob=1.0)
            return attacked
        return image
    raise ValueError(f"Unknown attack: {attack!r}")


DEFAULT_ATTACKS: Sequence[str] = ("none", "patch", "physical", "universal", "lowfreq")


# ── mAP (VOC-style, 11-point) ────────────────────────────────────────────


def _compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
    return inter / (union + 1e-10)


def voc_map(
    all_gt: List[List[Tuple[int, Sequence[float]]]],
    all_preds: List[List[Tuple[int, float, Sequence[float]]]],
    iou_threshold: float = 0.5,
) -> float:
    """VOC-style 11-point mean average precision at a fixed IoU threshold."""
    class_ids = sorted({cls_id for img_gt in all_gt for cls_id, _ in img_gt})
    aps = []
    for cls_id in class_ids:
        npos = sum(1 for img_gt in all_gt for c, _ in img_gt if c == cls_id)
        preds = [
            (conf, img_idx, bbox)
            for img_idx, img_pred in enumerate(all_preds)
            for c, conf, bbox in img_pred
            if c == cls_id
        ]
        if npos == 0 or not preds:
            continue
        preds.sort(key=lambda p: p[0], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched: List[set] = [set() for _ in range(len(all_gt))]
        for i, (_, img_idx, bbox) in enumerate(preds):
            best_iou, best_j = 0.0, -1
            for j, (c, gt_bbox) in enumerate(all_gt[img_idx]):
                if c != cls_id or j in matched[img_idx]:
                    continue
                iou = _compute_iou(bbox, gt_bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp[i] = 1.0
                matched[img_idx].add(best_j)
            else:
                fp[i] = 1.0
        rec = np.cumsum(tp) / npos
        prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = rec >= t
            ap += float(prec[mask].max()) if mask.any() else 0.0
        aps.append(ap / 11.0)
    return float(np.mean(aps)) if aps else 0.0


# ── Benchmark runner ─────────────────────────────────────────────────────


class BenchmarkRunner:
    """
    Runs the attack x defense benchmark matrix on a YOLO-format dataset.

    Args:
        images_dir: Directory of clean images (jpg/png/...).
        labels_dir: Directory of YOLO label files.
        output_dir: Where matrix CSVs/JSON/Markdown are written.
        detector:   Optional detector implementing `predict(image) -> [dict]`.
                    When None, only the model-free FFT detectability matrix is
                    produced.
        fft_threshold:   FFT flagging threshold (default: calibrated p99 of clean samples).
        high_freq_ratio: FFT high-frequency band fraction.
        n_images:  Number of sorted images to sample.
        seed:      Random seed for attack reproducibility.
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str = "runs/benchmark",
        detector: Optional[Any] = None,
        fft_threshold: Optional[float] = None,
        high_freq_ratio: float = 0.25,
        n_images: int = 40,
        seed: int = 42,
        jpeg_quality: int = 30,
        blur_kernel: int = 5,
        blur_sigma: float = 1.0,
        bit_depth: int = 4,
        patch_scale_range: Tuple[float, float] = (0.1, 0.4),
        patch_alpha: float = 0.8,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.detector = detector
        self.fft_threshold = fft_threshold
        self.high_freq_ratio = high_freq_ratio
        self.n_images = n_images
        self.seed = seed
        self.jpeg_quality = jpeg_quality
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.bit_depth = bit_depth
        self.patch_scale_range = patch_scale_range
        self.patch_alpha = patch_alpha

    # ── data loading ─────────────────────────────────────────────────────

    def _load_samples(self) -> List[Tuple[Any, List[Tuple[int, List[float]]]]]:
        files = sorted(f for f in self.images_dir.iterdir() if f.suffix.lower() in self.IMG_EXTS)[
            : self.n_images
        ]
        samples: List[Tuple[Any, List[Tuple[int, List[float]]]]] = []
        for img_path in files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            class_ids, boxes_cxcywh = parse_yolo_labels(
                str(label_path), image.shape[1], image.shape[0]
            )
            gt_xyxy = [
                (
                    cls_id,
                    [
                        cx - w / 2.0,
                        cy - h / 2.0,
                        cx + w / 2.0,
                        cy + h / 2.0,
                    ],
                )
                for cls_id, (cx, cy, w, h) in zip(class_ids, boxes_cxcywh)
            ]
            samples.append((image, gt_xyxy))
        if not samples:
            raise ValueError(f"No readable images found in {self.images_dir}")
        return samples

    # ── detectability (model-free) ───────────────────────────────────────

    def _resolve_threshold(self, samples) -> float:
        """Return the FFT threshold, calibrating it from clean scores if unset."""
        threshold = self.fft_threshold
        if threshold is None:
            clean_scores = [
                compute_fft_anomaly_score(img, self.high_freq_ratio) for img, _ in samples
            ]
            threshold = float(FFTCalibrator.compute_threshold(clean_scores, "percentile", 0.99))
            self.fft_threshold = threshold
            logger.info("Calibrated FFT threshold from clean scores: %.4f", threshold)
        return threshold

    def detectability_matrix(self, samples, attacks) -> Dict[str, Any]:
        """FFT flag rate per (defense, attack) + clean FPR, model-free."""
        rng = np.random.RandomState(self.seed)
        universal = generate_universal_perturbation([img for img, _ in samples], seed=self.seed)
        threshold = self._resolve_threshold(samples)

        rows: List[Dict[str, Any]] = []
        for defense in DEFAULT_DEFENSES:
            for attack in attacks:
                flagged = 0
                fpr_flagged = 0
                for image, gt in samples:
                    bboxes = [(int(cx), int(cy), int(w), int(h)) for _, (cx, cy, w, h) in gt]
                    attacked = apply_attack(attack, image, rng, bboxes=bboxes, universal=universal)
                    processed = apply_defense(
                        defense,
                        attacked,
                        jpeg_quality=self.jpeg_quality,
                        blur_kernel=self.blur_kernel,
                        blur_sigma=self.blur_sigma,
                        bit_depth=self.bit_depth,
                    )
                    if compute_fft_anomaly_score(processed, self.high_freq_ratio) > threshold:
                        flagged += 1
                    clean = apply_defense(
                        defense,
                        image,
                        jpeg_quality=self.jpeg_quality,
                        blur_kernel=self.blur_kernel,
                        blur_sigma=self.blur_sigma,
                        bit_depth=self.bit_depth,
                    )
                    if compute_fft_anomaly_score(clean, self.high_freq_ratio) > threshold:
                        fpr_flagged += 1
                rows.append(
                    {
                        "defense": defense,
                        "attack": attack,
                        "fft_flag_rate": flagged / len(samples),
                        "clean_fpr": fpr_flagged / len(samples),
                    }
                )
        return {"rows": rows, "n_images": len(samples), "fft_threshold": self.fft_threshold}

    # ── mAP matrix (requires detector) ───────────────────────────────────

    def _detector_predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.detector is None:
            raise RuntimeError("No detector configured")
        return self.detector.predict(image)

    def _map_for(self, samples, attack: str, defense: str) -> Dict[str, Any]:
        rng = np.random.RandomState(self.seed)
        universal = generate_universal_perturbation([img for img, _ in samples], seed=self.seed)
        threshold = self._resolve_threshold(samples)
        all_gt: List[List[Tuple[int, Sequence[float]]]] = []
        all_preds: List[List[Tuple[int, float, Sequence[float]]]] = []
        flagged_frames = 0

        for image, gt in samples:
            bboxes = [(int(cx), int(cy), int(w), int(h)) for _, (cx, cy, w, h) in gt]
            attacked = apply_attack(attack, image, rng, bboxes=bboxes, universal=universal)
            processed = apply_defense(
                defense,
                attacked,
                jpeg_quality=self.jpeg_quality,
                blur_kernel=self.blur_kernel,
                blur_sigma=self.blur_sigma,
                bit_depth=self.bit_depth,
            )

            detections = self._detector_predict(processed)
            if defense in POLICY_DEFENSES:
                score = compute_fft_anomaly_score(processed, self.high_freq_ratio)
                if score > threshold:
                    flagged_frames += 1
                    detections = []  # fail-safe suppression
            all_gt.append(gt)
            all_preds.append(
                [(int(d["class_id"]), float(d["confidence"]), list(d["bbox"])) for d in detections]
            )

        return {
            "mAP50": voc_map(all_gt, all_preds),
            "flagged_frames": flagged_frames,
        }

    def map_matrix(self, samples, attacks) -> Dict[str, Any]:
        """VOC mAP@0.5 per (defense, attack) plus median defense latency."""
        if self.detector is None:
            raise RuntimeError("map_matrix requires a detector")
        rows: List[Dict[str, Any]] = []
        for defense in DEFAULT_DEFENSES:
            latency = measure_defense_latency(
                defense,
                samples[0][0],
                jpeg_quality=self.jpeg_quality,
                blur_kernel=self.blur_kernel,
                blur_sigma=self.blur_sigma,
                bit_depth=self.bit_depth,
            )
            for attack in attacks:
                result = self._map_for(samples, attack, defense)
                rows.append(
                    {
                        "defense": defense,
                        "attack": attack,
                        "mAP50": round(result["mAP50"], 4),
                        "flagged_frames": result["flagged_frames"],
                        "latency_ms": round(latency, 3),
                    }
                )
        return {"rows": rows, "n_images": len(samples)}

    # ── output ───────────────────────────────────────────────────────────

    def _write(self, name: str, payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / f"{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        with open(self.output_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def run(
        self,
        defenses: Sequence[str] = DEFAULT_DEFENSES,
        attacks: Sequence[str] = DEFAULT_ATTACKS,
    ) -> Dict[str, Any]:
        """
        Run the full benchmark.

        Returns a dict with `detectability` (always) and, when a detector is
        configured, `map_matrix`. Writes CSV/JSON (+ Markdown when mAP ran)
        into the output directory.
        """
        samples = self._load_samples()
        if not samples:
            raise ValueError("No usable samples found; cannot run benchmark.")
        self._resolve_threshold(samples)
        summary: Dict[str, Any] = {"seed": self.seed, "n_images": len(samples)}

        detectability = self.detectability_matrix(samples, attacks)
        self._write("benchmark_detectability", detectability, detectability["rows"])
        summary["detectability"] = detectability

        if self.detector is not None:
            m = self.map_matrix(samples, attacks)
            self._write("benchmark_map", m, m["rows"])
            summary["map_matrix"] = m
            self._write_markdown(m["rows"])

        logger.info("Benchmark complete -> %s", self.output_dir)
        return summary

    def _write_markdown(self, rows: List[Dict[str, Any]]) -> None:
        defenses = DEFAULT_DEFENSES
        attacks = ["none", "patch", "physical", "universal", "lowfreq"]
        lines = [
            "# KAAL Benchmark Matrix — mAP@0.5 (VOC 11-point)",
            "",
            f"Dataset images: {self.images_dir}  |  images: {self.n_images}  |  "
            f"FFT threshold: {self.fft_threshold}",
            "",
            "| Defense | " + " | ".join(a.capitalize() for a in attacks) + " | Latency (ms) |",
            "|---------|" + "---|" * (len(attacks) + 1),
        ]
        for defense in defenses:
            cells = []
            latency = 0.0
            for attack in attacks:
                match = next(r for r in rows if r["defense"] == defense and r["attack"] == attack)
                cells.append(f"{match['mAP50']:.3f}")
                latency = match["latency_ms"]
            lines.append(f"| {defense} | " + " | ".join(cells) + f" | {latency:.2f} |")
        (self.output_dir / "benchmark_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────


class _YoloDetector:
    """Thin adapter: ultralytics YOLO -> benchmark detector contract."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25) -> None:
        from ultralytics import YOLO

        self._yolo = YOLO(model_path)
        self._conf = conf_threshold

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = self._yolo.predict(
            source=image, conf=self._conf, iou=0.45, agnostic_nms=True, verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            detections.append(
                {
                    "bbox": [float(v) for v in box.xyxy[0].cpu().numpy()],
                    "confidence": float(box.conf[0].item()),
                    "class_id": int(box.cls[0].item()),
                }
            )
        return detections


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the KAAL attack x defense benchmark matrix.")
    parser.add_argument("--images", required=True, help="Directory of clean images")
    parser.add_argument("--labels", required=True, help="Directory of YOLO label files")
    parser.add_argument("--output", default="runs/benchmark", help="Output directory")
    parser.add_argument(
        "--detector", default=None, help="YOLO weights path (optional; enables mAP)"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument(
        "--fft-threshold",
        type=float,
        default=None,
        help="FFT flagging threshold (default: calibrated from clean images at p99)",
    )
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=30)
    parser.add_argument("--bit-depth", type=int, default=4)
    args = parser.parse_args(argv)

    detector = _YoloDetector(args.detector, args.conf) if args.detector else None
    runner = BenchmarkRunner(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        detector=detector,
        fft_threshold=args.fft_threshold,
        n_images=args.n_images,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        bit_depth=args.bit_depth,
    )
    summary = runner.run()
    det = summary["detectability"]
    logger.info(
        "FFT detectability matrix (%d images, threshold=%.2f):",
        det["n_images"],
        det["fft_threshold"],
    )
    for row in det["rows"]:
        logger.info(
            "  %-13s %-10s flag_rate=%.3f clean_fpr=%.3f",
            row["defense"],
            row["attack"],
            row["fft_flag_rate"],
            row["clean_fpr"],
        )
    if "map_matrix" in summary:
        for row in summary["map_matrix"]["rows"]:
            logger.info(
                "  mAP %-13s %-10s %.3f (%.2f ms)",
                row["defense"],
                row["attack"],
                row["mAP50"],
                row["latency_ms"],
            )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
