"""
failure_analysis.py — Document when the FFT defense fails (Phase 3).

Measures the two failure modes called out in the roadmap:

  1. **Natural high-frequency textures** (foliage, urban clutter, rain)
     produce FFT anomaly scores above a threshold calibrated on clean
     imagery -> false positives on perfectly clean frames.

  2. **Low-frequency adversarial perturbations** (smoothed universal noise,
     `add_low_freq_noise`, low-freq gradient patches) change content without
     adding high-frequency energy -> the FFT detector misses them.

Synthetic stand-ins are used so the analysis runs deterministically and
without a camera/dataset:

  - `sky`      : smooth gradient (calibration baseline).
  - `foliage`  : green speckle + coarse blobs (leaf texture).
  - `urban`    : dense sharp-edged rectangles (building clutter).
  - `rain`     : sparse thin diagonal streaks.

Output: `runs/failure_analysis/report.json` and `report.md`.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from .fft_scorer import compute_fft_anomaly_score
from .calibration import FFTCalibrator
from .augmentations import add_low_freq_noise
from .benchmark import generate_universal_perturbation
from .temporal_scorer import TemporalConsistencyDetector

logger = logging.getLogger(__name__)

TEXTURE_SIZE = 160


def _smooth_gradient(
    rng: np.random.RandomState, h: int = TEXTURE_SIZE, w: int = TEXTURE_SIZE
) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    base = (yy / h * 0.6 + xx / w * 0.4) * 255.0
    base = np.stack([base, base * 0.9, base * 0.7], axis=-1)
    noise = rng.normal(0, 8, (h, w, 1))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _foliage(
    rng: np.random.RandomState, h: int = TEXTURE_SIZE, w: int = TEXTURE_SIZE
) -> np.ndarray:
    img = _smooth_gradient(rng, h, w)
    speckle = rng.randint(0, 255, (h, w, 3)).astype(np.float32)
    speckle = cv2.GaussianBlur(speckle, (3, 3), 0)
    mask = (rng.rand(h, w, 1) > 0.55).astype(np.float32)
    img = np.clip(img.astype(np.float32) * 0.7 + speckle * mask * 0.6, 0, 255)
    img[..., 0] = np.clip(img[..., 0] * 0.8, 0, 255)  # lean green
    return img.astype(np.uint8)


def _urban(rng: np.random.RandomState, h: int = TEXTURE_SIZE, w: int = TEXTURE_SIZE) -> np.ndarray:
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    for _ in range(40):
        x, y = rng.randint(0, w - 8), rng.randint(0, h - 8)
        bw, bh = rng.randint(4, 24), rng.randint(4, 24)
        color = int(rng.randint(0, 255))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (color, color // 2, 255 - color), 1)
    return img


def _rain(rng: np.random.RandomState, h: int = TEXTURE_SIZE, w: int = TEXTURE_SIZE) -> np.ndarray:
    img = _smooth_gradient(rng, h, w).astype(np.float32)
    for _ in range(120):
        x, y = rng.randint(0, w), rng.randint(0, h)
        length = rng.randint(8, 18)
        cv2.line(
            img,
            (x, y),
            (x + length // 3, y + length),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.clip(img, 0, 255).astype(np.uint8)


TEXTURE_GENERATORS: Dict[str, Any] = {
    "sky": _smooth_gradient,
    "foliage": _foliage,
    "urban": _urban,
    "rain": _rain,
}


def generate_textures(n: int = 40, seed: int = 0) -> Dict[str, List[np.ndarray]]:
    """Deterministic synthetic stand-ins for clean scene textures."""
    rng = np.random.RandomState(seed)
    out: Dict[str, List[np.ndarray]] = {}
    for name, gen in TEXTURE_GENERATORS.items():
        out[name] = [gen(rng) for _ in range(n)]
    return out


def run_failure_analysis(
    n: int = 40,
    seed: int = 0,
    high_freq_ratio: float = 0.25,
    percentile: float = 0.99,
    region: int = 0,
) -> Dict[str, Any]:
    """
    Measure FPR on natural textures and TPR on low-frequency attacks.

    With ``region > 0``, additionally scores a centered ``region``-pixel crop
    (a detector-box proxy) to quantify how much of the FPR comes from
    background texture outside the region of interest.

    Returns a dict of measured results (also written to
    `runs/failure_analysis/report.json` + `report.md`).
    """
    textures = generate_textures(n, seed)

    # 1. Calibrate a threshold on "sky" (the clean baseline), like Phase 1.
    sky_scores = [compute_fft_anomaly_score(img, high_freq_ratio) for img in textures["sky"]]
    threshold = float(FFTCalibrator.compute_threshold(sky_scores, "percentile", percentile))

    def _crop(img: np.ndarray) -> np.ndarray:
        if region <= 0:
            return img
        h, w = img.shape[:2]
        r = min(region, h, w)
        y0, x0 = (h - r) // 2, (w - r) // 2
        return img[y0 : y0 + r, x0 : x0 + r]

    # 2. FPR per natural texture at that threshold (whole-frame and ROI crop).
    fpr: Dict[str, float] = {}
    mean_scores: Dict[str, float] = {}
    fpr_region: Dict[str, float] = {}
    region_threshold: float | None = None
    for name, images in textures.items():
        scores = [compute_fft_anomaly_score(img, high_freq_ratio) for img in images]
        fpr[name] = sum(1 for s in scores if s > threshold) / len(scores)
        mean_scores[name] = float(np.mean(scores))
        if region > 0:
            region_scores = [
                compute_fft_anomaly_score(_crop(img), high_freq_ratio) for img in images
            ]
            if name == "sky":
                region_threshold = float(
                    FFTCalibrator.compute_threshold(region_scores, "percentile", percentile)
                )
                fpr_region[name] = 0.0
            else:
                assert region_threshold is not None
                fpr_region[name] = sum(1 for s in region_scores if s > region_threshold) / len(
                    region_scores
                )

    # 3. Low-frequency attacks: TPR (should be LOW — the failure case).
    base = textures["sky"]
    universal = generate_universal_perturbation(base, seed=seed, low_freq=True)
    rng = np.random.RandomState(seed)

    def _flag_rate(images: List[np.ndarray]) -> float:
        return sum(
            1 for img in images if compute_fft_anomaly_score(img, high_freq_ratio) > threshold
        ) / len(images)

    lowfreq_images = [add_low_freq_noise(img) for img in base]
    universal_images = [
        np.clip(img.astype(np.float32) + universal, 0, 255).astype(np.uint8) for img in base
    ]
    noise_images = [
        np.clip(img.astype(np.float32) + rng.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
        for img in base
    ]

    # 4. Temporal-consistency cue on the same attack set (mitigation #3).
    temporal_detector = TemporalConsistencyDetector()
    temp_threshold = temporal_detector.calibrate(base, percentile)
    temp_fpr = sum(
        1 for img in base if temporal_detector.score_and_update(img) > temp_threshold
    ) / len(base)
    temp_lowfreq = temporal_detector.detection_rate(lowfreq_images, temp_threshold)
    temp_universal = temporal_detector.detection_rate(universal_images, temp_threshold)

    report: Dict[str, Any] = {
        "seed": seed,
        "n_images": n,
        "high_freq_ratio": high_freq_ratio,
        "calibrated_threshold": threshold,
        "sky_score": {"mean": mean_scores["sky"], "p99": threshold, "max": max(sky_scores)},
        "fpr_by_texture": fpr,
        "mean_score_by_texture": mean_scores,
        "region": region,
        "region_threshold": region_threshold,
        "fpr_by_texture_region": fpr_region,
        "attack_detection_rate": {
            "low_freq_noise": _flag_rate(lowfreq_images),
            "universal_low_freq": _flag_rate(universal_images),
            "gaussian_noise_high_freq": _flag_rate(noise_images),
        },
        "temporal_cue": {
            "threshold": temp_threshold,
            "clean_fpr": temp_fpr,
            "low_freq_noise": temp_lowfreq,
            "universal_low_freq": temp_universal,
        },
        "failure_modes": [],
        "mitigations": [],
    }

    # 4. Interpret the measurements.
    for name in ("foliage", "urban", "rain"):
        if fpr[name] > 0.05:
            region_note = ""
            if region > 0 and name in fpr_region and fpr_region[name] < fpr[name]:
                region_note = (
                    f" Scoring only the ROI crop drops this to FPR {fpr_region[name]:.2f}, "
                    "confirming the background texture (not the detector's box) is the culprit."
                )
            report["failure_modes"].append(
                f"Natural {name} texture produces FPR {fpr[name]:.2f} at the calibrated "
                f"threshold ({threshold:.4f}) — clean frames may be flagged as adversarial."
                f"{region_note}"
            )
    if report["attack_detection_rate"]["low_freq_noise"] < 0.5:
        report["failure_modes"].append(
            "Low-frequency perturbation is largely invisible to the FFT detector "
            "(detection rate < 50%) — an attacker can modify content without being flagged."
        )
    if report["attack_detection_rate"]["universal_low_freq"] < 0.5:
        report["failure_modes"].append(
            "Smoothed universal perturbation evades the FFT detector (detection rate < 50%)."
        )
    if report["temporal_cue"]["low_freq_noise"] >= 0.5:
        report["failure_modes"].append(
            "The temporal-consistency cue (MODEL/temporal_scorer.py) recovers the low-frequency "
            "attacks the FFT misses: fusing FFT + temporal residual closes this gap."
        )

    report["mitigations"] = [
        "Combine FFT with a preprocessing defense (e.g. JPEG/bit-depth): 'fft + jpeg' neutralises "
        "high-frequency attacks before scoring and only suppresses when a real signature survives.",
        "Score per-region (object crop) instead of whole-frame to ignore static background texture; "
        "foliage/urban clutter often lives outside the detector's boxes.",
        "Calibrate per environment and include a few 'challenging clean' frames (foliage, rain) in "
        "the calibration set so the threshold adapts to natural high-frequency content.",
        "Fuse FFT with a second, non-frequency cue: MODEL/temporal_scorer.TemporalConsistencyDetector "
        "tracks an EMA background model and flags persistent per-patch residuals that the FFT misses.",
        "For low-frequency attacks, monitor per-patch residual vs. temporal consistency across "
        "frames rather than single-frame spectral energy.",
    ]

    out_dir = Path("runs/failure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (out_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    logger.info("Failure analysis written -> %s", out_dir)
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# FFT Defense — Failure Analysis",
        "",
        f"- Images per texture: {report['n_images']}",
        f"- Calibrated threshold (99th percentile of 'sky'): {report['calibrated_threshold']:.4f}",
        "",
        "## False-positive rate on natural clean textures",
        "",
        "| Texture | Mean FFT score | FPR at threshold |",
        "|---------|----------------|------------------|",
    ]
    for name, fpr in report["fpr_by_texture"].items():
        lines.append(f"| {name} | {report['mean_score_by_texture'][name]:.4f} | {fpr:.3f} |")
    lines += [
        "",
        "## Attack detection rate (should be HIGH for the FFT detector)",
        "",
        "| Attack | Detection rate |",
        "|--------|----------------|",
    ]
    for name, rate in report["attack_detection_rate"].items():
        lines.append(f"| {name} | {rate:.3f} |")
    if report.get("region", 0) > 0:
        lines += [
            "",
            f"### FPR scoring only the ROI crop (center {report['region']}px, "
            f"region-calibrated threshold {report['region_threshold']:.4f})",
            "",
            "| Texture | Whole-frame FPR | ROI-crop FPR |",
            "|---------|-----------------|--------------|",
        ]
        for name in report["fpr_by_texture"]:
            lines.append(
                f"| {name} | {report['fpr_by_texture'][name]:.3f} | "
                f"{report['fpr_by_texture_region'][name]:.3f} |"
            )
    temp = report["temporal_cue"]
    lines += [
        "",
        "## Temporal-consistency cue (non-frequency, EMA background residual)",
        "",
        f"- Calibrated threshold: {temp['threshold']:.4f}",
        f"- Clean FPR: {temp['clean_fpr']:.3f}",
        f"- low_freq_noise detection rate: {temp['low_freq_noise']:.3f}",
        f"- universal_low_freq detection rate: {temp['universal_low_freq']:.3f}",
    ]
    lines += ["", "## Failure modes", ""]
    for mode in report["failure_modes"]:
        lines.append(f"- {mode}")
    lines += ["", "## Proposed mitigations", ""]
    for mit in report["mitigations"]:
        lines.append(f"- {mit}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run FFT failure analysis.")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--high-freq-ratio", type=float, default=0.25)
    parser.add_argument(
        "--region",
        type=int,
        default=64,
        help="Center-crop size (px) to also score as a detector-ROI proxy; 0 disables",
    )
    args = parser.parse_args(argv)

    report = run_failure_analysis(
        n=args.n,
        seed=args.seed,
        high_freq_ratio=args.high_freq_ratio,
        region=args.region,
    )
    for name, fpr in report["fpr_by_texture"].items():
        logger.info("FPR %-8s %.3f", name, fpr)
    if report.get("region", 0) > 0:
        logger.info("ROI-crop FPR: %s", report["fpr_by_texture_region"])
    logger.info("Attack detection rates: %s", report["attack_detection_rate"])
    logger.info("Temporal cue: %s", report["temporal_cue"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
