"""
calibration.py — Adaptive FFT threshold calibration (Phase 1).

Calibration mode:
  1. Run the FFT anomaly scorer over N clean images for a camera/environment.
  2. Compute the score distribution (mean, std, percentiles).
  3. Set the threshold at the 99th percentile (or mean + k*std) of that
     distribution.
  4. Store the calibration profile per camera/environment as JSON so it can be
     reused at inference time.

This replaces the hardcoded `fft_threshold: 0.3` with values measured from
real clean imagery for the specific deployment environment.
"""

import os
import json
import cv2
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .fft_scorer import compute_fft_anomaly_score

logger = logging.getLogger(__name__)


@dataclass
class CalibrationProfile:
    """
    Immutable record of a calibration run for one camera/environment.

    Attributes:
        environment_id: Name of the camera/environment (e.g. "drone_cam_01").
        high_freq_ratio: FFT high-freq ratio used during scoring.
        scores: Raw clean-image FFT scores collected during calibration.
        threshold: The recommended anomaly threshold for this environment.
        method: "percentile" or "sigma".
        count: Number of clean images scored.
    """

    environment_id: str
    high_freq_ratio: float = 0.25
    scores: List[float] = field(default_factory=list)
    threshold: float = 0.3
    method: str = "percentile"
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationProfile":
        return cls(
            environment_id=str(data.get("environment_id", "unknown")),
            high_freq_ratio=float(data.get("high_freq_ratio", 0.25)),
            scores=[float(s) for s in data.get("scores", [])],
            threshold=float(data.get("threshold", 0.3)),
            method=str(data.get("method", "percentile")),
            count=int(data.get("count", len(data.get("scores", [])))),
            mean=float(data.get("mean", 0.0)),
            std=float(data.get("std", 0.0)),
            p50=float(data.get("p50", 0.0)),
            p95=float(data.get("p95", 0.0)),
            p99=float(data.get("p99", 0.0)),
            p999=float(data.get("p999", 0.0)),
        )


class FFTCalibrator:
    """
    Calibrates the FFT anomaly threshold from a set of clean images.

    Usage:
        calibrator = FFTCalibrator(environment_id="drone_cam_01")
        profile = calibrator.calibrate_from_dir("/data/clean", n=100)
        calibrator.save_profile(profile, "profiles/drone_cam_01.json")
    """

    def __init__(
        self,
        environment_id: str = "default",
        high_freq_ratio: float = 0.25,
    ):
        self.environment_id = environment_id
        self.high_freq_ratio = high_freq_ratio

    # ── Score Collection ──────────────────────────────────────────────────

    def collect_scores(
        self,
        images_dir: str,
        n: int = 100,
    ) -> List[float]:
        """
        Run the FFT anomaly scorer over clean images.

        Args:
            images_dir: Directory of clean images for this environment.
            n:          Maximum number of images to score.

        Returns:
            List of per-image FFT anomaly scores.
        """
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = sorted(
            f for f in os.listdir(images_dir) if Path(f).suffix.lower() in img_extensions
        )[:n]

        if not files:
            logger.warning("No clean images found in %s", images_dir)
            return []

        scores = []
        for fname in files:
            img = cv2.imread(os.path.join(images_dir, fname))
            if img is None:
                continue
            scores.append(compute_fft_anomaly_score(img, self.high_freq_ratio))
        logger.info("Collected %d FFT scores from %s", len(scores), images_dir)
        return scores

    # ── Threshold Computation ─────────────────────────────────────────────

    @staticmethod
    def compute_threshold(
        scores: List[float],
        method: str = "percentile",
        percentile: float = 0.99,
        sigma: float = 3.0,
    ) -> float:
        """
        Derive an anomaly threshold from a clean score distribution.

        Args:
            scores:     Clean-image FFT scores.
            method:     "percentile" → np.percentile(scores, percentile*100).
                        "sigma"      → mean + sigma * std.
            percentile: Percentile (0-1) for the "percentile" method.
            sigma:      Number of std devs above mean for the "sigma" method.

        Returns:
            Threshold value. Falls back to 0.3 if no scores are available.
        """
        if not scores:
            logger.warning("No scores to calibrate from; using default 0.3")
            return 0.3

        arr = np.asarray(scores, dtype=np.float64)
        if method == "sigma":
            return float(arr.mean() + sigma * arr.std())
        return float(np.percentile(arr, percentile * 100.0))

    # ── Profile Build / Save / Load ───────────────────────────────────────

    def build_profile(
        self,
        scores: List[float],
        method: str = "percentile",
        percentile: float = 0.99,
        sigma: float = 3.0,
    ) -> CalibrationProfile:
        """Build a CalibrationProfile from collected scores."""
        if not scores:
            return CalibrationProfile(environment_id=self.environment_id)

        arr = np.asarray(scores, dtype=np.float64)
        threshold = self.compute_threshold(scores, method, percentile, sigma)
        return CalibrationProfile(
            environment_id=self.environment_id,
            high_freq_ratio=self.high_freq_ratio,
            scores=list(arr),
            threshold=threshold,
            method=method,
            count=int(arr.size),
            mean=float(arr.mean()),
            std=float(arr.std()),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            p999=float(np.percentile(arr, 99.9)),
        )

    def calibrate_from_dir(
        self,
        images_dir: str,
        n: int = 100,
        method: str = "percentile",
        percentile: float = 0.99,
        sigma: float = 3.0,
    ) -> CalibrationProfile:
        """One-shot: collect scores from a directory and build a profile."""
        scores = self.collect_scores(images_dir, n)
        return self.build_profile(scores, method, percentile, sigma)

    @staticmethod
    def save_profile(profile: CalibrationProfile, path: str):
        """Persist a calibration profile as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        logger.info("Calibration profile saved to: %s (threshold=%.4f)", path, profile.threshold)

    @staticmethod
    def load_profile(path: str) -> CalibrationProfile:
        """Load a calibration profile from JSON."""
        with open(path, "r") as f:
            return CalibrationProfile.from_dict(json.load(f))


def load_calibrated_threshold(
    profile_path: Optional[str],
    fallback: float,
    high_freq_ratio: float = 0.25,
) -> float:
    """
    Load a saved calibration profile and return its threshold.

    Args:
        profile_path: Path to a calibration JSON profile (may be None).
        fallback:     Value to use if no profile is available.
        high_freq_ratio: FFT high-freq ratio to sanity-check the profile matches.

    Returns:
        Calibrated threshold if the profile exists, else the fallback.
    """
    if not profile_path:
        return fallback
    if not os.path.exists(profile_path):
        logger.warning(
            "Calibration profile not found: %s (using fallback %.4f)", profile_path, fallback
        )
        return fallback

    try:
        profile = FFTCalibrator.load_profile(profile_path)
        if profile.high_freq_ratio != high_freq_ratio:
            logger.warning(
                "Calibration profile was built with high_freq_ratio=%.2f but "
                "the model uses %.2f; consider re-calibrating.",
                profile.high_freq_ratio,
                high_freq_ratio,
            )
        logger.info("Using calibrated FFT threshold: %.4f (%s)", profile.threshold, profile.method)
        return profile.threshold
    except Exception as exc:
        logger.warning(
            "Failed to load calibration profile %s: %s (using fallback %.4f)",
            profile_path,
            exc,
            fallback,
        )
        return fallback


def main():
    """CLI: calibrate the FFT threshold from a directory of clean images."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate the FFT anomaly threshold from clean images."
    )
    parser.add_argument(
        "--images", required=True, help="Directory of clean images for this environment"
    )
    parser.add_argument(
        "--environment", default="default", help="Camera/environment identifier for the profile"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the calibration profile JSON"
    )
    parser.add_argument("--n", type=int, default=100, help="Number of clean images to score")
    parser.add_argument(
        "--high-freq-ratio", type=float, default=0.25, help="FFT high-frequency band fraction"
    )
    parser.add_argument(
        "--method",
        default="percentile",
        choices=("percentile", "sigma"),
        help="Threshold derivation method",
    )
    parser.add_argument(
        "--percentile", type=float, default=0.99, help="Percentile (0-1) when method=percentile"
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0, help="Std-deviation multiple when method=sigma"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    calibrator = FFTCalibrator(
        environment_id=args.environment,
        high_freq_ratio=args.high_freq_ratio,
    )
    profile = calibrator.calibrate_from_dir(
        args.images,
        n=args.n,
        method=args.method,
        percentile=args.percentile,
        sigma=args.sigma,
    )
    if not profile.scores:
        raise SystemExit("No images scored — calibration failed.")
    calibrator.save_profile(profile, args.output)
    logger.info(
        "Recommended threshold: %.4f (%s, mean=%.4f, std=%.4f)",
        profile.threshold,
        profile.method,
        profile.mean,
        profile.std,
    )


if __name__ == "__main__":
    main()
