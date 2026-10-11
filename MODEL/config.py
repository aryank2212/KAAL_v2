"""
config.py — Centralized configuration for the YOLOv8 adversarial robustness pipeline.

All hyperparameters, paths, thresholds, and defense mode settings are managed here
so that nothing is hardcoded across the codebase.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional

from .fft_scorer import DEFAULT_FFT_THRESHOLD


class DefenseMode(Enum):
    """Available defense strategies during training and inference."""

    NONE = "none"  # Baseline: no adversarial defense
    ADVERSARIAL_TRAINING = (
        "adversarial_training"  # Train on clean + adversarial, standard inference
    )
    DUAL_STREAM = "dual_stream"  # Adversarial training + FFT dual-stream at inference


class PatchType(Enum):
    """Type of adversarial patch to generate."""

    NOISE = "noise"  # Random pixel noise
    PATTERN = "pattern"  # Structured high-frequency pattern
    LOADED = "loaded"  # Load from an image file
    GRADIENT = "gradient"  # Gradient-optimized against the model (PGD-style)


@dataclass
class Config:
    """
    Master configuration object.

    Populate from CLI args (main.py) or instantiate directly for programmatic use.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    data_yaml: str = "data.yaml"  # Path to YOLO-format data.yaml
    model_weights: str = "yolov8m.pt"  # Pretrained model checkpoint
    output_dir: str = "runs/adversarial"  # Directory for all outputs
    patch_image_path: Optional[str] = None  # Path to a pre-made patch (PatchType.LOADED)

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 8
    lr: float = 0.001
    imgsz: int = 640
    device: str = "0"  # "0" for GPU, "cpu" for CPU fallback

    # ── Adversarial Patch ──────────────────────────────────────────────────
    patch_prob: float = 0.5  # Probability of injecting a patch per image
    patch_scale_range: Tuple[float, float] = (
        0.05,
        0.5,
    )  # Patch size as fraction of bbox (Phase 1: 5%–50%)
    patch_type: PatchType = PatchType.NOISE  # Which patch generation method to use
    patch_alpha: float = 0.8  # Blending alpha (1.0 = fully opaque)
    patch_perspective: bool = True  # Apply random perspective warp to patches (physical-world sim)
    patch_lighting: bool = True  # Apply random lighting/exposure variation to patches
    patch_gradient: bool = False  # Use gradient-based (PGD) patch optimization
    patch_opt_steps: int = 30  # PGD optimization steps for gradient patches
    patch_opt_lr: float = 0.1  # PGD step size for gradient patches

    # ── Defense / Dual-Stream ──────────────────────────────────────────────
    defense_mode: DefenseMode = DefenseMode.DUAL_STREAM
    fft_threshold: Optional[float] = (
        None  # Anomaly threshold; None = use calibration profile or default
    )
    fft_default_threshold: float = (
        DEFAULT_FFT_THRESHOLD  # Fallback when no calibration profile exists
    )
    fft_high_freq_ratio: float = 0.25  # Fraction of spectrum considered "high-freq"
    fft_region_mode: bool = True  # Per-bbox crop FFT (Phase 1: per-object flags, not whole-frame)
    fft_region_min_size: int = 16  # Skip regions smaller than this many pixels
    fft_overlap_iou: float = 0.5  # Reuse score for boxes overlapping an already-scored box

    # ── Calibration (Phase 1) ──────────────────────────────────────────────
    calibration_profile: Optional[str] = None  # Path to saved calibration profile (JSON)
    calibration_samples: int = 100  # N clean images to run FFT over during calibration
    calibration_method: str = "percentile"  # "percentile" | "sigma"
    calibration_percentile: float = 0.99  # Threshold at this percentile of clean score distribution
    calibration_sigma: float = 3.0  # Threshold at mean + k*std (when method == "sigma")

    # ── Evaluation ─────────────────────────────────────────────────────────
    conf_threshold: float = (
        0.45  # Confidence threshold for predictions (increased to avoid weak edge detection)
    )
    iou_threshold: float = (
        0.40  # IoU threshold for mAP/NMS (lowered to force highly overlapping fragments to merge)
    )

    # ── Resume ────────────────────────────────────────────────────────────
    resume: bool = False  # Resume training from last checkpoint

    # ── Visualization ──────────────────────────────────────────────────────
    num_vis_samples: int = 8  # Number of sample images to visualize

    def __post_init__(self):
        """Validate configuration values after initialization."""
        assert 0.0 <= self.patch_prob <= 1.0, "patch_prob must be in [0, 1]"
        assert 0.0 < self.patch_alpha <= 1.0, "patch_alpha must be in (0, 1]"
        assert self.fft_threshold is None or 0.0 < self.fft_threshold < 1.0, (
            "fft_threshold must be in (0, 1)"
        )
        assert 0.0 < self.fft_default_threshold < 1.0, "fft_default_threshold must be in (0, 1)"
        assert self.calibration_method in ("percentile", "sigma"), (
            "calibration_method must be 'percentile' or 'sigma'"
        )
        assert self.patch_scale_range[0] < self.patch_scale_range[1], (
            "patch_scale_range[0] must be < patch_scale_range[1]"
        )
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
