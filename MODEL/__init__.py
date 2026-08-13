"""
KAAL MODEL — Adversarial Robustness Pipeline for YOLOv8.

Core modules:
    - config:              Centralized hyperparameters and defense mode settings
    - adversarial_patch:   Patch generation, loading, and injection
    - augmentations:       Attack augmentation functions (patch, noise, occlusion, low-freq)
    - fft_scorer:          Standalone 2D-FFT anomaly scoring (model-free)
    - calibration:         Adaptive FFT threshold calibration per environment
    - dataset:             YOLO-format dataset loading with adversarial augmentation
    - trainer:             Fine-tuning YOLOv8 with adversarial robustness
    - dual_stream_defense: FFT-based dual-stream adversarial defense (per-region)
    - evaluator:           Robustness evaluation (clean / attacked / defended)
    - ablation:            Phase 1 hyperparameter sweep harness (paper curves)
    - visualizer:          Before/after comparisons and annotated predictions
    - pipeline:            Full train → evaluate CLI entry point
"""
