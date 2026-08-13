# YOLOv8 Adversarial Robustness Pipeline

A production-quality pipeline for fine-tuning YOLOv8 with adversarial patch robustness and dual-stream defense.

## Features

- **Adversarial patch injection** — noise, structured patterns, gradient-optimized, or custom patch images
- **Mixed training** — fine-tune on both clean and adversarial-augmented images
- **Dual-stream defense** — FFT-based anomaly detection fused with YOLO predictions
- **Per-region detection** — individual objects flagged, not just whole frames
- **Adaptive calibration** — threshold auto-tuned per camera/environment from clean images
- **Physical-world realism** — perspective-warped, relit patches; low-frequency perturbations
- **Ablation harness** — sweeps thresholds/patch params, exports paper-ready curves
- **Three defense modes** — switch between no defense, adversarial training only, or full dual-stream
- **Comprehensive evaluation** — mAP on clean, attacked, and defended scenarios
- **Visualizations** — before/after attacks, annotated predictions, confusion matrices

---

## Project Structure

```
MODEL/
├── __init__.py                # Package init
├── config.py                  # Centralized configuration (hyperparams, paths, thresholds)
├── adversarial_patch.py       # Patch generation, placement, and probabilistic injection
├── augmentations.py           # Attack augmentation functions (patch, noise, occlusion, low-freq)
├── fft_scorer.py              # Standalone 2D-FFT anomaly scoring (model-free)
├── calibration.py             # Adaptive FFT threshold calibration per environment
├── dataset.py                 # YOLO-format dataset loading + augmentation pipeline
├── trainer.py                 # Fine-tuning: clean + adversarial mixed training
├── dual_stream_defense.py     # FFT anomaly detection + fusion with YOLO predictions
├── evaluator.py               # mAP computation, confusion matrix, metrics reporting
├── ablation.py                # Phase 1 hyperparameter sweep harness (paper curves)
├── visualizer.py              # Before/after visualization, bbox overlays, adversarial flags
├── pipeline.py                # CLI entry point (train → evaluate)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Setup

### Install Dependencies

```bash
pip install -e .                # core (MODEL pipeline)
pip install -e ".[sim]"         # + AirSim simulation scripts
pip install -e ".[server]"      # + FastAPI backend
```

Or install the split requirements directly:

```bash
pip install -r MODEL/requirements.txt -r requirements-sim.txt -r requirements-server.txt
```

> Run the pipeline from the repo root as `python -m MODEL.pipeline` so the
> `MODEL` package and its relative imports resolve correctly.

### 2. Prepare Your Dataset

Your dataset must be in **YOLO format** with a `data.yaml` file:

```
my_dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/       (optional)
    ├── images/
    └── labels/
```

Example `data.yaml`:

```yaml
train: train/images
val: val/images
nc: 3
names: ['car', 'person', 'bicycle']
```

---

## Usage

### Full Pipeline (Train → Evaluate)

```bash
python -m MODEL.pipeline --data path/to/data.yaml --train --epochs 50 --batch 16 --defense dual_stream
```

### Defense Modes

| Mode | Training | Inference | Description |
|------|----------|-----------|-------------|
| `none` | Clean data only | Standard YOLO | Baseline, no defense |
| `adversarial_training` | Clean + adversarial | Standard YOLO | Robust training only |
| `dual_stream` | Clean + adversarial | YOLO + FFT anomaly | Full defense (recommended) |

```bash
# No defense (baseline)
python -m MODEL.pipeline --data data.yaml --train --defense none

# Adversarial training only
python -m MODEL.pipeline --data data.yaml --train --defense adversarial_training

# Full dual-stream defense (default)
python -m MODEL.pipeline --data data.yaml --train --defense dual_stream
```

### Evaluate Only (Skip Training)

```bash
python -m MODEL.pipeline --data data.yaml --model path/to/best.pt
```

### Resume Training from Checkpoint

```bash
python -m MODEL.pipeline --data data.yaml --train --resume
```

### All CLI Options

```
--data              Path to data.yaml (required)
--model             Pretrained model weights (default: yolov8n.pt)
--train             Run training before evaluation
--resume            Resume training from last checkpoint
--defense           Defense mode: none/adversarial_training/dual_stream (default: dual_stream)
--epochs            Training epochs (default: 50)
--batch             Batch size (default: 16)
--imgsz             Image size (default: 640)
--device            Device: '0' for GPU, 'cpu' for CPU (default: 0)
--fft-threshold     FFT anomaly threshold (default: calibration profile or 0.3)
--high-freq-ratio   Fraction of spectrum considered high-frequency (default: 0.25)
--frame-fft         Use whole-frame FFT only (disable per-region mode)
--region-min-size   Skip per-region FFT for detections smaller than this (default: 16)
--calibration       Path to a saved calibration profile (JSON)
```

---

## Phase 1: Harden the Defense

### Adaptive Thresholding (calibration)

The old hardcoded `fft_threshold: 0.3` is now the *fallback*. For a real
deployment, calibrate the threshold from that camera's clean imagery:

```bash
python -m MODEL.calibration \
    --images /path/to/clean/camera/feed \
    --environment drone_cam_01 \
    --output profiles/drone_cam_01.json \
    --n 100 --method percentile --percentile 0.99
```

The profile records the score distribution (mean/std, p50/p95/p99/p99.9) and
a recommended threshold. Point the pipeline at it with `--calibration`:

```bash
python -m MODEL.pipeline --data data.yaml --calibration profiles/drone_cam_01.json
```

Profiles are per camera/environment, so swap the JSON per deployment site.

### Per-Region FFT

By default the defense now scores **each detection on its own bbox crop**
instead of flagging entire frames:

- Every detection gets `anomaly_score` and `is_adversarial` fields.
- Boxes smaller than `--region-min-size` (default 16px) are skipped as too
  noisy to score (`region_scored: False`).
- Overlapping detections reuse the score of their strongest overlapping box
  instead of double-flagging.
- `predict()` returns `objects_flagged` plus the whole-frame score.
- Visualizer draws flagged objects in red with an `fft=` label.

Disable per-region behavior with `--frame-fft` if you prefer whole-frame only.

### Gradient & Physical-World Patches

`PatchType.GRADIENT` produces a **gradient-optimized** patch:
- With a detector model attached, runs PGD (torch) to suppress the model's
  objectness confidence under the patch (evasion).
- Without a model, produces a derivative-free **low-frequency** perturbation
  (smooth gradient field) that is designed to slip under the FFT defense —
  the failure case the ablation study exposes.

Physical-world simulation (on by default for all patch types):
- **Perspective warp** — patch corners jittered to simulate a sticker viewed
  at an angle.
- **Lighting/exposure** — random per-channel exposure, gamma, brightness and
  contrast.
- `patch_scale_range` now spans **5%–50%** of the bounding box (was 10–30%).

`augmentations.add_low_freq_noise()` adds a smooth low-frequency perturbation
as a first-class attack type (`lowfreq` in `apply_random_attack`).

### Ablation Study

Sweep the four key hyperparameters and export precision/recall/F1 curves:

```bash
python -m MODEL.ablation \
    --images path/to/val/images \
    --labels path/to/val/labels \
    --output runs/ablation
```

Produces `sweep_{high_freq_ratio,fft_threshold,patch_scale_range,patch_alpha}`
as `.csv`, `.json`, and `.png` (matplotlib) — the paper figures.

---

## How It Works

### 1. Adversarial Patch Injection

During training, adversarial patches are injected onto objects within their bounding boxes:
- **Noise patches**: random pixel noise acting as universal perturbation
- **Pattern patches**: high-frequency checkerboard patterns
- **Gradient patches**: PGD-optimized against the detector (or low-frequency
  FFT-aware perturbations when no model is attached)
- **Loaded patches**: custom adversarial images (e.g., printed stickers)

Patches are alpha-blended within bounding boxes so annotations remain valid.
Perspective warp and lighting/exposure variation simulate physical-world
placement.

### 2. Dual-Stream Defense

At inference time, each image is processed through two streams:

- **Stream 1 (YOLO)**: Standard object detection
- **Stream 2 (FFT)**: Convert to frequency domain → measure high-frequency energy ratio

**Fusion**: If the FFT anomaly score exceeds the threshold, the image — or, in
per-region mode, the individual detection — is flagged as adversarial and
predictions are marked unreliable. The threshold can be calibrated per
camera/environment via `MODEL.calibration` instead of using the fixed default.

### 3. Evaluation

Three-pass evaluation on the validation set:
1. **Clean**: mAP on unmodified images
2. **Adversarial**: mAP on attacked images (no defense)
3. **Defended**: mAP on attacked images with dual-stream defense active

---

## Output Structure

```
runs/adversarial/
├── augmented_dataset/         # Mixed clean + adversarial training data
│   ├── train/
│   ├── val/
│   └── data.yaml
├── train/                     # Ultralytics training outputs
│   └── adversarial_finetune/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── visualizations/            # Generated images
│   ├── adversarial_comparison.png
│   ├── sample_predictions_clean.png
│   ├── sample_predictions_attacked.png
│   └── confusion_matrix.png
└── evaluation_report.txt      # Metrics summary
```

---

## Example Output

```
============================================================
  ROBUSTNESS EVALUATION REPORT
============================================================

  Scenario                   | mAP@0.5 | TP  | FP | FN
  ---------------------------+---------+-----+----+---
  Clean (baseline)           | 0.7823  | 245 | 32 | 18
  Adversarial (no defense)   | 0.4156  | 128 | 89 | 135
  Adversarial + Defense      | 0.6891  | 210 | 45 | 53

  Adversarial images flagged: 42/50
  Mean anomaly score: 0.4521
  Max anomaly score: 0.7834

  mAP drop from attack: -0.3667
  mAP recovery with defense: +0.2735
```
