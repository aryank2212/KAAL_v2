# KAAL Phase 3 — Benchmarking & Failure Analysis Results

Generated with `MODEL/failure_analysis.py` and `MODEL/benchmark.py` on a synthetic
evaluation set (blurred-noise scenes with a synthetic object box, YOLO labels).
Artifacts: `runs/failure_analysis/`, `runs/benchmark/`.

## 1. Failure analysis (detector-free)

Settings: 40 images per natural texture, `high_freq_ratio=0.25`,
threshold = 99th percentile of clean sky scores (`0.5580`).

### False-positive rate on natural clean textures

| Texture | Mean FFT score | FPR at threshold |
|---------|----------------|------------------|
| sky     | 0.5570         | 0.025            |
| foliage | 0.5579         | **0.425**        |
| urban   | 0.5277         | 0.000            |
| rain    | 0.5528         | 0.000            |

Foliage (natural high-frequency texture) is flagged as adversarial ~42% of the
time at a threshold calibrated on clean sky — a real-world FPR failure mode.

### Attack detection rate (FFT detector should flag these)

| Attack                   | Detection rate |
|--------------------------|----------------|
| low_freq_noise           | **0.000**      |
| universal_low_freq       | **0.000**      |
| gaussian_noise_high_freq | 0.200          |

Low-frequency and smoothed-universal perturbations are effectively invisible to
the FFT detector — an attacker can modify content without being flagged.

## 2. Benchmark matrix — FFT flag rate (model-free)

30 images, threshold auto-calibrated from clean scores (p99 = `0.503`, clean FPR
3.3%). `fft_flag_rate` = fraction of attacked frames flagged; `clean_fpr` =
fraction of clean frames flagged per defense.

| Defense        | none   | patch  | physical | universal | lowfreq | clean_fpr |
|----------------|--------|--------|----------|-----------|---------|-----------|
| none           | 0.03   | 1.00   | 1.00     | 0.30      | **0.60**| 0.03      |
| jpeg           | 0.37   | 1.00   | 0.80     | 0.40      | 0.30    | 0.37      |
| blur           | 0.07   | 0.00   | 0.00     | 0.00      | 0.03    | 0.07      |
| bitdepth       | 1.00   | 1.00   | 1.00     | 1.00      | 1.00    | **1.00**  |
| featuresqueeze | 1.00   | 1.00   | 1.00     | 1.00      | 1.00    | **1.00**  |
| fft            | 0.03   | 1.00   | 1.00     | 0.30      | 0.77    | 0.03      |
| fft_jpeg       | 0.03   | 1.00   | 1.00     | 0.30      | 0.67    | 0.03      |

## 3. Findings

1. **Low-frequency attacks evade FFT.** With no defense, `lowfreq` is flagged
   only 60% of the time (0% for stronger low-freq noise in the failure
   analysis). High-frequency attacks (patch, physical) are caught at 100%.
2. **Spectral defenses are incompatible with the FFT cue.** `bitdepth` and
   `featuresqueeze` push clean FPR to 100% — they inject high-frequency
   artifacts that make every frame look adversarial. They are unsafe to pair
   with FFT-based detection as-is.
3. **`blur` masks everything** — including real attacks (0% flag rate) — it is
   a latency sink, not a detector.
4. **Natural high-frequency texture (foliage) causes false positives** at ~42%
   when the threshold is calibrated on clean sky.

## 4. Mitigations (evaluated model-free)

### Temporal-consistency cue (`MODEL/temporal_scorer.py`) — closes the low-freq gap

A non-frequency detector: EMA background model + block-wise residual. On the
same attack set where the FFT detected nothing:

| Cue     | low_freq_noise | universal_low_freq | clean FPR |
|---------|----------------|--------------------|-----------|
| FFT     | 0.000          | 0.000              | 0.025     |
| temporal| **1.000**      | **0.675**          | **0.000** |

Fusing the two cues closes the failure mode the FFT is blind to.

### Per-region (ROI-crop) scoring — does NOT fix foliage FPR

With a region-calibrated threshold, the center-crop FPR is *worse* (foliage
0.95 vs 0.42 whole-frame): the synthetic foliage texture fills the crop, so
background-vs-object separation doesn't help. The effective fix for foliage is
**challenging-clean calibration** (mitigation below), not region scoring — an
honest negative result.

### Recommended fusion

- `fft + jpeg` keeps clean FPR ~3% while catching patch/physical at 100%.
- Add the temporal residual to catch low-frequency attacks (1.0 / 0.675).
- Calibrate per environment including "challenging clean" frames (foliage, rain).

## 5. Reproduce

```bash
# failure analysis (detector-free; temporal cue + ROI-crop FPR included)
python -m MODEL.failure_analysis --n 40 --seed 0 --region 64

# benchmark matrix (threshold auto-calibrated from clean images at p99)
python -m MODEL.benchmark --images <images> --labels <labels> \
    --output runs/benchmark --n-images 30 --seed 42

# mAP matrix requires torch/ultralytics (GPU-gated); weights in MODEL/*.pt
python -m MODEL.benchmark --images <images> --labels <labels> \
    --detector MODEL/yolov8n.pt --output runs/benchmark --n-images 30
```
