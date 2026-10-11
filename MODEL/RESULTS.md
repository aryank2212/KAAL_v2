# KAAL Phase 3 — Benchmarking & Failure Analysis Results

Generated with `MODEL/failure_analysis.py` and `MODEL/benchmark.py` on a synthetic
evaluation set (blurred-noise scenes with a synthetic object box, YOLO labels).
Artifacts: `runs/failure_analysis/`, `runs/benchmark/`.

> **Note (2026-10-11):** Sections 1–4 below were measured with the original
> `log1p`-magnitude FFT scorer, which was later found to track crop *shape*
> rather than content (§0). They are kept for the record; §0 and §6 supersede
> them. Re-run with the commands in §5.

## 0. FFT scorer fix — shape bias removed (2026-10-11)

The original score (`log1p|F|` energy outside a circle of radius
`min(h, w)·(1−r)/2` in index space) had two defects:

* **Shape bias.** Same photo, same centre, different crop shape:

  | Crop | 240×240 | 480×120 | 120×480 | 600×100 | 640×480 |
  |------|---------|---------|---------|---------|---------|
  | old score | 0.496 | 0.857 | 0.866 | 0.905 | 0.607 |
  | new score | 0.074 | 0.069 | 0.069 | 0.068 | 0.079 |

  Per-box scoring therefore flagged every tall/wide detection (people,
  vehicles) at the old server default of 0.5.
* **No content signal.** At a fixed 240×240 size the old score was 0.532 for a
  smooth blur, 0.496 for a real photo and 0.558 for pure white noise — the
  last equals the pixel-area fraction of the band (0.558): `log1p` flattened
  the spectrum until the ratio was essentially geometry.

The new score is the share of AC spectral **power** above
`0.5·(1−high_freq_ratio)` cycles/pixel, computed on a mean-removed,
Hann-windowed crop (smooth 0.009 · photo 0.074 · white noise 0.555).

**Real photos** (44 photos, 264 random crops 80–400 px of any aspect, the
repo's own noise / pattern patch injector):

| Scorer | AUROC vs noise patch | AUROC vs pattern patch | clean p99 |
|---|---|---|---|
| old (`log1p`, r=0.25) | 0.591 | 0.533 | 0.857 |
| **new (power, r=0.25)** | **0.890** | **0.846** | 0.086 |
| new, r=0.40 | 0.876 | 0.839 | 0.160 |
| new, r=0.50 | 0.859 | 0.824 | 0.250 |
| new, r=0.75 | 0.798 | 0.801 | 0.518 |

`high_freq_ratio=0.25` stays the default; the fallback threshold
`DEFAULT_FFT_THRESHOLD` is now one shared constant (previously 0.3 in the
model and 0.5 in the server).

**Synthetic failure analysis, re-run with the new scorer**
(`python -m MODEL.failure_analysis --n 40 --seed 0 --region 64`, threshold =
p99 of smooth "sky"):

| | old scorer | new scorer |
|---|---|---|
| gaussian_noise_high_freq detection | 0.200 | **1.000** |
| low_freq_noise / universal_low_freq | 0.000 / 0.000 | 0.075 / 0.000 |
| FPR foliage / urban / rain | 0.43 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 |
| temporal cue: low_freq / universal (clean FPR) | 1.00 / 0.68 (0.00) | 1.00 / 0.65 (0.00) |

Read this carefully: the synthetic "foliage" is per-pixel random speckle and
"urban"/"rain" are 1-px aliased lines — spectrally they *are* noise patches,
with power right up to Nyquist that real optics/demosaicing/JPEG remove. The
old scorer only looked better on them because it barely separated anything.
The honest conclusion is the one §4 already drew: calibrate per camera with
challenging clean frames, score per box, and fuse the temporal cue. The
decisive measurement is per-box scoring on real imagery (§6).

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

## 6. Detector comparison — RF-DETR vs YOLO26 on `kaal_v3` (2026-10-11)

*Pending: the overnight run is in progress (`MODEL/runs/train/pipeline.log`).*
Protocol: both detectors fine-tuned from COCO weights on the same `kaal_v3`
train split (≈11.6k images, 9 classes) with effective batch 16 —
RF-DETR-Small for 12 epochs, YOLO26s for 30 epochs (≈ equal GPU time on an
RTX 4060 Laptop GPU, 8 GB). Evaluated by `python -m MODEL.compare_detectors`
on the held-out test split, clean and with a noise/pattern patch inside every
ground-truth box; full tables in `runs/compare/report.md`.

## 5. Reproduce

```bash
# failure analysis (detector-free; temporal cue + ROI-crop FPR included)
python -m MODEL.failure_analysis --n 40 --seed 0 --region 64

# benchmark matrix (threshold auto-calibrated from clean images at p99)
python -m MODEL.benchmark --images <images> --labels <labels> \
    --output runs/benchmark --n-images 30 --seed 42

# detector comparison (needs the `ml` extra + a GPU)
python -m MODEL.build_dataset
python -m MODEL.train_detector --arch rfdetr-small --epochs 12
python -m MODEL.train_detector --arch yolo26s --epochs 30
python -m MODEL.compare_detectors \
    --detector rfdetr-small=MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth \
    --detector yolo26s=MODEL/runs/train/yolo26s/weights/best.pt

# mAP matrix requires torch/ultralytics (GPU-gated); weights in MODEL/*.pt
python -m MODEL.benchmark --images <images> --labels <labels> \
    --detector MODEL/yolov8n.pt --output runs/benchmark --n-images 30
```
