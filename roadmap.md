# KAAL — Project Roadmap

> From hackathon prototype to production-grade adversarial defense system.

---

## Phase 0 — Cleanup & Foundation (Week 1–2)

**Goal:** Turn a hackathon repo into a maintainable codebase.

### Codebase Surgery
- [x] Delete duplicate pipeline (`main.py`, `training/`, `evaluation/`, `visualization/`, `detection/`, `attacks/`, `defenses/`) — `main.py` deleted; the 6 subdirectories kept as-is by decision (safest, no data loss). Their `sys.path` hacks were removed; `attacks/`/`defenses/` still referenced by Phase 3 benchmarks
- [x] Keep `pipeline.py` as the single entry point, backed by `trainer.py`, `evaluator.py`, `visualizer.py`
- [x] Remove empty `MainDashboard/` directory
- [x] Move `kaal_swarm.py` out of `dashboard-app/src/` into `SCRIPTS/`

### Packaging
- [x] Add `pyproject.toml` — make KAAL an installable Python package (console script `kaal-pipeline`; run as `python -m MODEL.pipeline`)
- [x] Eliminate all `sys.path.insert()` hacks — use proper relative imports (also in `SCRIPTS/simulated_eye.py`)
- [x] Add `MODEL/__init__.py`
- [x] Split requirements: `requirements.txt` (core ML), `requirements-sim.txt` (AirSim/pynput), `requirements-server.txt` (FastAPI/uvicorn)

### Bug Fixes (from audit)
- [x] Fix Gaussian noise uint8 wrap in `augmentations.py:58`
- [x] Fix async `time.sleep()` → `asyncio.sleep()` in `server.py:164`
- [x] Fix resume no-op in `pipeline.py:48-51`
- [x] Fix undefined `h` in `simulated_eye.py:130`
- [x] Fix missing `rotate_to_yaw` in `virtual_drone_controller.py:116`
- [x] Remove duplicate `hover()` in `virtual_drone_controller.py`
- [x] Remove all hardcoded Windows paths (`kaal_integrated_radar.py`, `test_model.py`)

### Hygiene
- [x] Replace all `print()` with `logging` module calls (across MODEL + SCRIPTS + backend; telemetry `json.dumps` stdout stream in swarm scripts intentionally kept)
- [x] Add `package-lock.json` back to git tracking
- [x] Align `MODEL/README.md` CLI docs with actual `pipeline.py` interface

---

## Phase 1 — Harden the Defense (Week 3–5)

**Goal:** Make the FFT dual-stream defense rigorous and benchmarkable.

### Adaptive Thresholding
- [x] Implement calibration mode: run FFT scorer on N clean images, compute score distribution
- [x] Set threshold at 99th percentile (or configurable σ) of clean distribution
- [x] Store calibration profile per camera/environment — `MODEL/calibration.py` (JSON profiles, CLI `python -m MODEL.calibration`)
- [x] Replace the hardcoded `fft_threshold: 0.3` with calibrated values — now the fallback; `--calibration` overrides

### Per-Region FFT
- [x] Switch default from whole-image FFT to per-bbox crop FFT
- [x] `compute_anomaly_score_region()` already exists — wire it into the main `predict()` path
- [x] Score each detection independently — flag individual objects, not entire frames
- [x] Handle edge cases: tiny boxes (<16px) skipped as unscored, overlapping boxes reuse strongest score

### Stronger Adversarial Augmentation
- [x] Implement gradient-based patch optimization (not just random noise) — `PatchType.GRADIENT` (torch PGD vs detector, or low-freq FFT-aware fallback)
- [x] Add perspective transforms to simulate physical-world patch placement
- [x] Add lighting/exposure variation to patches
- [x] Expand `patch_scale_range` to cover 5%–50% of bbox
- [x] Implement low-frequency adversarial perturbations (to test FFT failure cases) — `add_low_freq_noise()` + low-freq gradient patch

### Ablation Study
- [x] Sweep `high_freq_ratio`: 0.1, 0.15, 0.2, 0.25, 0.3, 0.35
- [x] Sweep `fft_threshold`: measure precision/recall/F1 at each
- [x] Sweep `patch_scale_range`: detection rate vs patch size
- [x] Sweep `patch_alpha`: detection rate vs patch opacity
- [x] Document all curves — these become paper figures — `MODEL/ablation.py` exports CSV/JSON/PNG

---

## Phase 2 — Testing & CI (Week 4–6)

**Goal:** Prove the system works, catch regressions automatically.

### Unit Tests
- [x] `test_anomaly_score.py` — clean images score low, patched images score high
- [x] `test_parse_yolo_labels.py` — empty files, malformed lines, normalized edge cases
- [x] `test_adversarial_patch.py` — patch placement within bbox, alpha blending correctness
- [x] `test_augmentations.py` — noise in valid range, blur doesn't crash on small images
- [x] `test_config.py` — validation catches bad values

### Integration Tests
- [x] End-to-end: train 1 epoch → evaluate → check report file exists and has valid numbers (GPU-gated, `-m gpu`)
- [x] Dual-stream: run `predict()` on 10 clean + 10 patched images, assert detection rate > 80% (calibration-derived threshold, not a magic 0.3)
- [x] Server: start FastAPI, hit `/api/python-data`, verify JSON schema (plus streaming `/api/video_feed` route check)

### CI Pipeline
- [x] GitHub Actions workflow: lint (ruff) + type check (mypy) + unit tests on every push — `.github/workflows/ci.yml`
- [x] Separate workflow for integration tests (needs GPU — runs on self-hosted/gpu runner, skip otherwise) — `.github/workflows/integration.yml`
- [x] Badge in README

---

## Phase 3 — Benchmarking & Baselines (Week 6–8)

**Goal:** Prove FFT defense is competitive against established methods.

### Implement Comparison Defenses
- [x] JPEG compression defense (already in `defenses/preprocess.py` — extract and benchmark independently) — `MODEL/benchmark.py`
- [x] Spatial smoothing (Gaussian blur at inference) — `MODEL/benchmark.py`
- [x] Bit-depth reduction (already implemented — benchmark independently) — `MODEL/benchmark.py`
- [x] Feature squeezing (combine multiple preprocessing defenses) — `MODEL/benchmark.py`

### Implement Comparison Attacks
- [x] PGD adversarial patches (gradient-optimized, not random) — `PatchType.GRADIENT` (Phase 1)
- [x] Physical-world patch simulation (perspective, lighting, print noise)
- [x] Universal adversarial perturbations (image-agnostic) — `generate_universal_perturbation`
- [x] Low-frequency adversarial perturbations (FFT-aware attacks) — `add_low_freq_noise`

### Benchmark Matrix
- [x] Build evaluation harness: every attack × every defense × mAP measurement — `MODEL/benchmark.py` (flag-rate matrix run model-free; mAP side GPU-gated)
- [ ] Produce comparison table: (needs a real YOLO detector — torch/ultralytics)

| Defense | Clean mAP | PGD mAP | Physical Patch mAP | Low-Freq mAP | Latency (ms) |
|---------|-----------|---------|---------------------|--------------|---------------|
| None | — | — | — | — | — |
| JPEG | — | — | — | — | — |
| Blur | — | — | — | — | — |
| FFT (ours) | — | — | — | — | — |
| FFT + JPEG | — | — | — | — | — |

### Failure Analysis
- [x] Document when FFT fails (low-freq perturbations, natural high-freq textures) — `MODEL/failure_analysis.py`, `MODEL/RESULTS.md`
- [x] Measure false positive rate on challenging clean images (foliage, urban clutter, rain)
- [x] Propose mitigations for identified failure modes — incl. **new `MODEL/temporal_scorer.py`** (non-frequency EMA-residual cue: low_freq detection 0.0→1.0, clean FPR 0.0) and ROI-crop FPR measurement (foliage 0.95 — negative result, foliage fills the crop; real fix is challenging-clean calibration)

---

## Phase 4 — Backend & Dashboard (Week 8–10)

**Goal:** Make the system deployable, not just demo-able.

### Backend Overhaul
- [x] Replace thread + global dict with asyncio producer-consumer + WebSocket — `Frontend_Backend/backend/server.py` (`camera_worker` → `asyncio.Queue` → `/ws`)
- [x] Add thread-safe data sharing (asyncio Queue or lock)
- [x] Real system metrics via `psutil` + `nvidia-smi` (graceful fallbacks)
- [x] Health check endpoint (`/api/health`)
- [x] Configuration via environment variables (`KAAL_MODEL_PATH`, `KAAL_CAMERA_INDEX`, `KAAL_FFT_THRESHOLD`, `KAAL_CONF`, `KAAL_ENABLE_CAMERA`, …)
- [x] Graceful shutdown handling (lifespan stop + worker join)
- [x] Error recovery on camera disconnect (exponential-backoff reconnect)

### Dashboard
- [x] WebSocket connection for real-time data (replace 1.5s polling) — `useDefenseSocket.js` + `/ws` vite proxy, polling fallback
- [x] Per-detection adversarial flag display (not just global noise score) — per-box `anomaly_score`/`is_adversarial` overlay
- [x] Historical anomaly score timeline — 60-point rolling `data` array (existing `GraphPanel`/`MultiGraph`)
- [x] Alert system: visual + audio on adversarial detection — banner + beep + red frame
- [x] Settings panel: adjust FFT threshold, confidence threshold live — `SettingsPanel.jsx` → `PUT /api/settings`

---

## Phase 5 — Edge Deployment (Week 10–14)

**Goal:** Run the defense on actual drone hardware.

### Performance Profiling
- [ ] Benchmark on Raspberry Pi Zero 2W: FPS with YOLO only vs YOLO + FFT
- [ ] Benchmark on Jetson Nano / Jetson Orin Nano (likely more realistic target)
- [ ] Profile memory usage — can YOLO + FFT fit in 512MB RAM?
- [ ] Identify bottleneck: is it YOLO inference or FFT computation?

### Model Optimization
- [ ] Export to ONNX / TensorRT for edge inference
- [ ] Quantize model (INT8) and re-benchmark mAP loss
- [ ] Test YOLOv8n vs YOLOv8s vs YOLOv8m: accuracy/speed tradeoff on target hardware
- [ ] Consider YOLO11n / YOLO26n (newer, faster architectures — you already have `yolo26n.pt`)

### FFT Optimization
- [ ] Use `cv2.dft()` instead of `np.fft.fft2()` (OpenCV DFT uses hardware acceleration)
- [ ] Downsample image before FFT (e.g., 160×120 instead of 640×480)
- [ ] Skip FFT on frames where YOLO detects nothing (no targets = no attack risk)
- [ ] Batch FFT computation across multiple bbox crops

---

## Phase 6 — Swarm Intelligence (Week 14–20)

**Goal:** Actually build the GPS-free coordination layer (only after Layer 1 is solid).

### Simulation Environment
- [ ] Set up multi-agent gym environment (PettingZoo or custom)
- [ ] Define observation space: own velocity, relative positions of K nearest neighbors
- [ ] Define action space: velocity vector (vx, vy, vz)
- [ ] Implement reward: formation maintenance + collision avoidance + target tracking

### MAPPO Training
- [ ] Implement MAPPO with centralized critic, decentralized actors
- [ ] Train on 4-drone, 8-drone, 16-drone scenarios
- [ ] Evaluate: formation recovery after 30% agent removal
- [ ] Evaluate: mission completion without GPS signal
- [ ] Compare against handcrafted boid rules (your current canvas simulation)

### Integration
- [ ] Swarm policy runs alongside Layer 1 defense
- [ ] If drone's camera is flagged adversarial → increase weight on neighbor-based navigation
- [ ] Shared threat map: drones broadcast adversarial flags to neighbors

---

## Phase 7 — Publication (Parallel with Phase 3+)

**Goal:** Get the FFT defense into a peer-reviewed venue.

### Paper Structure
1. **Problem:** Adversarial patches defeat drone object detection; existing defenses are compute-heavy or model-specific
2. **Contribution:** Lightweight, model-agnostic frequency-domain anomaly detector that runs alongside any detector with <5ms overhead
3. **Method:** FFT energy ratio scoring, per-region analysis, adaptive calibration
4. **Experiments:** Comparison table from Phase 3, ablation curves from Phase 1
5. **Limitations:** Low-frequency attacks, natural high-freq textures (honest failure analysis)

### Target Venues
- **Workshop:** CVPR/ECCV Adversarial ML Workshop
- **Conference:** AAAI, ICML (safety track), IEEE S&P (if security angle)
- **Journal:** IEEE TIFS, Pattern Recognition Letters (faster review cycle)

### Supporting Material
- [ ] Open-source the benchmark harness
- [ ] Release calibrated FFT thresholds for common cameras
- [ ] Provide pre-trained adversarially-robust YOLO weights

---

## Timeline Overview

```
Month 1    ██████████ Phase 0 (Cleanup) + Phase 1 (Defense)
Month 2    ██████████ Phase 2 (Testing) + Phase 3 (Benchmarks)
Month 3    ██████████ Phase 4 (Backend) + Phase 5 (Edge)
Month 4-5  ██████████ Phase 6 (Swarm) + Phase 7 (Paper)
```

> **Rule: No phase starts until the previous phase's checkboxes are >80% done.**
> The biggest risk is spreading thin again. Depth beats breadth.
