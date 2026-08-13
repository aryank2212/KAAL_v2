# KAAL — Progress Log

> Auto-updated during repo reorganization work.

---

## Status: ✅ COMPLETE (Phase 0) + ✅ COMPLETE (Phase 1) + ✅ COMPLETE (Phase 2) + ✅ COMPLETE (Phase 3) + ✅ COMPLETE (Phase 4)

### Phase 4 — Backend & Dashboard

**Goal:** Real-time adversarial defense streaming with secure, observable control.

- [x] **Backend overhaul** — `Frontend_Backend/backend/server.py` rewritten: asyncio producer-consumer (camera worker thread → `asyncio.Queue` → `/ws` WebSocket broadcast), replacing the thread + global-dict design.
- [x] **Thread-safe data sharing** — `asyncio.Queue` for snapshots + `threading.Lock` around the live-settings dict.
- [x] **Real system metrics** — `psutil` (CPU/mem/bandwidth) + `nvidia-smi` (GPU util/vRAM) with graceful fallbacks when unavailable.
- [x] **Health check** — `GET /api/health` (liveness, mode, camera/model status, uptime).
- [x] **Env config** — `KAAL_MODEL_PATH`, `KAAL_CAMERA_INDEX`, `KAAL_FFT_THRESHOLD`, `KAAL_CONF`, `KAAL_CALIBRATION`, `KAAL_ENABLE_CAMERA`, `KAAL_LOG_LEVEL`, `KAAL_BROADCAST_INTERVAL`.
- [x] **Graceful shutdown** — lifespan stop event + worker thread join.
- [x] **Camera reconnect** — exponential backoff with live camera status surfaced in snapshots.
- [x] **Per-detection adversarial flags** — each box carries `anomaly_score` + `is_adversarial` via `MODEL/fft_scorer.compute_region_scores`; red overlays + `flagged_count` on the stream.
- [x] **Live settings** — `GET/PUT /api/settings` (FFT threshold + detector confidence, clamped, applied without restart).
- [x] **Dashboard WS** — `src/lib/useDefenseSocket.js` replaces 1.5s polling (WS with automatic fallback); `vite.config.js` proxies `/ws`.
- [x] **Alert system** — flashing banner + audio beep + red frame on adversarial detection (`App.jsx`, `MainPanel.jsx`).
- [x] **Per-detection readout** — flagged labels + anomaly scores overlaid on the mission feed.
- [x] **Settings panel** — `SettingsPanel.jsx` sliders → `PUT /api/settings`.
- [x] **Status bar** — NOISE vs THRESH, FLAGS count, WS connection badge (`StatusBar.jsx`).
- [x] Verification: 4 new server integration tests; **116 passed, 1 skipped, 1 warning**; ruff + mypy clean; live smoke test against the real `nvidia-smi` GPU metrics.

#### Phase 4 notes

- Real YOLO inference requires `torch` + `ultralytics` at runtime (stubbed in tests; not installed on this box). The server still boots, serves snapshots, and streams the annotated MJPEG when the camera pipeline is unavailable.
- Server dependencies in `requirements-server.txt` (now incl. `psutil`, `websockets`); both degrade gracefully if absent.
- `gpu` is `None` until the worker thread runs a frame with a live camera; `_gpu_metrics()` populates it via `nvidia-smi`.

### Phase 3 — Benchmarking & Failure Analysis

- [x] **NEW `MODEL/benchmark.py`** — attack x defense benchmark matrix. Comparison defenses: none/jpeg/blur/bit-depth/feature-squeeze/fft/fft_jpeg; comparison attacks: none/patch/physical/universal/lowfreq. Model-free `fft_flag_rate` + `clean_fpr` rows, plus optional mAP matrix when a YOLO detector is supplied
- [x] **NEW `MODEL/failure_analysis.py`** — runs detector-free; calibrates threshold on a clean baseline texture, measures FPR across natural textures (sky/foliage/urban/rain), measures attack detection rate, and emits a structured failure-mode + mitigation report (JSON + Markdown)
- [x] **NEW `MODEL/RESULTS.md`** — full Phase 3 results and findings (see below)
- [x] Threshold auto-calibration in the benchmark: `fft_threshold` is `Optional`; when unset it is calibrated at p99 from the clean sample scores (`FFTCalibrator`), mirroring the real Phase 1 → Phase 2 flow
- [x] Fixed mypy issues surfaced by Phase 3 work: `_jpeg_compress` decode-None guard, `matched` annotation, `_load_samples` typing, `report` dict annotation

#### Phase 3 findings (synthetic eval set, n=30–40)

- **Low-frequency attacks evade the FFT detector.** `none` defense flags `lowfreq` only 60%; in the failure analysis, `low_freq_noise` and `universal_low_freq` detection rates are 0.0%. High-frequency attacks (patch/physical) are caught at 100%.
- **Spectral defenses break the FFT cue.** `bitdepth` and `featuresqueeze` force clean FPR to 100% (they inject high-frequency artifacts) — unsafe to pair with FFT detection as-is.
- **Foliage FPR ~42%.** Natural high-frequency texture is flagged at a threshold calibrated on clean sky.
- **`fft + jpeg` is the best pairing:** clean FPR ~3% while keeping patch/physical at 100% and lifting lowfreq to 67%.

#### Phase 3 follow-up — mitigation evaluation (model-free)

- [x] **NEW `MODEL/temporal_scorer.py`** — `TemporalConsistencyDetector`: EMA background model + block-wise residual, a non-frequency cue that flags persistent per-patch perturbations
- [x] `failure_analysis.py` now runs the temporal cue on the same low-freq attack set and reports it (`temporal_cue` in report.json/md)
- [x] `failure_analysis.py --region N` measures FPR on a centered ROI crop with a *region-calibrated* threshold
- [x] Unit tests: `tests/test_temporal_scorer.py` (9 tests) + region-FPR test in `tests/test_failure_analysis.py`
- [x] **Temporal cue closes the low-freq gap:** low_freq_noise detection **0.000 → 1.000**, universal_low_freq **0.000 → 0.675**, clean FPR 0.000
- [x] **Honest negative result:** per-region scoring does NOT fix foliage FPR (0.95 crop vs 0.42 whole-frame with region-calibrated threshold) — foliage fills the crop; the real fix is challenging-clean calibration

#### Phase 2 — Testing & CI

#### Unit Tests
- [x] `tests/unit/test_anomaly_score.py` — clean vs patched scoring, region scoring, high-frequency content detection
- [x] `tests/unit/test_parse_yolo_labels.py` — empty/malformed files, normalized coordinates, dedup
- [x] `tests/unit/test_adversarial_patch.py` — patch placement inside bbox, alpha blending correctness, scale clamp
- [x] `tests/unit/test_augmentations.py` — noise in valid uint8 range, blur on tiny images, low-freq noise registered
- [x] `tests/unit/test_config.py` — validation rejects bad values, defaults sane

#### Integration Tests
- [x] `tests/integration/test_end_to_end.py` — train 1 epoch → evaluate → report file exists with valid mAP; **GPU-gated** (`-m gpu`, `KAAL_RUN_GPU_TESTS=1`)
- [x] `tests/integration/test_dual_stream.py` — full `predict()` path (FakeYOLO + real FFT) on 10 clean + 10 patched; **calibration-derived threshold**, detection rate > 80%, zero clean false positives; tiny detections unscored
- [x] `tests/integration/test_server.py` — FastAPI `/api/python-data` JSON schema + types; `/api/video_feed` streaming route (bounded chunk read, doesn't hang on the infinite stream)

#### CI Pipeline
- [x] `.github/workflows/ci.yml` — `ruff check` + `ruff format --check` + `mypy` + `pytest -m "not gpu"` on every push (python 3.11 + 3.12 matrix)
- [x] `.github/workflows/integration.yml` — full suite incl. GPU tests, gated to self-hosted/gpu runner or opt-in
- [x] CI badge in `README.md` (repo `aryank2212/KAAL`)
- [x] `pyproject.toml`: `[tool.ruff]` line-length 100, E501 ignored (enforced by formatter); `[tool.mypy]` excludes legacy dirs; `[tool.pytest.ini_options]` skips gpu by default

#### Fixes made while stabilizing the suite
- [x] Ruff 0.16 ignores `lint.ignore` from pyproject — moved `ignore = ["E501"]` to top-level `[tool.ruff]`
- [x] Dual-stream tests used a fixed 0.3 threshold → calibrated from clean distribution (matches Phase 1 flow; synthetic noise images legitimately score ~0.52, patched ~0.56)
- [x] Server video-feed test hung on the infinite stream → assert route registration + read bounded chunks of `video_feed()` with a seeded frame
- [x] Local `ruff format --check` tripped on phantom `MODEL/pytest.ini` / `MODEL/analysis.py` directory entries on the external USB drive (files don't exist per `stat`/`find`) — an environment artifact, not repo code

### Phase 1 — Harden the Defense

#### Adaptive Thresholding
- [x] **NEW `MODEL/calibration.py`** — `FFTCalibrator`: scores N clean images, computes distribution (mean/std, p50/p95/p99/p99.9), derives threshold via `percentile` (default 0.99) or `sigma` (mean + k·std), saves/loads per-environment JSON profiles
- [x] CLI: `python -m MODEL.calibration --images ... --output profile.json`
- [x] `Config.fft_threshold` now `Optional`; `fft_default_threshold: 0.3` used only as fallback; `--calibration` wires a profile into the defense via `load_calibrated_threshold()`

#### Per-Region FFT
- [x] **`MODEL/fft_scorer.py`** — standalone model-free `compute_fft_anomaly_score()` + `compute_region_scores()` (so calibration/ablation match inference exactly)
- [x] `DualStreamDefense.predict()` now scores each detection on its own bbox crop by default (`region_mode=True`); per-detection `anomaly_score` + `is_adversarial` fields
- [x] Edge cases: boxes < `region_min_size` (16px) → `region_scored: False`, score `None`; overlapping boxes (IoU ≥ 0.5) reuse strongest box score
- [x] `predict()` returns `objects_flagged`; frame flagged if whole-image OR any object flagged
- [x] Visualizer: flagged objects drawn red with `fft=` label

#### Stronger Adversarial Augmentation
- [x] **`PatchType.GRADIENT`** — gradient-optimized patch: torch PGD vs detector (suppress objectness) when a model is attached; else derivative-free **low-frequency** perturbation (smooth field) to test FFT failure cases
- [x] Perspective warp (`_apply_perspective`) + lighting/exposure/gamma (`_apply_lighting`) applied to all patch types (on by default)
- [x] `patch_scale_range` default expanded to **(0.05, 0.5)** = 5%–50% of bbox
- [x] `augmentations.add_low_freq_noise()` — smooth low-frequency perturbation; registered as `lowfreq` attack in `apply_random_attack`

#### Ablation Study
- [x] **NEW `MODEL/ablation.py`** — `AblationStudy` sweeps `high_freq_ratio` (0.1–0.35), `fft_threshold` (precision/recall/F1), `patch_scale_range`, `patch_alpha`; exports `.csv` + `.json` + `.png` curves (matplotlib)
- [x] CLI: `python -m MODEL.ablation --images ... --labels ... --output runs/ablation`

### Phase 0 — Completed (see below)

### Completed
- [x] Deleted duplicate entry point `main.py`
- [x] Removed empty `MainDashboard/` directory
- [x] Moved `kaal_swarm.py` from `dashboard-app/src/` → `SCRIPTS/`
- [x] Added `MODEL/__init__.py` with module docstring
- [x] **BUG FIX:** Gaussian noise uint8 wrap in `augmentations.py`
- [x] **BUG FIX:** Async `time.sleep()` → `asyncio.sleep()` in `server.py`
- [x] **BUG FIX:** Added thread lock around `latest_data` in `server.py`
- [x] **BUG FIX:** Resume no-op in `pipeline.py` — added `resume` field to Config, wired through trainer
- [x] **BUG FIX:** Undefined `h` in `simulated_eye.py`
- [x] **BUG FIX:** Added `rotate_to_yaw()` to `virtual_drone_controller.py`
- [x] **BUG FIX:** Removed duplicate `hover()` in `virtual_drone_controller.py`
- [x] Removed hardcoded Windows path in `kaal_integrated_radar.py`
- [x] Removed 3 hardcoded Windows paths in `test_model.py`
- [x] Moved stray `import os` to top of `server.py`
- [x] Created `requirements-sim.txt` (airsim, pynput)
- [x] Created `requirements-server.txt` (fastapi, uvicorn)
- [x] Fixed `.gitignore` — stopped ignoring `package-lock.json`
- [x] **LOGGING:** Migrated `trainer.py` — all print() → logger.info/warning
- [x] **LOGGING:** Migrated `dataset.py` — all print() → logger.info
- [x] **LOGGING:** Migrated `evaluator.py` — all print() → logger.info/warning (report refactored to return text, dropped stdout-capture in `save_report`)
- [x] **LOGGING:** Migrated `pipeline.py` — all print() → logger + added basicConfig
- [x] **LOGGING:** Migrated `visualizer.py` — all print() → logger.info/warning
- [x] **LOGGING:** Migrated remaining `print()` across MODEL utils & SCRIPTS (merge_all, merge_datasets, analyze_dataset, validate_dataset, evaluate_custom_report, inference_quality_fix, test_model, use_model, user_test, evaluate.py, anomaly.py, compare.py, simulated_eye.py, virtual_drone_controller.py, kaal_fft_radar, kaal_integrated_radar, server.py). Kept `print(json.dumps(...))` telemetry in swarm scripts (stdout protocol for dashboard).
- [x] **PACKAGING:** Added `pyproject.toml` — installable `kaal` package with `kaal-pipeline` console script; `MODEL` core modules converted to relative imports; scripts run via `python -m MODEL.pipeline`
- [x] **PACKAGING:** Eliminated all `sys.path` hacks (`evaluation/evaluate.py`, `training/adv_train.py`, `SCRIPTS/simulated_eye.py` → relative/sibling imports)
- [x] Updated `MODEL/README.md` — CLI docs verified against `pipeline.py` (all 9 flags match); usage examples updated to `python -m MODEL.pipeline`

### Queued
- **Phase 3 mAP matrix (GPU-gated):** re-run the attack x defense matrix with a real YOLO detector (`MODEL/*.pt`) to fill the roadmap mAP comparison table; requires torch/ultralytics
- **Fusion deployment:** wire the temporal cue into `DualStreamDefense.predict()` alongside the FFT region scoring
- **Phase 5 (roadmap):** Deployment & Hardening — Dockerize, JSON structured logging, `/metrics`, ONNX export, security hardening

---

*Started: 2026-08-12 23:55 IST*
*Last updated: 2026-08-13 01:10 IST*
*Updated: 2026-08-13 (Phase 3 — benchmarking & failure analysis in `MODEL/RESULTS.md`)*
*Updated: 2026-08-13 (Phase 3 follow-up — temporal-consistency cue + ROI-crop FPR; results in `MODEL/RESULTS.md`)*
*Updated: 2026-08-13 (Phase 4 — backend rewrite: asyncio producer-consumer, WebSocket `/ws`, `/api/health`, live `/api/settings`, psutil + nvidia-smi metrics; dashboard WS with fallback, alert system, per-detection flags, settings panel; 4 new integration tests)*
