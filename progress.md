# KAAL — Progress Log

> Auto-updated during repo reorganization work.

---

## Status: ✅ COMPLETE (Phase 0–4) + ✅ Phase 4.5 (review fixes, RF-DETR, kaal_v3 dataset) + ✅ Phase 4.6 (model decision, data audit, kaal_v4, docs) + ✅ Phase 4.7 (model release on the Hugging Face Hub) + ✅ Phase 4.8 (v5 longer-schedule experiment — not promoted)

### Phase 4.8 — v5 longer-schedule experiment (2026-10-12) — not promoted

- [x] `train_detector --lr-scheduler step|cosine --lr-min-factor`, recorded in `kaal_train_summary.json`; `tests/test_train_detector.py`.
- [x] `rfdetr-small-v5`: warm start from v4, lr 5e-5 cosine, early-stopped at epoch 8 (best epoch 3), 1.87 h.
- [x] **Results** (`MODEL/RESULTS.md` §8): kaal_v4 test 0.405 → 0.412, but unseen ground vehicles 0.119 → 0.107 (APC/IFV 0.210 → 0.150, every source down).
- [x] **Not promoted:** v4 stays production. v5 is registered as `rejected` and is not published to the Hub.
- [x] **Finding:** more epochs on the same data only fit the training sources better. The next lever is data diversity (APC/IFV, artillery and engineering-vehicle scenes, DVIDS relabel) or a larger model.
- [x] **First attempt froze the PC:** 6 loader workers plus desktop apps exhausted the 14 GB of RAM; swap thrash caused NVMe read timeouts and systemd-oomd killed the session.
  - Recovery: integrity verified (git fsck; registry SHA-256s; all 38,886 kaal_v4 files read back), and the SSD's SMART report was clean.
  - Rerun: 3 workers and a memory-pressure guard. Free RAM never fell below 3.6 GB.
  - Documented in the WALKTHROUGH troubleshooting table.

### Phase 4.7 — Model release on the Hugging Face Hub (2026-10-12)

- [x] **`MODEL/publish_model.py`** (`kaal-publish`):
  - `strip` writes an inference-only release checkpoint. Training state and machine paths are removed, and it loads with `torch.load(weights_only=True)`.
  - `stage` checks every file against the registry SHA-256 and renders the Hub card.
  - `upload` creates the repo (private), turns on gating, commits, and tags.
- [x] **`MODEL/fetch_weights.py`** (`kaal-fetch-weights`): restores a published model into the registry paths. It downloads at the pinned Hub commit and checks each SHA-256 before moving a file into place. `--format onnx` is for boxes without torch.
- [x] `rfdetr-small-v4` release checkpoint (`MODEL/runs/release/rfdetr-small-v4/rfdetr-small-v4.pth`, SHA-256 `fb2a2f05…cc8f0`):
  - Predictions are identical to the training checkpoint: 100 test images, 21,714 detections, zero difference, on GPU and CPU.
  - No local paths remain in the `.pth`, the `.onnx` or the sidecar.
- [x] **Registry:** production `weights` now point to the release checkpoint, the training checkpoint is kept as provenance, and a `hub:` block (repo, tag, commit) is validated by a test.
- [x] **ONNX fallback:** `production_weights()` uses the ONNX file when it is the only one present.
- [x] **Hub model card** (`MODEL/hub/model_card_template.md`):
  - CC BY-NC 4.0, gated access with four conditions, `model-index` metrics.
  - Its ONNX Runtime and PyTorch quick-start code was executed against the release files and gives the same detections as KAAL's detectors (4 images, up to 93 detections each).
- [x] **Decision record:** `Docs/DECISIONS.md` ADR-008 (host, licence, gating, private-first).
- [x] **Published** [`ak-2468/kaal-rfdetr-small`](https://huggingface.co/ak-2468/kaal-rfdetr-small): uploaded private and gated (auto), made public by the owner after review; tag `v4` = commit `496fd0e9`. The Hub's SHA-256 values match the registry, and a real `fetch_weights` round trip was verified.
- [x] **Card metadata fix:** the Hub refuses card metadata keys containing `.` or `$`, so `render_card` now checks for them before any upload.

### Phase 4.6 — Production model decision, data audit, kaal_v4, documentation (2026-10-11)

**Model decision (`MODEL/RESULTS.md` §6, `Docs/DECISIONS.md` ADR-001)**
- [x] RF-DETR-Small vs YOLO26s trained on identical data: kaal_v3 test mAP50:95 **0.552 vs 0.517** (patched 0.516 vs 0.472); unchanged after removing 68 near-duplicate test images.
- [x] Independent ground-vehicle test (`MODEL/build_ground_test.py`, 798 unseen photos from Orion / Russian-military-annotated / TNO Search_2): any-class mAP50 **0.722 vs 0.015** — YOLO did not generalise.
- [x] **RF-DETR is the production detector**; YOLO kept as an optional backend. The backend now loads the registry's production weights by default (`MODEL.detectors.production_weights()`).

**Data audit — problems found and fixed**
- [x] Legacy val/test folders contain no labelled ground vehicles or soldiers (`merge_all.py` split the military set 90/10 by file order; the val slice is 181/198 unlabeled).
- [x] Legacy `mil_*` images are mostly near-identical video frames under unrelated numbers (1,977 → ~615 scenes); a name-based re-split leaked 36% of test images → flip-aware dHash near-duplicate grouping.
- [x] 254 unlabeled legacy frames were training as background → dropped.
- [x] Legacy `air_*` and the military-aircraft source share photos (124 cross-split pairs) → near-duplicate grouping across **all** sources before splitting.

**kaal_v4 and training**
- [x] New sources: DVIDS military photos (`llama-farm/military-labeled-yolo`, public domain / CC0 — soldiers, tanks, APCs, artillery, MLRS) and Open Images V7 tank photos (CC BY; photos with people or group-of boxes excluded); all 6,471 VisDrone train images.
- [x] `MODEL/datasets_registry.yaml` — every dataset's licence, status and use per model, validated by a test; finished/redundant data moved to the trash (experimental `kaal_v3_1`, extracted archives, Open Images CSV, resume checkpoints).
- [x] `train_detector --init-weights` (warm start), `compare_detectors --dedup-against / --eval-classes / --agnostic`, ignore regions, per-source tables.
- [x] RF-DETR-Small v4 fine-tune on kaal_v4 (warm start, 10 epochs, 2.43 h) — **production**: kaal_v4 test mAP50:95 0.313 → **0.405**, unseen ground vehicles any-class mAP50 0.720 → **0.915**, soldier 0.00 → 0.30, artillery 0.07 → 0.26 (`MODEL/RESULTS.md` §7, `Docs/MODEL_CARD.md`).
- [x] Deployment: ONNX export with parity gate (v4 Δ +0.005 mAP50:95 on 200 val images), ONNX Runtime backend verified live without torch; attack demo with v4 flags the patch 4 frames after onset with no false alarms.

**Documentation (`Docs/`)** — WALKTHROUGH, FILE_REFERENCE, DATASETS, DECISIONS, MODEL_CARD, FUTURE_SCALING.


### Phase 4.5 — Review fixes, RF-DETR detector, cleaned dataset (2026-10-11)

**Bugs fixed (from the repo review)**
- [x] **Live dashboard froze after the first message** — `CameraWorker._emit` scheduled an `async def` via `call_soon_threadsafe` (coroutine never awaited, queue never filled). Now enqueues directly with drop-oldest; regression test added.
- [x] **FFT score tracked box shape, not content** — rewritten as a Hann-windowed power ratio in cycles/pixel. Real-photo AUROC 0.59 → 0.89 (noise patch), 0.53 → 0.85 (pattern). One shared `DEFAULT_FFT_THRESHOLD`. See `MODEL/RESULTS.md` §0.
- [x] **Fresh clone could not build the dashboard** — unanchored `lib/` in `.gitignore` hid `dashboard-app/src/lib/`.
- [x] **Confidence slider was a no-op** — live `conf` now reaches the detector every frame.
- [x] **Unauthenticated control plane** — default bind 127.0.0.1, CORS + WebSocket origin allow-list, optional `KAAL_API_TOKEN`, Pydantic validation (NaN/inf → 422).
- [x] **Metrics** — nvidia-smi throttled to 1 Hz (was spawned every frame), real latency, bandwidth over real elapsed time and labelled KB/s; dashboard `alive`/`kia` no longer overwritten by the flag count; fake random "CPU LOAD" replaced.
- [x] **Evaluator** — attacked and defended passes now use identical (seeded) patches; the misleading "mAP recovery" claim removed (the defense flags, it does not correct).
- [x] **CI** — server tests actually run (server extra installed), torch no longer installed for CPU tests (`ml` extra), Python 3.11 + 3.12 matrix, backend linted + type-checked, new dashboard job (eslint + vite build), GPU workflow manual-only.
- [x] Dashboard ESLint 16 errors → 0; unused npm deps removed (axios, clsx, tailwind-merge); duplicate `SCRIPTS/b.py` and empty `ui/dottedglowback` removed; READMEs merged.

**New**
- [x] `MODEL/detectors.py` — YOLO + **RF-DETR** (ICLR 2026, Apache-2.0) behind one interface; used by the defense, server (`KAAL_DETECTOR`, `KAAL_MODEL_PATH`) and benchmarks.
- [x] Temporal-cue fusion — `StreamingTemporalCue`: per-cell residuals with illumination normalization, warm-up calibration, persistence (moving objects ignored), masked background updates, local absorb / global re-baseline. Wired into `DualStreamDefense.predict()` and the live server; shown on the dashboard. Verified live: a printed-noise patch on a tank (≈0.5% of the frame) → cell residual 0.284 vs threshold 0.01; the frame-mean version had missed it (0.00097 vs 0.00121).
- [x] Live demo without a webcam — `KAAL_CAMERA_SOURCE` (looping video file / stream URL) + `python -m MODEL.make_demo_video` (clean → patch → clean clip). Server now keeps the aspect ratio (was a fixed 640×480 squash); dashboard WebSocket auto-reconnects with backoff (was stuck on polling after a backend restart).
- [x] `MODEL/fetch_datasets.py`, `MODEL/build_dataset.py` — `kaal_v3`: legacy set (corrupt class 0 dropped, split-leakage removed) + VisDrone2019-DET + military-aircraft subset + Simuletic UAV tanks, 9-class taxonomy, `manifest.json` with counts and licenses.
- [x] `MODEL/train_detector.py`, `MODEL/compare_detectors.py` — matched RF-DETR vs YOLO26 training and a clean + under-attack COCO mAP / latency / FFT-flag comparison.
- [x] Tests: 116 → 160 passing.

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
- **Phase 5 (roadmap):** Deployment & Hardening — Dockerize, JSON structured logging, `/metrics`, ONNX export, security hardening

---

*Started: 2026-08-12 23:55 IST*
*Last updated: 2026-08-13 01:10 IST*
*Updated: 2026-08-13 (Phase 3 — benchmarking & failure analysis in `MODEL/RESULTS.md`)*
*Updated: 2026-08-13 (Phase 3 follow-up — temporal-consistency cue + ROI-crop FPR; results in `MODEL/RESULTS.md`)*
*Updated: 2026-08-13 (Phase 4 — backend rewrite: asyncio producer-consumer, WebSocket `/ws`, `/api/health`, live `/api/settings`, psutil + nvidia-smi metrics; dashboard WS with fallback, alert system, per-detection flags, settings panel; 4 new integration tests)*
