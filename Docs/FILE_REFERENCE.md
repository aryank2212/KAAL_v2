# KAAL — File Reference

Every tracked file, what it does, and its status. Companion to
[`WALKTHROUGH.md`](WALKTHROUGH.md) (how the pieces fit) and
[`DATASETS.md`](DATASETS.md) (data lineage).

**Status labels**

| Label | Meaning |
|---|---|
| **core** | On the production path (detector → defense → backend → dashboard). Tested in CI. |
| **tool** | Command-line tool used to build data, train, evaluate or demo. Tested in CI. |
| **research** | Phase 1–3 experiment harnesses (ablation, benchmark, failure analysis). Tested in CI, not on the live path. |
| **legacy** | Pre-Phase-0 scripts kept for reference. Not imported by anything active; several contain hard-coded Windows paths (`D:\...`) and will not run unmodified. Candidates for removal. |
| **test** | pytest suite. |
| **config** | Packaging, CI, dependency pins. |
| **doc** | Documentation. |

---

## Repository root

| File | Status | Purpose |
|---|---|---|
| `README.md` | doc | Project overview, status table, install, backend env vars, dataset / training / evaluation commands. |
| `progress.md` | doc | Chronological log of completed phases (0 → 4.5+) with measured outcomes. |
| `roadmap.md` | doc | Phase plan (0–7) with checkboxes; Phase 5 (edge) and 6 (swarm RL) still open. |
| `pyproject.toml` | config | Package metadata (`kaal`), Python ≥ 3.10, light core deps, extras `ml` (torch, ultralytics, rfdetr[train], pycocotools, supervision, huggingface_hub), `server` (fastapi, uvicorn, psutil, websockets), `dev` (pytest, ruff, mypy, httpx, pycocotools); ruff/mypy/pytest config; console scripts. |
| `requirements-dev.txt` | config | Dev tooling pins mirroring the `dev` extra. |
| `requirements-server.txt` | config | Backend deps mirroring the `server` extra. |
| `requirements-sim.txt` | config | AirSim + pynput for `SCRIPTS/` simulation demos. |
| `.gitignore` | config | Ignores weights caches, `runs/`, `MODEL/Datasets/`, node_modules, venvs; `/lib/` and `/lib64/` anchored to the root so `dashboard-app/src/lib` is tracked. |
| `.github/workflows/ci.yml` | config | On every push/PR: ruff check + format check, mypy, pytest (`-m "not gpu"`) on Python 3.11 and 3.12, dashboard `npm ci` + eslint + `vite build`. |
| `.github/workflows/integration.yml` | config | Manual (`workflow_dispatch`) GPU end-to-end test; needs a self-hosted CUDA runner. |

## `Docs/` — documentation (and the original project photos / screenshots)

| File | Status | Purpose |
|---|---|---|
| `Docs/WALKTHROUGH.md` | doc | Deep structural walkthrough: architecture, data flow, every subsystem, what was done and how it was verified. |
| `Docs/FILE_REFERENCE.md` | doc | This file. |
| `Docs/DATASETS.md` | doc | Dataset lineage, licences, which model used which data; human-readable view of `MODEL/datasets_registry.yaml`. |
| `Docs/DECISIONS.md` | doc | Architecture decision records (detector choice, scorer rewrite, temporal cue, data policy, security). |
| `Docs/MODEL_CARD.md` | doc | Model card for the production detector: intended use, data, metrics, limitations, ethics. |
| `Docs/FUTURE_SCALING.md` | doc | Scaling paths to new models, tasks and hardware tiers, with concrete tech stacks. |

## `MODEL/` — detection, defense, data, training, evaluation

### Production path

| File | Status | Purpose |
|---|---|---|
| `MODEL/__init__.py` | core | Package marker and module index. |
| `MODEL/config.py` | core | `Config` dataclass (paths, patch, FFT, training settings) with validation; enums `DefenseMode` (`none` / `adversarial_training` / `dual_stream`) and `PatchType` (`noise` / `pattern` / `loaded` / `gradient`). |
| `MODEL/detectors.py` | core | One `Detector` protocol over backends. `RFDETRDetector` (BGR→RGB, `predict(threshold=conf)`, class names from the checkpoint) and `UltralyticsDetector` (YOLO; class-agnostic NMS, IoU 0.45). `infer_backend()` picks by suffix (`.pth`/`.ckpt` → RF-DETR, else YOLO); `load_detector()` and `as_detector()` build or wrap one. Output for every backend: `{"bbox": [x1,y1,x2,y2], "confidence", "class_id", "class_name"}`. |
| `MODEL/fft_scorer.py` | core | Stream 2. `compute_fft_anomaly_score()` = share of AC spectral **power** above `0.5·(1−high_freq_ratio)` cycles/pixel on a mean-removed, Hann-windowed grayscale crop (rfft2, frequency-radius mask, cached windows). `compute_region_scores()` scores each box (None for boxes < `MIN_FFT_SIDE` = 8 px). `DEFAULT_FFT_THRESHOLD = 0.08` (≈ p93 of clean kaal_v3 boxes). |
| `MODEL/temporal_scorer.py` | core | Stream 3. `TemporalConsistencyDetector` (EMA background, block residual; research API) and `StreamingTemporalCue` (live): 32×32 cell grid, median-normalised signed residual, warm-up calibration (p99.9 × 1.5, floor 0.01), 5-frame persistence, masked background update, local absorb after `max_flagged_frames`, global re-baseline when > `global_frac` of cells change. |
| `MODEL/dual_stream_defense.py` | core | `DualStreamDefense`: runs the detector, whole-frame FFT, per-box FFT (overlap de-dup, tiny boxes unscored) and the optional temporal cue; returns detections + `anomaly_score`, `objects_flagged`, `temporal`, `is_adversarial`, `defense_action`. Calibrated threshold via a profile JSON. |
| `MODEL/calibration.py` | core | `FFTCalibrator`: collect clean scores (whole images, or per ground-truth box with `--labels`), derive a percentile/sigma threshold, save/load a `CalibrationProfile` JSON. `load_calibrated_threshold()` falls back to the default on missing/corrupt files. |
| `MODEL/adversarial_patch.py` | core | `AdversarialPatchInjector`: noise / pattern / gradient / loaded patches, scale range, perspective + lighting + alpha blending inside boxes; `parse_yolo_labels()`. Used to create attacks for every evaluation. |
| `MODEL/augmentations.py` | core | Attack-style augmentations (`add_patch`, `add_noise`, `blur`, `occlude`, `add_low_freq_noise`, `apply_random_attack`, `apply_all_attacks`). |

### Data tools

| File | Status | Purpose |
|---|---|---|
| `MODEL/fetch_datasets.py` | tool | Downloads the Simuletic UAV-tank set (HF snapshot) and a rarity-weighted subset of the military-aircraft set (per-file with 429 back-off). VisDrone is fetched with `gh release download` (documented in the docstring). |
| `MODEL/build_dataset.py` | tool | Builds `kaal_v3`: 9-class taxonomy (`aircraft, helicopter, tank, armored_vehicle, artillery, engineering_vehicle, soldier, vehicle, person`), source adapters (legacy, VisDrone, aircraft, UAV tank), legacy class-0 image drop, person≠soldier rule, source-photo grouping (`legacy_source`), flip-aware dHash near-duplicate grouping, leakage resolution, rarity-weighted capping, resize to 1024 px, `data.yaml` + `manifest.json`. `--legacy-split resplit` is an opt-in content-grouped re-split. |
| `MODEL/build_ground_test.py` | tool | Builds `kaal_ground_test`, an independent test-only set (Orion web images, Russian-military-annotated, TNO Search_2) mapped to the KAAL taxonomy with ignore regions for classes KAAL has no label for, excluding near-duplicates of training images. |
| `MODEL/analyze_dataset.py` | legacy | Flags tiny/fragmented boxes in a YOLO dataset. |
| `MODEL/validate_dataset.py` | legacy | Sanity checks a YOLO `data.yaml` (missing labels, bad coordinates). |
| `MODEL/merge_datasets.py` | legacy | Merges two YOLO datasets with class remapping (superseded by `build_dataset.py`). |
| `MODEL/merge_all.py` | legacy | Built the original merged set (`Datasets/master_data`): aircraft images (`air_*`, Roboflow `aircraft.v1i.yolov8` export) split across train/val/test, aerial cars (`car_*`), and the `military-artillery-and-tanks` CVAT export (`mil_*`, licence unknown) split 90/10 into train/val by file order — the val slice is almost entirely unlabeled (181 of 198 images) and nothing went to test: the root cause of the missing ground classes in the legacy val/test folders. Superseded by `build_dataset.py`. |
| `MODEL/dataset.py` | research | `DatasetPreparer`: produces the adversarially augmented YOLO dataset under `Config.output_dir` (`runs/adversarial/augmented_dataset` — the "legacy" source of `kaal_v3`), adding `_noisy` / `_occlude` / `_patch` variants. |

### Training

| File | Status | Purpose |
|---|---|---|
| `MODEL/train_detector.py` | tool | Fine-tunes RF-DETR (nano/small/medium) or a YOLO baseline (yolo26n/s/m, yolov8n/m) on a YOLO-format dataset; writes `runs/train/<name>/kaal_train_summary.json`. Optional `--resume-from` warm start. |
| `MODEL/trainer.py` | research | `AdversarialTrainer`: Ultralytics training with the three defense modes (none / augment / full). |
| `MODEL/pipeline.py` | research | CLI tying `Config` → `AdversarialTrainer` → `RobustnessEvaluator` (the Phase 0 pipeline). |
| `MODEL/training/adv_train.py` | legacy | `RobustDetectionTrainer`: PGD adversarial training hook for Ultralytics (experimental). |
| `MODEL/attacks/fgsm.py`, `MODEL/attacks/pgd.py` | legacy | Gradient attacks against YOLOv8 with a proxy loss (suppress / hallucinate). |
| `MODEL/defenses/preprocess.py` | legacy | JPEG / bit-depth / blur input-transform defenses on tensors (re-implemented in NumPy inside `benchmark.py`). |
| `MODEL/detection/anomaly.py` | legacy | Confidence-entropy anomaly detector (early alternative to the FFT cue). |

### Evaluation & research

| File | Status | Purpose |
|---|---|---|
| `MODEL/compare_detectors.py` | tool | Head-to-head evaluation: COCO mAP50 / mAP50:95 (pycocotools, maxDets 300) clean and with a seeded patch in every GT box, per class, per source, optional class filter (`--eval-classes`), ignore regions, class-agnostic mAP (`--agnostic`), median latency, model-free FFT flag rates + AUROC. Writes `report.json` / `report.md`. |
| `MODEL/evaluator.py` | research | `RobustnessEvaluator`: clean / attacked / defended passes with seeded patches; mAP, confusion, flag rate. |
| `MODEL/benchmark.py` | research | Phase 3 attack × defense matrix (detectability and mAP) with VOC mAP, universal/low-frequency perturbations, latency. |
| `MODEL/failure_analysis.py` | research | Synthetic textures (sky/foliage/urban/rain) and low-frequency attacks to find where the FFT fails; temporal-cue comparison. |
| `MODEL/ablation.py` | research | Phase 1 sweeps over `high_freq_ratio`, threshold, patch scale, region mode. |
| `MODEL/visualizer.py` | research | Side-by-side clean/attacked panels, prediction drawing, confusion matrices. |
| `MODEL/make_demo_video.py` | tool | Renders a clean → patched → clean clip from one labelled frame for the live demo (`KAAL_CAMERA_SOURCE`). |
| `MODEL/evaluation/evaluate.py`, `MODEL/evaluation/metrics.py` | legacy | Early PGD/defense evaluation loop and metric container. |
| `MODEL/visualization/compare.py` | legacy | Ultralytics predictions on clean/adversarial/defended sets for figures. |
| `MODEL/evaluate_custom_report.py` | legacy | Custom COCO-json evaluation that produced `custom_evaluation_report.txt`. |
| `MODEL/test_model.py` | legacy | Original YOLO test script that produced `test_report_*.txt`. |
| `MODEL/inference_quality_fix.py` | legacy | YOLO inference with stricter NMS to reduce fragmented boxes. |
| `MODEL/use_model.py`, `MODEL/user_testing/user_test.py` | legacy | One-off inference scripts with hard-coded `D:\` paths. |
| `MODEL/test/clean`, `MODEL/test/moisy`, `MODEL/test/patched` | legacy | Python attack-demo snippets saved without a `.py` extension (clean / noisy / patched image generation). |

### Results, reports, weights

| File | Status | Purpose |
|---|---|---|
| `MODEL/RESULTS.md` | doc | Measured results: §0 FFT scorer fix, §1–4 Phase 3 benchmark/failure analysis, §6 detector comparison and ground-vehicle test. |
| `MODEL/README.md` | doc | MODEL package usage. |
| `MODEL/requirements.txt` | config | Core ML dependencies for `MODEL/`. |
| `MODEL/datasets_registry.yaml` | doc/data | Machine-readable dataset registry (status, licence, use per model). Validated by `tests/test_datasets_registry.py`. |
| `MODEL/custom_evaluation_report.txt`, `MODEL/test_report_YOLOv8m_base.txt`, `MODEL/test_report_YOLOv8n_base.txt` | legacy | Reports from the original YOLO experiments (kept for history). |
| `MODEL/yolov8n.pt`, `MODEL/yolov8m.pt`, `MODEL/yolo26n.pt` | legacy | Stock Ultralytics COCO weights committed by the original project (backend default `KAAL_MODEL_PATH` is still `MODEL/yolov8m.pt`). |
| `MODEL/kaal_eye_v2.pt` | legacy | YOLO weights from the original KAAL training: 59 classes (aircraft types `A10`…, ground vehicles such as `Light Armoured Vehicle (LAV)`, `Military Engineering Vehicle (MEV)`, plus `Civilian Car`, `Civilian Aircraft`, `Gun`, `Human`). |

Trained KAAL weights are **not** committed (size); they live under
`MODEL/runs/train/` and are described in `Docs/MODEL_CARD.md`.

## `Frontend_Backend/`

| File | Status | Purpose |
|---|---|---|
| `backend/server.py` | core | FastAPI app. `Settings` (+ Pydantic `SettingsUpdate`, finite-only), `DetectorAdapter` (any `MODEL.detectors` backend; demo mode without weights), `CameraWorker` thread (webcam / video file / stream, aspect-preserving resize to `KAAL_PROCESS_SIDE`, live confidence, per-box FFT, temporal cue) → `loop.call_soon_threadsafe(enqueue_latest, ...)` drop-oldest queue → `_broadcast_loop` → WebSocket `/ws`; `MetricsSampler` (1 Hz net / CPU / GPU via nvidia-smi); REST `/api/health`, `/api/python-data`, `/api/settings` (PUT token-checked with `secrets.compare_digest`), `/api/video_feed` (MJPEG); CORS allow-list + WebSocket Origin check; NaN-safe 422 handler. |
| `.gitignore` | config | Empty placeholder. |
| `dashboard-app/package.json`, `package-lock.json` | config | React 19, `motion` (animations), recharts (graphs), leaflet + react-leaflet (map); dev: Vite 8, Tailwind 3, ESLint 9. Scripts `dev`, `build`, `lint`. |
| `dashboard-app/vite.config.js` | config | Dev server on 5173, proxies `/api` and `/ws` to `:3000`. |
| `dashboard-app/eslint.config.js` | config | Flat ESLint config (`varsIgnorePattern` for constants and `motion`). |
| `dashboard-app/tailwind.config.js`, `postcss.config.js`, `index.html`, `README.md`, `.gitignore` | config/doc | Build scaffolding. |
| `src/main.jsx` | core | React entry point. |
| `src/App.jsx` | core | Layout and state derived from the live snapshot (alerts, swarm telemetry, system metrics); alert-onset detection; `PUT /api/settings` with optional `VITE_KAAL_API_TOKEN`. |
| `src/lib/useDefenseSocket.js` | core | WebSocket hook with exponential-backoff reconnect and REST polling fallback while disconnected. |
| `src/lib/SwarmEngine.js` | core | Boids simulation (separation / alignment / cohesion) for the swarm canvas. |
| `src/components/KaalSwarmCanvas.jsx` | core | Canvas rendering of the swarm and threat markers. |
| `src/components/MainPanel.jsx` | core | Video feed, detections and alert banner. |
| `src/components/MultiGraph.jsx`, `GraphPanel.jsx` | core | Time series: FFT score, % boxes flagged, temporal residual ×100. |
| `src/components/SettingsPanel.jsx` | core | Threshold (0–0.5, step 0.005, default 0.08) and confidence controls. |
| `src/components/StatusBar.jsx` | core | Connection, FPS, latency, TEMPORAL and DETECTOR status. |
| `src/components/Controls.jsx`, `SystemStatus.jsx`, `Navbar.jsx`, `Sidebar.jsx` | core | CPU/GPU/network panels (real metrics) and navigation. |
| `src/components/MapPanel.jsx` | core | Leaflet map (OpenStreetMap tiles) with a fixed marker — no live positions yet. |
| `src/App.css`, `src/index.css` | core | Styles. |

## `SCRIPTS/` — simulation demos (AirSim)

| File | Status | Purpose |
|---|---|---|
| `kaal_swarm.py` | research | Pygame tactical swarm simulation (boids + walls + line-of-sight + KAAL mode). |
| `virtual_drone_controller.py` | research | AirSim multirotor wrapper (takeoff, hover, camera, velocity moves). |
| `simulated_eye.py` | research | AirSim camera feed + keyboard control + FFT patch check. |
| `kaal_fft_radar.py`, `kaal_integrated_radar.py`, `kaal_integrated_radar_0.py`, `a.py` | legacy | Early webcam "radar" demos combining FFT and YOLO; `a.py` is a near-copy of `kaal_fft_radar.py`. |

## `tests/`

| File | Covers |
|---|---|
| `conftest.py` | Ultralytics stub when the real package is absent; shared image fixtures. |
| `test_anomaly_score.py` | FFT score: clean vs patched, edge cases, region scoring, shape invariance. |
| `test_temporal_scorer.py` | Temporal detector + `StreamingTemporalCue` (warm-up, persistence, localisation, re-baseline). |
| `test_calibration.py` | Threshold derivation, per-box collection, profile round-trip, loader fallbacks. |
| `test_detectors.py` | Detector adapters (fake RF-DETR / Ultralytics models, no torch). |
| `test_adversarial_patch.py`, `test_augmentations.py`, `test_parse_yolo_labels.py` | Patch placement/blending, attack augmentations, label parsing. |
| `test_build_dataset.py` | Label remaps, person≠soldier, leakage, rarity sampling, writer, source grouping, near-duplicate grouping, stratified re-split. |
| `test_build_ground_test.py` | Orion / Russian / Search_2 mapping, Roboflow-copy collapse, crop geometry. |
| `test_compare_detectors.py` | COCO mAP maths, ignore regions, class filter, agnostic + per-source summaries. |
| `test_benchmark.py`, `test_failure_analysis.py`, `test_config.py`, `test_make_demo_video.py` | Research harnesses, config validation, demo clip. |
| `test_datasets_registry.py` | Registry schema and consistency with the dataset builders. |
| `integration/test_server.py` | Backend: REST schema, settings validation + token, CORS + WS origin, worker → broadcast path, drop-oldest queue, live confidence. |
| `integration/test_dual_stream.py` | Full `DualStreamDefense.predict()` with a fake detector (frame-level and per-region detection ≥ 80%). |
| `integration/test_end_to_end.py` | GPU-only (`-m gpu`): 1-epoch train + evaluation. |
