# KAAL — Structural Walkthrough

A complete tour of the system: what each part does, how data moves through
it, how it was built and verified, and how to run it. File-by-file details
are in [`FILE_REFERENCE.md`](FILE_REFERENCE.md); the reasons behind each
choice are in [`DECISIONS.md`](DECISIONS.md); measured numbers are in
[`MODEL/RESULTS.md`](../MODEL/RESULTS.md).

---

## 1. What KAAL is

KAAL is a drone-perception and swarm framework built around two attacks that
are cheap for an adversary and expensive for the operator:

1. **Visual deception** — a printed adversarial patch that makes a detector
   miss or mislabel an object.
2. **Signal denial** — jamming GPS and the command link so a swarm cannot
   coordinate.

KAAL answers the first with a detector plus **independent, model-free tamper
checks** that flag a frame or an object as untrustworthy, and the second with
**local, leaderless flocking**. This walkthrough covers the perception stack in
depth (it is what was rebuilt in Phase 4.5) and the swarm layer as it exists
today (hand-coded boids; learned policy is roadmap Phase 6).

---

## 2. Architecture at a glance

```
            ┌──────────────────────────── MODEL/ (Python package) ───────────────────────────┐
            │                                                                                 │
 camera ─┐  │  Stream 1  detectors.py      RF-DETR (production) │ YOLO (optional)            │
 video  ─┼─▶│  Stream 2  fft_scorer.py     spectral power ratio, whole frame + per box       │
 file   ─┘  │  Stream 3  temporal_scorer.py per-cell persistent residual (StreamingTemporalCue)│
            │  fusion    dual_stream_defense.py / backend process_frame()                     │
            └───────────────────────────────┬─────────────────────────────────────────────────┘
                                            │ snapshot dict (detections, scores, flags, metrics)
            ┌───────────────────────────────▼─────────────────────────────────────────────────┐
            │ Frontend_Backend/backend/server.py (FastAPI)                                     │
            │  CameraWorker thread ─call_soon_threadsafe─▶ asyncio.Queue(4, drop-oldest)       │
            │  ─▶ broadcast loop ─▶ WebSocket /ws      REST /api/*      MJPEG /api/video_feed  │
            └───────────────────────────────┬─────────────────────────────────────────────────┘
                                            │ JSON over WebSocket (REST polling fallback)
            ┌───────────────────────────────▼─────────────────────────────────────────────────┐
            │ Frontend_Backend/dashboard-app (React 19 + Vite)                                 │
            │  useDefenseSocket ─▶ App state ─▶ video, alerts, graphs, settings, swarm canvas  │
            └──────────────────────────────────────────────────────────────────────────────────┘

 Offline tooling (MODEL/):  fetch_datasets → build_dataset / build_ground_test → train_detector
                            → compare_detectors → RESULTS.md; registry: datasets_registry.yaml
```

---

## 3. The live pipeline, frame by frame

`CameraWorker` (a background thread in `server.py`) runs this loop:

1. **Capture.** Webcam `KAAL_CAMERA_INDEX`, or `KAAL_CAMERA_SOURCE` (a video
   file — looped, with a rewind guard — or a stream URL).
2. **Resize.** Longest side → `KAAL_PROCESS_SIDE` (default 960), aspect ratio
   preserved, `INTER_AREA`. (A fixed 640×480 resize used to squash non-4:3
   sources and blur small patches away.)
3. **Stream 1 — detect.** `DetectorAdapter.predict(frame, conf)` with the
   live confidence from `Settings` (default `KAAL_CONF` 0.4). Output per
   object: `{"bbox": [x1,y1,x2,y2], "confidence", "class_id", "class_name"}`.
   A rough distance is estimated from box width and a per-class known width.
4. **Stream 2 — per-box FFT.** `compute_region_scores(frame, boxes, 0.25,
   KAAL_REGION_MIN_SIZE=16)`; a box is flagged when its score exceeds the
   live FFT threshold (default 0.08). Boxes smaller than 16 px are unscored.
5. **Stream 2 — whole-frame FFT.** `compute_fft_anomaly_score(frame, 0.25)`.
6. **Stream 3 — temporal cue.** `StreamingTemporalCue.observe(frame)` →
   `{score, threshold, flagged, ready, hot_cells}` (§6.3).
7. **Fusion.** `is_attack = frame_fft > threshold OR any box flagged OR
   temporal flagged`. The frame is annotated (boxes, scores, banner).
8. **Snapshot.** Status, scores, threshold, flags, temporal result, detector
   info, detections, system metrics (`MetricsSampler`, sampled at most once
   per second: CPU, RAM, network rate, GPU via `nvidia-smi`, real pipeline
   latency) and a 60-point history (`KAAL_HISTORY`) for the graphs.
9. **Hand-off to asyncio.** The thread calls
   `loop.call_soon_threadsafe(enqueue_latest, queue, snapshot)`. The queue
   holds 4 snapshots; when full the **oldest** is dropped, so a slow client
   never grows memory and the dashboard always shows the newest frame.
10. **Broadcast.** `_broadcast_loop` sends each snapshot to every connected
    WebSocket client; the latest snapshot and JPEG are also kept for
    `GET /api/python-data` and the MJPEG stream.

`MODEL/dual_stream_defense.py` implements the same fusion as a library call
(`DualStreamDefense.predict(image)`), used by offline evaluation, tests and
the end-to-end demo.

---

## 4. Stream 1 — detectors (`MODEL/detectors.py`)

* **One protocol.** `Detector.predict(image_bgr, conf) -> list[dict]`. The
  defense, backend and evaluation never touch framework objects.
* **`RFDETRDetector`** — converts BGR → RGB, calls
  `model.predict(rgb, threshold=conf)` (RF-DETR returns `supervision`
  detections), maps class ids to the names stored in the checkpoint.
  `from_weights()` uses `rfdetr.RFDETR.from_checkpoint`, or a size class for
  COCO weights. Optional `optimize_for_inference`.
* **`ONNXDetector`** — the RF-DETR exported by `python -m MODEL.export_model`,
  run with ONNX Runtime (no torch): bilinear resize to the graph input,
  ImageNet normalisation, sigmoid over the class logits, top-300 over
  (query, class), boxes scaled to the frame — RF-DETR's own post-processing,
  parity-tested (ΔmAP50:95 +0.0004; 3.6× faster than PyTorch on CPU).
* **`UltralyticsDetector`** — YOLO with class-agnostic NMS at IoU 0.45.
* **Selection.** `infer_backend()` maps `.pth`/`.ckpt` → RF-DETR, anything
  `.onnx` → ONNX Runtime, anything else → YOLO; `KAAL_DETECTOR` overrides. `production_weights()` returns the
  weights of the registry model marked `status: production`; the backend
  uses it when `KAAL_MODEL_PATH` is unset, then falls back to the stock
  `MODEL/yolov8m.pt`, then to demo mode (no detector; FFT, temporal, metrics
  and streams still run).
* **Why RF-DETR** — ADR-001: +3.5 mAP50:95 on kaal_v3, smaller drop under
  attack, and 0.72 vs 0.02 any-class mAP50 on unseen ground-vehicle photos.

---

## 5. Stream 2 — FFT anomaly score (`MODEL/fft_scorer.py`)

Printed patches inject high-frequency energy that natural, optically blurred
scenes lack (natural images fall off roughly as 1/f²).

```
gray  = float32 grayscale crop, mean removed
win   = Hann window (cached per shape)
P     = |rfft2(gray · win)|²
r     = sqrt(fy² + fx²)      # cycles/pixel (fftfreq / rfftfreq)
score = Σ P[r > 0.5·(1 − high_freq_ratio)] / Σ P[AC]   (rfft columns weighted ×2
                                                        except DC/Nyquist)
```

* The band is defined in cycles/pixel, so a tall, wide or square crop of the
  same content scores the same (old scorer: 0.50 vs 0.91; new: 0.068–0.079).
* Smooth 0.009, natural photo ≈ 0.07, white noise 0.555.
* Crops smaller than 8 px (`MIN_FFT_SIDE`) score 0 / are unscored.
* `DEFAULT_FFT_THRESHOLD = 0.08` ≈ p93 of clean kaal_v3 ground-truth boxes
  (5.4% clean false flags, 28.8% of patched boxes flagged). Per camera:
  `python -m MODEL.calibration --labels ...` writes a profile JSON
  (`KAAL_CALIBRATION`).
* Limits: blind to low-frequency perturbations; weak on tiny boxes
  (per-box AUROC 0.74 on real data). Hence Stream 3.

---

## 6. Stream 3 — temporal consistency (`MODEL/temporal_scorer.py`)

### 6.1 Research form — `TemporalConsistencyDetector`
EMA background (α) over a block grid; score = mean absolute residual.
Used by `failure_analysis.py`: low-frequency attacks 0% (FFT) → 100%
(temporal) at 0% clean FPR.

### 6.2 Why a streaming form was needed
A frame-mean residual misses a patch that covers 0.5% of the frame, and a
fixed threshold breaks under lighting changes. `StreamingTemporalCue` fixes
both.

### 6.3 `StreamingTemporalCue(threshold=None, warmup_frames=30, percentile=0.999, margin=1.5, min_threshold=0.01, persist_frames=5, max_flagged_frames=150, global_frac=0.5, grid=32, alpha=0.05)`

1. Downsample the frame to a 32×32 cell grid.
2. Residual per cell = (cell − background) − median(cell − background),
   absolute, /255 — the median removes global brightness shifts.
3. **Warm-up:** the first 30 frames calibrate the threshold =
   max(p99.9 of residuals × 1.5, 0.01) unless one is given.
4. A cell is **hot** when its residual exceeds the threshold; the frame is
   **flagged** when any cell has been hot for ≥ 5 consecutive frames (a
   patch fixed on a target persists; a passing vehicle does not).
5. **Masked update:** hot cells are not blended into the background, so a
   patch is not "learned away".
6. **Absorb / re-baseline:** a cell hot for 150 frames is absorbed (a parked
   vehicle); if ≥ 50% of cells are stale the whole view changed (camera
   moved) and the background is relearned.

Live result with the trained RF-DETR (220-frame clip, patch on one of seven
tanks): no false alarms on 110 clean frames, flagged 4 frames (0.2 s at
20 fps) after the patch appeared, on 76/80 patched frames; cleared on the
first clean frame. Assumption: static or slowly moving camera.

---

## 7. Fusion and what "defense" means

* `DualStreamDefense.predict()` returns detections with per-box
  `anomaly_score`/`is_adversarial`, the frame score, `objects_flagged`, the
  temporal result and a human-readable `defense_action`.
* Overlapping boxes (IoU ≥ 0.5) reuse the score of the highest-confidence
  box; boxes under `region_min_size` are unscored.
* **The defense flags, it does not repair** (ADR-004). The dashboard shows
  "predictions unreliable"; the metric is flag rate vs clean false-flag rate.

---

## 8. Backend (`Frontend_Backend/backend/server.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness, mode (live / demo), camera, detector (`backend`, `weights`, `fallback`), temporal status |
| `GET /api/python-data` | latest snapshot (polling fallback) |
| `GET /api/settings` / `PUT /api/settings` | live FFT threshold and detector confidence; PUT requires `X-KAAL-Token` when `KAAL_API_TOKEN` is set |
| `GET /api/video_feed` | MJPEG of the annotated frame |
| `WS /ws` | snapshot broadcast |

Security defaults (ADR-006): bind `127.0.0.1`; CORS allow-list
(`KAAL_CORS_ORIGINS`) also enforced on the WebSocket `Origin` header; token
compared with `secrets.compare_digest`; Pydantic rejects NaN/∞ and a custom
422 handler keeps NaN input from crashing the error path. Full variable list:
module docstring and README.

---

## 9. Dashboard (`Frontend_Backend/dashboard-app`)

* **Transport:** `useDefenseSocket(1000)` opens `ws(s)://<host>/ws` (Vite
  proxies `/ws` and `/api` to `:3000`). If the socket drops it polls
  `/api/python-data` every second and retries the socket with exponential
  backoff (1 s → 10 s max); polling stops when the socket is back.
* **State:** `App.jsx` derives everything from the latest snapshot (no
  duplicated state); alert onset is detected during render.
* **Views:** `MainPanel` shows the annotated MJPEG feed with detection tags
  and a pulsing border on alert, or — via its toggle — the boids swarm
  simulation (`KaalSwarmCanvas` + `SwarmEngine.js`); FFT score / % boxes
  flagged / temporal ×100 graphs (`MultiGraph`, `GraphPanel`); live threshold
  and confidence controls (`SettingsPanel`, threshold 0–0.5 in 0.005 steps,
  default 0.08); status bar with FLAGS, NOISE, THRESH, TEMPORAL and DETECTOR;
  real CPU/GPU/memory/latency/network panels.
* **Not wired yet:** `MapPanel` (Leaflet) is not mounted, and the sidebar's
  model list (QUANTUM-V1 / HELIX-V2 / ATLAS-V3) is a cosmetic label — the
  detector is selected on the backend.

---

## 10. Data pipeline

```
fetch_datasets.py ──▶ MODEL/Datasets/<raw sources>      (registry: sources)
                         │
build_dataset.py ────────┼──▶ kaal_v3 (11.6k train)  ──▶ trained RF-DETR-S v3, YOLO26s v3
                         └──▶ kaal_v4 (+DVIDS, +Open Images tanks, all VisDrone,
                                       cross-source de-dup)  ──▶ RF-DETR-S v4 (production)
build_ground_test.py ───────▶ kaal_ground_test (798 unseen ground-vehicle photos, test only)
```

`build_dataset.py`, in order:

1. **Load** each source with its adapter → `Sample(source, split, image,
   group, boxes(h, w), classes, augmented)`; every adapter maps to the
   9-class taxonomy and documents what it drops.
2. **Group** copies of one photo: Roboflow `.rf.<hash>` and
   `_noisy/_occlude/_patch` names (`legacy_source`), DVIDS upload blocks, and
   — across all sources — near-identical images by a flip-aware 256-bit
   difference hash (`merge_near_duplicate_groups`, ≤ 20 bits).
3. **Resolve leakage:** each group lives in one split (test > val > train).
4. **Cap** per source and split with rarity-weighted sampling (images with
   rare classes are kept preferentially).
5. **Write** images (longest side ≤ 1024), YOLO labels, `data.yaml`,
   `manifest.json` (arguments, counts, licences, duplicates removed).

Why each rule exists — the data problems found in Phase 4.5 — is in
[`DATASETS.md`](DATASETS.md) §3.1.

---

## 11. Training (`MODEL/train_detector.py`)

* `--arch rfdetr-{nano,small,medium}` or `yolo26{n,s,m}` / `yolov8{n,m}`;
  same YOLO-format dataset for both.
* RF-DETR: `model.train(dataset_dir, dataset_file="yolo", epochs,
  batch_size=4, grad_accum_steps=4, lr, early_stopping, checkpoint_interval,
  tensorboard)`; output `MODEL/runs/train/<name>/checkpoint_best_total.pth`.
  `--init-weights` warm-starts from a previous KAAL checkpoint.
* YOLO: Ultralytics `train` with batch = batch × grad_accum.
* The process `chdir`s into `MODEL/runs/weights/` so framework downloads land
  in one ignored cache; `kaal_train_summary.json` records arch, data, epochs,
  init weights and wall time.
* Hardware used: RTX 4060 Laptop (8 GB). RF-DETR-S ≈ 10 img/s at batch 4.

---

## 12. Evaluation (`MODEL/compare_detectors.py`)

For each detector on one split:

* **Clean and attacked passes.** Attacked = a seeded noise or pattern patch
  (scale 0.15–0.4) inside every ground-truth box ≥ 8 px.
* **COCO mAP** via pycocotools with maxDets 300 (dense drone scenes), read
  directly from the precision array; per class and per source (file-name
  prefix).
* **Options:** `--eval-classes` (score a subset; other-class detections are
  dropped), ignore regions (`<split>/ignore/*.txt` → COCO `iscrowd`),
  `--agnostic` (merge scored classes: did it find the object at all?),
  `--dedup-against` (exclude test images that near-duplicate a reference set).
* **Latency:** median single-image `predict()` after 5 warm-up images.
* **FFT defense (model-free):** share of patched GT boxes vs clean GT boxes
  above the threshold, plus threshold-free AUROC.

Research harnesses: `benchmark.py` (attack × defense matrix),
`failure_analysis.py` (where FFT fails), `ablation.py` (hyper-parameter
sweeps), `evaluator.py` (clean/attacked/defended passes).

---

## 13. Quality gates

* **CI** (`.github/workflows/ci.yml`): ruff + format check and mypy on
  `MODEL`, `tests`, backend; pytest (`-m "not gpu"`) on Python 3.11 and 3.12
  including server integration tests; dashboard `npm ci`, ESLint, `vite build`.
* **Tests** cover the math (FFT shape invariance, COCO maths with ignore
  regions, AUROC), the data rules (person ≠ soldier, leakage, near-duplicate
  grouping, adapters), the registry, the backend contract (CORS, token, NaN,
  drop-oldest queue, live confidence) and the dual-stream integration.
* **GPU end-to-end** (`integration.yml`) is manual on a self-hosted runner.

---

## 14. What was done in Phase 4.5 (2026-10-11), in order

| Commit | What and why |
|---|---|
| `d70ff82` | Review fixes in the defense: the FFT score was shape-biased → rewritten as a cycles/pixel power ratio; one shared threshold; per-box calibration. |
| `27885b6` | RF-DETR and YOLO behind one detector interface; temporal cue fused into the defense. |
| `f415bca` | Backend: real WebSocket broadcast from the worker thread, drop-oldest queue, locked-down settings (CORS, Origin, token, NaN), real metrics, video-file source. |
| `9f2473a` | Dashboard: `src/lib` was untracked (gitignore), live data instead of mocks, reconnect with backoff, lint clean. |
| `6581dc5` | `kaal_v3` dataset builder (4 sources, 9 classes), training and comparison tooling, demo clip. |
| `4f37ef7` | CI matrix + dashboard job, `ml` extra, merged README. |
| `bb31d83` | Independent ground-vehicle test set, ignore regions, duplicate-aware splits. |
| `82e2c11` | RF-DETR chosen as production (RESULTS §6); `kaal_v4` pipeline (DVIDS, Open Images tanks), dataset registry. |
| next | Production RF-DETR fine-tuned on `kaal_v4`; documentation set (`Docs/`). |

Measured outcomes are in `MODEL/RESULTS.md` (§0 scorer, §6 comparison, §7
kaal_v4 model).

---

## 15. Runbook

```bash
# install
pip install -e ".[ml,server,dev]"

# data
python -m MODEL.fetch_datasets uav_tank
python -m MODEL.fetch_datasets aircraft --train 3500 --val 400 --test 400
python -m MODEL.fetch_datasets dvids
python -m MODEL.fetch_datasets openimages_tank
gh release download v0.0.0 --repo ultralytics/assets --pattern 'VisDrone2019-DET-*.zip' --dir MODEL/Datasets/visdrone
python -m MODEL.build_dataset                       # -> MODEL/Datasets/kaal_v4
python -m MODEL.build_ground_test --train-ref MODEL/Datasets/kaal_v4/train/images

# train (production recipe)
python -m MODEL.train_detector --arch rfdetr-small --data MODEL/Datasets/kaal_v4 \
    --init-weights MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth \
    --epochs 10 --lr 5e-5 --name rfdetr-small-v4

# evaluate
python -m MODEL.compare_detectors --data MODEL/Datasets/kaal_v4 \
    --detector v4=MODEL/runs/train/rfdetr-small-v4/checkpoint_best_total.pth
python -m MODEL.compare_detectors --data MODEL/Datasets/kaal_ground_test \
    --eval-classes tank,armored_vehicle,artillery,engineering_vehicle --agnostic \
    --detector v4=MODEL/runs/train/rfdetr-small-v4/checkpoint_best_total.pth

# export for deployment (ONNX + parity check; also --format tensorrt/openvino/...)
python -m MODEL.export_model --parity 200
KAAL_MODEL_PATH=MODEL/runs/export/rfdetr-small-v4/rfdetr-small-v4.onnx python Frontend_Backend/backend/server.py

# run (uses the production weights from the registry)
python Frontend_Backend/backend/server.py
cd Frontend_Backend/dashboard-app && npm ci && npm run dev

# attack demo without a webcam
python -m MODEL.make_demo_video --image MODEL/Datasets/kaal_v4/test/images/<frame>.jpg --out demo.mp4
KAAL_CAMERA_SOURCE=demo.mp4 python Frontend_Backend/backend/server.py
```

---

## 16. Known limitations (honest list)

* Ground-vehicle **sub-classes** (tank vs APC vs artillery vs engineering)
  are the weakest part of the model; finding the vehicle is much better than
  naming it (RESULTS §6.2, §7).
* Tiny objects from drone altitude (VisDrone people) remain hard at 512 px.
* The temporal cue assumes a static camera; per-box FFT is weak on tiny boxes.
* Several sources are non-commercial or of unknown licence (DATASETS §6).
* The swarm layer is hand-coded boids; MAPPO, simulation and thermal are
  roadmap items (`roadmap.md`, `FUTURE_SCALING.md`).
* The snapshot `status` field still uses the legacy strings `ENGAGING` /
  `NOMINAL` (detections or alert present / absent); it carries no action.

---

## 17. Glossary

| Term | Meaning in KAAL |
|---|---|
| **Adversarial patch** | A printed/stuck-on pattern that makes a detector miss or mislabel an object. Simulated by `adversarial_patch.py` (noise, pattern, gradient, loaded image; perspective + lighting + alpha). |
| **Vanishing attack** | A patch that makes the object disappear from the detections — why the defense must run even on frames with no detections. |
| **Stream 1 / 2 / 3** | Detector / FFT spectral check / temporal-consistency check. Streams 2 and 3 are model-free: they do not depend on the network being attacked. |
| **FFT score** | Share of a crop's spectral power above a cut-off frequency (cycles/pixel). Printed noise/patterns raise it; natural optics lower it. |
| **Temporal residual** | How far a 32×32 grid cell differs from its learned background (after removing global brightness change). |
| **Flag rate / false-flag rate** | Share of attacked / clean inputs the defense marks as untrustworthy. The defense metric (ADR-004). |
| **AUROC** | Threshold-free separability of clean vs attacked scores: 0.5 = chance, 1.0 = perfect. |
| **mAP50 / mAP50:95** | COCO mean average precision at IoU 0.5 / averaged over IoU 0.5–0.95 (stricter box quality). |
| **Any-class (agnostic) mAP** | mAP after merging the scored classes into one: did the detector find the object at all, regardless of its label? |
| **IoU** | Intersection-over-union of two boxes. |
| **Ignore region** | A ground-truth area (e.g. a light armoured car KAAL has no class for) where detections are neither rewarded nor penalised (COCO `iscrowd`). |
| **DETR / RF-DETR** | Detection Transformer: predicts a fixed set of boxes directly, so no NMS post-processing. RF-DETR uses a DINOv2 (self-supervised ViT) backbone. |
| **NMS** | Non-maximum suppression — removes overlapping duplicate boxes (YOLO needs it; RF-DETR does not). |
| **EMA (weights)** | Exponential moving average of model weights during training; RF-DETR selects its best checkpoint from the EMA copy. |
| **Warm start** | Initialising training from an earlier KAAL checkpoint (`--init-weights`) instead of COCO weights. |
| **dHash** | Difference hash: a 256-bit fingerprint of an image's brightness gradients; near-identical images differ in few bits. KAAL also hashes the mirror image so flipped copies match. |
| **Leakage** | The same (or near-identical) photo in both train and test, which inflates test scores. |
| **Rarity sampling** | Keeping images with rare classes preferentially when capping a source (weighted sampling without replacement). |
| **Registry** | `MODEL/datasets_registry.yaml` — which data exists, its licence, and which model used it; its `status: production` model is what the backend loads. |
| **ONNX / ONNX Runtime** | Framework-neutral model file and the runtime that executes it without PyTorch (`.onnx` backend). |
| **Parity test** | Scoring the exported model and the PyTorch original on the same images; export fails if mAP moves by > 0.01. |
| **Demo mode** | Backend running without a detector (missing weights/stack): FFT, temporal cue, metrics and streams still work. |
