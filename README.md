# KAAL — Perception Robustness & Swarm Research Framework
**KAAL is a computer-vision robustness and swarm-simulation research project. It combines object detection, image-anomaly cues, evaluation tooling, and a real-time monitoring dashboard.**

> **Scope and limitations:** KAAL's anomaly streams flag potentially unreliable inputs; they do not prove that an image is adversarial, repair a model's predictions, or guarantee robustness. Swarm coordination is currently a hand-coded simulation, not a validated autonomous flight system. Keep consequential actions under human control.

[![CI — lint, typecheck, tests, dashboard](https://github.com/aryank2212/KAAL_v2/actions/workflows/ci.yml/badge.svg)](https://github.com/aryank2212/KAAL_v2/actions/workflows/ci.yml)

---

## 🛑 The Problem

Unmanned aerial systems face two cheap, mission-ending attacks:

1. **Visual deception (adversarial patches).** A printed sticker on a target can make a state-of-the-art detector misclassify or miss it entirely.
2. **Signal denial (GPS / link jamming).** A low-cost jammer cuts a swarm off from GPS and its command hub, and the formation falls apart.

*Both attacks cost the enemy pennies. Both wipe out multimillion-dollar missions.*

## 💡 The Idea

Drones that **don't trust a single view blindly** and **don't depend on a central command**:

1. **Anti-Deception Eye** — every frame goes through a detector *and* independent, model-free tamper checks. Tampered inputs are flagged instead of silently trusted.
2. **GPS-Free Swarm Brain** — drones coordinate from local observations only (separation, alignment, cohesion), so the swarm reorganises when members are lost.

---

## 📍 Status — what is implemented today

| Component | State |
|---|---|
| Detector (Stream 1) — **RF-DETR** (production, chosen on measured results) or YOLO (optional), one interface | ✅ `MODEL/detectors.py` |
| FFT patch detector (Stream 2), per-object scoring, calibration | ✅ `MODEL/fft_scorer.py`, `MODEL/calibration.py` |
| Temporal-consistency cue (Stream 3) for low-frequency attacks, fused live | ✅ `MODEL/temporal_scorer.py` |
| Cleaned multi-source dataset (6 sources, cross-source de-duplication) + training + head-to-head evaluation | ✅ `MODEL/build_dataset.py`, `train_detector.py`, `compare_detectors.py` |
| Independent ground-vehicle test set + dataset registry | ✅ `MODEL/build_ground_test.py`, `MODEL/datasets_registry.yaml` |
| Model release on the Hugging Face Hub (gated, CC BY-NC 4.0) + verified download | ✅ `MODEL/publish_model.py`, `MODEL/fetch_weights.py` |
| FastAPI backend (WebSocket stream, live settings, health, metrics) | ✅ `Frontend_Backend/backend/server.py` |
| React tactical dashboard | ✅ `Frontend_Backend/dashboard-app/` |
| Swarm flocking | ⚙️ Hand-coded boid rules (dashboard canvas + `SCRIPTS/kaal_swarm.py`) |
| MAPPO swarm policy, PyBullet/Gazebo sim, thermal/IR fallback | 🗺️ Planned — see `roadmap.md` (Phase 6) |

---

## 🧭 V3 direction — trustworthy perception first

KAAL V3 is the next software-release direction, not a claim that a separate V3 implementation or model checkpoint already exists. The current production detector remains the registry-selected RF-DETR model; preserve it as the baseline until a candidate passes the same evaluation gates.

V3 priorities:

1. **Truthful operating modes:** label telemetry as `LIVE`, `REPLAY`, or `SIMULATION`; never present generated coordinates or simulated values as live sensor readings.
2. **Explicit detector health:** surface model load state and fallback/demo mode in the API and dashboard. Missing or invalid weights must not look like successful object detection.
3. **Calibrated measurements:** mark distance estimates as approximate unless the camera and object assumptions are calibrated; represent unavailable measurements explicitly.
4. **Temporal-cue robustness:** evaluate false alarms under camera motion, illumination changes, scene cuts, and moving objects before relying on the static-scene cue.
5. **Deployment security:** require authentication for exposed settings endpoints, validate configuration, and test stale-feed and reconnect behavior.
6. **Reproducible promotion gates:** record dataset/version provenance, class mapping, model hash, clean and perturbed metrics, false-flag rate, latency, and memory. Do not promote a model solely because one aggregate metric improves.

Suggested first milestone: add explicit operating-mode and detector-health fields to the backend contract, reflect them in the dashboard, and cover fallback/stale-feed behavior with regression tests. Keep V3 focused on monitoring, robustness evaluation, and simulation; consequential decisions remain human-approved.

---

## 🚀 Layer 1 — The Anti-Deception Eye

```
Camera frame
  ├─► Stream 1  Detector (RF-DETR / YOLO) ──► boxes, classes, confidences
  ├─► Stream 2  FFT high-frequency power ratio, whole frame + per detected box
  └─► Stream 3  Temporal residual vs. a learned background (video only)
                         │
                 Fusion: flag the frame / each box if any stream exceeds its
                 calibrated threshold → dashboard alert, "predictions unreliable"
```

* **Stream 2 (FFT).** The score is the share of spectral *power* above `0.5·(1−high_freq_ratio)` cycles/pixel, computed on a mean-removed, Hann-windowed crop. Natural images follow a 1/f² law; printed noise/pattern patches inject high-frequency power. The band is defined in cycles/pixel, so tall, wide and square boxes are scored on equal terms. (The original scorer used `log1p` magnitudes and an index-space radius; it tracked box *shape* rather than content and was near chance on real photos — AUROC 0.59 vs 0.89 now. See `MODEL/RESULTS.md`.)
* **Stream 3 (temporal).** An EMA background model on a 32×32 grid; each cell's residual (after removing the global brightness shift) is compared with a warm-up-calibrated threshold. A frame is flagged when a cell stays hot for several consecutive frames — a patch fixed on a target does, a vehicle driving through does not. Hot cells are not folded into the background, cells hot for too long are absorbed (a parked vehicle), and a whole-view change re-baselines. It catches the smooth, low-frequency perturbations the FFT cannot see, and small printed patches the frame-level FFT dilutes. It assumes a static or slowly moving camera.
* **What the defense does:** it *flags*. It does not repair detections, so mAP under attack is unchanged; the defense metric is the flag rate vs. false-flag rate.

### Detector: RF-DETR instead of YOLO

[RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, ICLR 2026) is a real-time detection transformer on a DINOv2 backbone: NMS-free, Apache-2.0 (Ultralytics YOLO is AGPL-3.0), and the leader on the RF100-VL domain-transfer benchmark — which is what matters when fine-tuning on a small, specialised dataset like ours. Trained on identical data on the same GPU (`MODEL/RESULTS.md` §6):

| | RF-DETR-Small | YOLO26s |
|---|---|---|
| kaal_v3 test mAP50:95, clean / patched | **0.552 / 0.516** | 0.517 / 0.472 |
| Unseen ground-vehicle photos (798), any-class mAP50 | **0.722** | 0.015 |
| Latency (RTX 4060, FP32, batch 1) | 17.1 ms | 7.5 ms |

RF-DETR is therefore the production detector (`Docs/DECISIONS.md`, ADR-001); YOLO stays available (`KAAL_DETECTOR=yolo`) for speed-critical edge targets.

### Production model — `rfdetr-small-v4`

RF-DETR-Small fine-tuned on `kaal_v4` (warm start from v3, 10 epochs, 2.43 h on an RTX 4060 Laptop). Full card: [`Docs/MODEL_CARD.md`](Docs/MODEL_CARD.md); numbers: `MODEL/RESULTS.md` §7.

| | v3 (kaal_v3) | **v4 (kaal_v4)** |
|---|---|---|
| kaal_v4 test mAP50:95, clean / patched | 0.313 / 0.291 | **0.405 / 0.379** |
| Unseen ground vehicles (798 photos), any-class mAP50 | 0.720 | **0.915** |
| soldier / artillery / tank (kaal_v4 test, mAP50:95) | 0.00 / 0.07 / 0.53 | **0.30 / 0.26 / 0.66** |
| ONNX export parity (ΔmAP50:95, 200 val images) | — | +0.005 (gate ≤ 0.01) |

The backend loads it automatically (registry `status: production`); `python -m MODEL.export_model` produces the ONNX version for torch-free deployment.

**Getting the weights.** They are not in git. `python -m MODEL.fetch_weights` downloads them from the Hugging Face Hub ([`ak-2468/kaal-rfdetr-small`](https://huggingface.co/ak-2468/kaal-rfdetr-small), tag `v4`) at the commit pinned in the registry and checks every file's SHA-256 before it is used. The model repo is gated: accept the use conditions on its page once (non-commercial, no weapon targeting, no identification of individuals, human review), then `hf auth login`. Licence: CC BY-NC 4.0, because of the training data (`Docs/DECISIONS.md` ADR-008).

## 🛰️ Layer 2 — GPS-Free Swarm Brain

Each drone acts on local knowledge only — its own velocity, objects within sight, and the bearing/distance of its nearest neighbours — following three rules: **separation**, **alignment**, **cohesion**. Today this runs as hand-coded boids in the dashboard simulation; a learned MAPPO policy is the Phase 6 roadmap item.

---

## 🗂️ Dataset (`kaal_v4`)

`python -m MODEL.build_dataset` merges six sources into one YOLO-format dataset with a coarse 9-class taxonomy that both RF-DETR and YOLO train on (full lineage, licences and data findings: [`Docs/DATASETS.md`](Docs/DATASETS.md); machine-readable registry: `MODEL/datasets_registry.yaml`):

`aircraft · helicopter · tank · armored_vehicle · artillery · engineering_vehicle · soldier · vehicle · person`

| Source | Content | Notes / license |
|---|---|---|
| Original KAAL set (`MODEL/runs/adversarial/augmented_dataset`) | aircraft types, tanks, APC/IFV, artillery, soldiers, aerial cars | Legacy class 0 (“A10”) mixed cars and generic aircraft — those images are **dropped**. Augmented copies of one photo never straddle splits. |
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | drone-view people & vehicles | People map to `person`, **never** `soldier`. Academic use. |
| [Military Aircraft Detection](https://huggingface.co/datasets/a2015003713/military-aircraft-detection-dataset) | 114 aircraft types (rarity-weighted subset) | Rotorcraft → `helicopter`. No license declared upstream. |
| [Simuletic UAV battle tank](https://huggingface.co/datasets/Simuletic/UAV-Aerial-View-Battle-Tank-Detection-Dataset) | synthetic drone-view tanks | CC BY-NC 4.0 per HF metadata. |
| [DVIDS military (llama-farm)](https://huggingface.co/datasets/llama-farm/military-labeled-yolo) | soldiers, tanks, APCs, artillery, MLRS, trucks, aircraft | U.S. public-domain imagery, CC0 labels (VLM-assisted, human-corrected). |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) — Tank class | real-world tank photos | annotations CC BY 4.0, images CC BY 2.0. Photos with people are excluded. |

Near-duplicate photos are grouped **across all sources** before splitting (the legacy and aircraft sources share images), and an independent test-only set (`python -m MODEL.build_ground_test`: Orion web images, Russian-military-annotated, TNO Search_2 — 798 photos) measures generalisation to imagery no model has seen.

```bash
python -m MODEL.fetch_datasets uav_tank
python -m MODEL.fetch_datasets aircraft --train 3500 --val 400 --test 400
python -m MODEL.fetch_datasets dvids
python -m MODEL.fetch_datasets openimages_tank
gh release download v0.0.0 --repo ultralytics/assets \
    --pattern 'VisDrone2019-DET-*.zip' --dir MODEL/Datasets/visdrone   # then unzip
python -m MODEL.build_dataset            # -> MODEL/Datasets/kaal_v4 (+ manifest.json)
```

## 🏋️ Training & evaluation

```bash
# production recipe: continue the RF-DETR checkpoint on kaal_v4
python -m MODEL.train_detector --arch rfdetr-small --data MODEL/Datasets/kaal_v4 \
    --init-weights MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth \
    --epochs 10 --lr 5e-5 --name rfdetr-small-v4
# from scratch (COCO weights) / YOLO baseline
python -m MODEL.train_detector --arch rfdetr-small --epochs 12
python -m MODEL.train_detector --arch yolo26s --epochs 30
# head-to-head report (add --dedup-against, --eval-classes, --agnostic as needed)
python -m MODEL.compare_detectors \
    --detector rfdetr-s=MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth \
    --detector yolo26s=MODEL/runs/train/yolo26s/weights/best.pt     # -> MODEL/runs/compare/report.md
```

---

## 📚 Documentation

| Document | Contents |
|---|---|
| [`Docs/WALKTHROUGH.md`](Docs/WALKTHROUGH.md) | deep structural walkthrough: architecture, live pipeline frame by frame, every subsystem, runbook |
| [`Docs/KAAL_V3_ROADMAP.md`](Docs/KAAL_V3_ROADMAP.md) | V3 release gates, phased execution plan, acceptance criteria, and next tasks |
| [`Docs/FILE_REFERENCE.md`](Docs/FILE_REFERENCE.md) | every file, its status (core / tool / research / legacy) and purpose |
| [`Docs/DATASETS.md`](Docs/DATASETS.md) | dataset lineage, class mappings, licences, data problems found, how to add a dataset |
| [`Docs/MODEL_CARD.md`](Docs/MODEL_CARD.md) | production detector: data, metrics, limits, intended use |
| [`Docs/DECISIONS.md`](Docs/DECISIONS.md) | architecture decision records (RF-DETR, FFT scorer, temporal cue, data policy, security) |
| [`Docs/FUTURE_SCALING.md`](Docs/FUTURE_SCALING.md) | scaling to new models, tasks and hardware tiers with concrete tech stacks |
| [`MODEL/RESULTS.md`](MODEL/RESULTS.md) | every measured result |

---

## 🛠️ Installation

```bash
git clone https://github.com/aryank2212/KAAL_v2.git && cd KAAL_v2
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -e ".[ml,server,hub]" # detectors + training + backend + Hub download
# pip install -e ".[server]"      # model-free defense + backend only (demo mode, no detector)

hf auth login                      # once, after accepting the terms on the model page
python -m MODEL.fetch_weights      # production weights (.pth + .onnx), SHA-256 verified
```

### Export for deployment

```bash
pip install -e ".[export]"
python -m MODEL.export_model --parity 200        # production RF-DETR -> ONNX, fails if mAP moves > 0.01
python -m MODEL.export_model --format tensorrt   # also openvino, coreml, tflite, ... (RF-DETR exporter)
pip install -e ".[onnx,server,hub]"              # edge box: ONNX Runtime, no torch
python -m MODEL.fetch_weights --format onnx      # the backend then loads the registry's ONNX file
KAAL_MODEL_PATH=MODEL/runs/export/<run>/<run>.onnx python Frontend_Backend/backend/server.py   # or name one explicitly
```

### Backend

```bash
python Frontend_Backend/backend/server.py
```

Serves `http://127.0.0.1:3000`: `/api/health`, `/api/python-data`, `/api/settings` (GET/PUT), `/api/video_feed` (MJPEG) and the `/ws` WebSocket stream.

| Variable | Default | Purpose |
|---|---|---|
| `KAAL_MODEL_PATH` | production model from `MODEL/datasets_registry.yaml` (`.pth`, else `.onnx`) if on disk, else `MODEL/yolov8m.pt` | `.pth`/`.ckpt` → RF-DETR, `.onnx` → ONNX Runtime (no torch), `.pt` → YOLO |
| `KAAL_DETECTOR` | `auto` | force `rfdetr`, `onnx` or `yolo` (auto = from the weights suffix) |
| `KAAL_HOST` | `127.0.0.1` | bind address (local only by default) |
| `KAAL_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed dashboard origins (HTTP + WebSocket) |
| `KAAL_API_TOKEN` | — | require `X-KAAL-Token` on `PUT /api/settings` (set `VITE_KAAL_API_TOKEN` for the dashboard) |
| `KAAL_FFT_THRESHOLD` / `KAAL_CALIBRATION` | `0.08` / — | FFT threshold or a calibration profile JSON |
| `KAAL_TEMPORAL`, `KAAL_TEMPORAL_THRESHOLD`, `KAAL_TEMPORAL_WARMUP` | `1`, auto, `30` | temporal cue on/off, fixed threshold, warm-up frames |
| `KAAL_CAMERA_INDEX`, `KAAL_ENABLE_CAMERA`, `KAAL_CONF` | `0`, `1`, `0.4` | camera + detector confidence |
| `KAAL_CAMERA_SOURCE` | — | video file or stream URL instead of a webcam (files loop) |

If the ML stack or weights are missing, the server starts in **demo mode** (no detector; FFT, temporal cue, metrics and streams still run).

**Attack demo without a webcam:** render a clean → patched → clean clip from any labelled drone frame and play it through the full pipeline:

```bash
python -m MODEL.make_demo_video --image MODEL/Datasets/kaal_v4/test/images/<frame>.jpg --out demo.mp4
KAAL_CAMERA_SOURCE=demo.mp4 python Frontend_Backend/backend/server.py   # production weights from the registry
```

### Dashboard

```bash
cd Frontend_Backend/dashboard-app
npm ci
npm run dev        # http://localhost:5173 (proxies /api and /ws to :3000)
```

---

## 🧪 Testing & CI

GitHub Actions (`.github/workflows/ci.yml`) on every push and PR:

1. **Lint** — `ruff check` + `ruff format --check` on `MODEL`, `tests`, `Frontend_Backend/backend`
2. **Type check** — `mypy` (MODEL, tests, backend)
3. **Tests** — `pytest -m "not gpu"` on Python 3.11 and 3.12, including the server integration tests
4. **Dashboard** — `npm ci`, `eslint`, `vite build`

The GPU end-to-end suite (`integration.yml`) is manual (`workflow_dispatch`) and needs a self-hosted CUDA runner.

```bash
pip install -e ".[dev,server]"
ruff check MODEL tests Frontend_Backend/backend && mypy && pytest
```

---

## 📁 Project structure

```
KAAL/
├── MODEL/
│   ├── detectors.py            # YOLO + RF-DETR behind one interface
│   ├── fft_scorer.py           # Stream 2: FFT power-ratio scorer
│   ├── temporal_scorer.py      # Stream 3: temporal cue (+ streaming wrapper)
│   ├── dual_stream_defense.py  # fusion of all streams
│   ├── calibration.py          # per-camera threshold profiles
│   ├── fetch_datasets.py / build_dataset.py
│   ├── train_detector.py / compare_detectors.py
│   ├── adversarial_patch.py, augmentations.py, benchmark.py, failure_analysis.py, ablation.py
│   └── RESULTS.md              # measured results
├── Frontend_Backend/
│   ├── backend/server.py       # FastAPI + WebSocket
│   └── dashboard-app/          # React + Vite dashboard
├── SCRIPTS/                    # AirSim drone control & swarm demos
├── tests/                      # unit + integration tests
└── roadmap.md, progress.md
```

---
*“KAAL is a swarm intelligence system where the drones share the instincts of a flock of birds and the paranoia of a soldier — they need no signal to coordinate, and they trust nothing their cameras show them.”*
