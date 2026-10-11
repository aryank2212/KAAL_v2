# KAAL — Future Scaling Guide

How KAAL grows from today's system (RF-DETR-Small + FFT + temporal cue,
FastAPI backend, React dashboard, one RTX 4060) to newer models, new tasks
and any hardware tier — with the concrete technology for each step.

Companion documents: [`WALKTHROUGH.md`](WALKTHROUGH.md) (what exists),
[`DECISIONS.md`](DECISIONS.md) (why), [`DATASETS.md`](DATASETS.md) (data),
[`MODEL_CARD.md`](MODEL_CARD.md) (current model).

---

## 0. Ground rules for every extension

These are design constraints, not afterthoughts:

* **Human in the loop.** KAAL perceives, flags and reports. Any action with
  real-world consequences stays a human decision with an audit trail. No
  autonomous engagement or weapon-targeting features belong in this codebase.
* **Robustness is a feature.** Every new model or task ships with an attacked
  evaluation (clean vs patched) and a false-alarm budget, as
  `compare_detectors` does today.
* **People are not targets.** `person` ≠ `soldier` unless a trusted source says
  so (ADR-005); no face recognition or individual re-identification.
* **Licences travel with data.** Non-commercial sources (VisDrone, Simuletic)
  must be removed before any commercial use; the registry
  (`MODEL/datasets_registry.yaml`) makes that a filter, not an archaeology
  project.
* **Same evaluation, every time.** A change is accepted only if it beats the
  production model on `kaal_v4` test **and** `kaal_ground_test` without
  raising the clean false-flag rate.

---

## 1. Detector scaling (Stream 1)

### 1.1 Bigger / sharper RF-DETR (no code change)

`train_detector` already supports every size; `--imgsz` sets resolution.

| Step | Command | Expected effect | Cost vs Small |
|---|---|---|---|
| RF-DETR-Medium (576 px) | `--arch rfdetr-medium` | ~+2 COCO AP over Small; better small objects | estimate ~1.5× train and latency |
| Higher input | `--imgsz 640` / `768` (must be a multiple of 32 = patch 16 × 2 windows) | VisDrone people/vehicles (today 0.07 / 0.35 mAP50:95) | ∝ pixels |
| RF-DETR-Large (704 px) | add `"rfdetr-large": "large"` to `RFDETR_ARCHS` in `train_detector.py` (`detectors.py` already maps it) | best accuracy of the family | estimate ~2–3× train time; plan for a ≥ 16 GB card |
| Longer schedule | `--epochs 24` on kaal_v4 | convergence on new sources | linear |

### 1.2 Small and distant objects

* **Tiled inference (SAHI)** — slice 1024–6000 px frames into overlapping
  tiles, detect per tile, merge with NMS/WBF. Libraries: `sahi`,
  `supervision.InferenceSlicer`. Directly addresses Search_2-style long-range
  targets and VisDrone crowds.
* **Tiled training** — cut training images into tiles so small objects are
  seen at native resolution.
* **Super-resolution front-end** for long range (Real-ESRGAN / SwinIR) —
  only behind an A/B gate; SR can hallucinate.

### 1.3 New architectures to evaluate (same harness)

| Family | Why | Integration |
|---|---|---|
| RF-DETR-Seg | instance masks: footprint, occlusion reasoning, cleaner FFT crops | `rfdetr` segmentation configs; masks into `DualStreamDefense` crops |
| D-FINE / DEIM / DEIMv2 (DINOv3 backbones) | strong real-time DETRs | add a `Detector` adapter in `MODEL/detectors.py` |
| RT-DETRv2 | mature real-time DETR | adapter |
| YOLO (Ultralytics) | fastest edge baseline — already supported | `KAAL_DETECTOR=yolo` |

Adding a backend = one class implementing `predict(image, conf) -> list[dict]`
in `MODEL/detectors.py`; everything else (defense, backend, evaluation)
works unchanged.

### 1.4 Open-vocabulary and foundation models

* **Zero-shot detection** of classes with no labels (e.g. "air-defense radar",
  "pontoon bridge"): Grounding DINO 1.5, OWLv2, YOLO-World, Florence-2.
  Use them as **labelers**, not as the runtime detector: they are slower and
  less calibrated.
* **VLM-assisted labeling with human review** (how the DVIDS set was made):
  Florence-2 / Qwen2.5-VL / Gemini propose boxes → CVAT or Label Studio for
  correction → registry entry → `build_dataset` adapter.
* **Segment Anything 2** to turn boxes into masks for RF-DETR-Seg training.

### 1.5 Fine-grained identification

A second-stage classifier on detected crops (e.g. T-72 vs T-90, BMP-2 vs
BTR-80) trained on classification sets such as
`leibnitz-lab/military_vehicles` (MIT; registry: `military_vehicles_cls`).
Stack: timm / DINOv2-small linear probe → ONNX. Keeps the detector's 9-class
head stable.

### 1.6 Tracking and multi-frame reasoning

* Multi-object tracking: ByteTrack, BoT-SORT, OC-SORT (`supervision`,
  `trackers`, Ultralytics trackers). Gives stable IDs and per-track
  confidence smoothing.
* **Track-level defense:** a patch that suddenly flips a track's class or
  confidence is suspicious — a fourth, cheap cue that complements Stream 3.

### 1.7 Compression and distillation for the edge

* Knowledge distillation RF-DETR-Medium → RF-DETR-Nano / YOLO26n.
* Post-training quantisation (TensorRT INT8 with a calibration set from
  `kaal_v4` val; ONNX Runtime QDQ), quantisation-aware training if INT8 drops
  > 1 mAP. DETR attention is sensitive to INT8 — keep FP16 for attention
  blocks if needed.
* Structured pruning (Torch-Pruning) only with a re-fine-tune.

---

## 2. Defense scaling (Streams 2 and 3)

| Gap today | Extension | Technology |
|---|---|---|
| Temporal cue assumes a static camera | ego-motion compensation: register frame t-1 onto t before the residual | OpenCV ORB + RANSAC homography; dense optical flow (RAFT / NVIDIA Optical Flow SDK on Jetson) |
| Per-box FFT weak on tiny boxes (AUROC 0.74) | learned patch detector on crops | small CNN/ViT trained on real vs patched crops from `adversarial_patch.py`; evaluate with the same AUROC harness |
| No certified guarantee | certifiable patch defenses | PatchCleanser, ObjectSeeker, DetectorGuard (research code → adapters) |
| Detector itself not hardened | patch-aware adversarial training | EOT physical patches (perspective, lighting, print noise) during training; `MODEL/training/adv_train.py` is the starting point |
| Single viewpoint | **swarm multi-view consistency** — a patch fools one camera angle, rarely all | per-drone detections → shared map (geo-referenced) → disagreement flag |
| RGB only | sensor fusion | thermal (FLIR Boson / Lepton), RGB-T fusion detectors; a printed patch has no thermal signature |
| Out-of-distribution inputs | OOD / uncertainty monitor | energy score or Mahalanobis distance on backbone features; MC-dropout-free ensembles of heads |
| Model supply chain | integrity | signed weights (Sigstore `cosign`), SBOM (`syft`), hash check at load |

Benchmarks to adopt: APRICOT (physical patches), the ATR datasets used in
`kaal_ground_test`, and a KAAL red-team set generated by
`adversarial_patch.py` with EOT.

---

## 3. New tasks

| Task | What it adds | Data (examples) | Model / stack |
|---|---|---|---|
| Counter-UAS (detect drones) | KAAL swarms seeing other drones | Anti-UAV410, seraphim drone set (CC BY 4.0), Simuletic swarm set (CC BY 4.0) | new `drone` class in the taxonomy; RF-DETR fine-tune; tracking |
| Thermal / night | 24 h operation | HIT-UAV, FLIR ADAS, KAIST multispectral | RGB-T detector or separate thermal RF-DETR |
| Search & rescue / disaster response | humanitarian use of the same pipeline | HERIDAL, AIDER, xBD | person/vehicle/damage classes |
| Change detection | what moved since the last pass | paired aerial captures | Siamese ViT, ChangeFormer |
| Geo-localisation of detections | map coordinates instead of pixels | drone pose + DEM | camera model + `pyproj`; GDAL; Cesium/deck.gl display |
| Mapping / SLAM | GPS-denied navigation (KAAL Layer 2) | own flights, EuRoC, TUM-VI | ORB-SLAM3, VINS-Fusion, OpenVINS; Isaac ROS Visual SLAM on Jetson |
| Swarm coordination (MAPPO) | learned flocking + task allocation (roadmap Phase 6) | simulation | PettingZoo / VMAS environments; BenchMARL, MARLlib or EPyMARL; Ray RLlib |
| Semantic segmentation of terrain | traversability, cover | LoveDA, UAVid | SegFormer / Mask2Former |

Taxonomy changes go through `build_dataset.TAXONOMY`, the registry and a
fresh training run — never by silently re-purposing a class id.

---

## 4. Data scaling

* **Synthetic data:** NVIDIA Isaac Sim Replicator, Unreal Engine + AirSim
  (Colosseum fork), CARLA; domain randomisation (lighting, weather, camo,
  altitude). AMOD-V1.0 (647 GB, CC BY-NC) is the ready-made option once
  storage allows (registry: `amod_v1`).
* **Active learning loop:** run the production model on new footage →
  FiftyOne (uniqueness, mistakenness, low-confidence) → CVAT / Label Studio →
  new registry entry → `build_dataset` → `compare_detectors` gate.
* **Dataset versioning:** DVC or lakeFS on top of the registry so every model
  points at an immutable dataset hash.
* **Label quality:** cleanlab / FiftyOne mistakenness on existing sets; the
  legacy `mil_*` frames are the first candidate for relabelling.
* **Account-gated sources** to add when credentials are available: Roboflow
  Universe military sets, Kaggle "Military Assets", xView, DOTA (registry:
  `account_gated`).

---

## 5. MLOps

| Concern | Today | Next | Stack |
|---|---|---|---|
| Experiment tracking | `metrics.csv`, TensorBoard, `kaal_train_summary.json` | central tracking + artefacts | MLflow (self-hosted) or Weights & Biases |
| Model registry | `models:` section of the dataset registry | versioned, signed models with stage tags | MLflow Model Registry |
| Evaluation gate | manual `compare_detectors` | CI job on a GPU runner, fails on regression | GitHub Actions self-hosted runner (`integration.yml`) |
| Data lineage | `datasets_registry.yaml` + test | content hashes | DVC / lakeFS |
| Training scale | one GPU, grad-accum | multi-GPU / multi-node | PyTorch DDP via Lightning (RF-DETR trains on Lightning), Ray Train, Slurm |
| Serving | in-process detector in FastAPI | model server | NVIDIA Triton (ONNX/TensorRT), Ray Serve, KServe |
| Monitoring | 1 Hz metrics to the dashboard | metrics + drift | Prometheus + Grafana, OpenTelemetry, Evidently (embedding drift) |
| Packaging | `pip install -e .[ml,server]` | images | Docker (CUDA base), docker-compose → Helm; `nvcr.io/nvidia/l4t-*` images on Jetson |

---

## 6. Hardware tiers — exact stacks

The same model family runs everywhere; what changes is export format,
precision, resolution and model size.

| Tier | Example hardware | Runtime stack | Model / precision | Expected throughput* |
|---|---|---|---|---|
| T0 — CPU only | laptop CPU, Intel NUC, Raspberry Pi 5 | ONNX Runtime (CPU EP) or Intel OpenVINO | RF-DETR-Nano or YOLO26n, 320–384 px, INT8 | ~2–15 FPS |
| T1 — NPU add-on | Hailo-8/8L (Raspberry Pi AI Kit), Rockchip RK3588, Qualcomm (AI Hub/QNN), Apple Neural Engine | Hailo Dataflow Compiler + HailoRT; RKNN-Toolkit2; Qualcomm AI Engine Direct; Core ML | YOLO-class CNN INT8 first (best supported); DETRs need operator checks per NPU | ~20–60 FPS |
| T2 — Embedded GPU | NVIDIA Jetson Orin Nano / NX / AGX | JetPack 6, TensorRT FP16/INT8, DeepStream 7 for multi-camera, ROS 2 + Isaac ROS | RF-DETR-Nano/Small FP16 at 384–512 px | Orin Nano ~10–25 FPS, AGX ~60+ FPS |
| T3 — Workstation GPU | RTX 4060 → 4090 (today: 4060 Laptop) | PyTorch / ONNX Runtime CUDA EP / TensorRT | RF-DETR-Small/Medium FP16 | today 17 ms FP32 unoptimised (~58 FPS); TensorRT FP16 typically 2–4× faster |
| T4 — Datacenter GPU | L4, L40S, A100, H100 | Triton Inference Server with TensorRT backend, dynamic batching; Kubernetes + GPU operator | RF-DETR-Large, many streams per GPU | hundreds of FPS aggregate |
| T5 — Multi-node | GPU cluster | Ray / Slurm, PyTorch DDP/FSDP, DeepSpeed for very large backbones | training-time scale (synthetic pre-training, Large/2XL) | — |

\* Order-of-magnitude planning figures, not measurements; measure on the
target with `compare_detectors` (latency column) before committing.

**Training compute per tier:** a T3 laptop GPU fine-tunes Small (batch 4 ×
grad-accum 4, ~10 img/s); a 24 GB card trains Medium at batch 8; Large and
synthetic pre-training need T4/T5.

**Export path (planned `MODEL/export_model.py`):** checkpoint →
`model.export()` ONNX (opset ≥ 17) → `trtexec --fp16` engine (T2–T4) /
OpenVINO IR (T0) / vendor compiler (T1) → parity test (≤ 1 mAP drop vs
PyTorch on kaal_v4 val) → signed artefact.

---

## 7. System architecture at scale

```
 drones (T1/T2 edge)                     ground station / cloud (T3/T4)
 ┌───────────────────────────┐           ┌────────────────────────────────────────┐
 │ camera(s) RGB / thermal   │           │ fleet ingest  (NATS / Kafka / Zenoh)   │
 │ detector (TensorRT)       │──events──▶│ fusion: multi-view consistency, tracks │
 │ FFT + temporal + track cue│  + clips  │ storage: TimescaleDB + object store    │
 │ ROS 2 nodes, MAVLink/PX4  │◀─tasks────│ API: FastAPI → gRPC; auth: OIDC        │
 └───────────────────────────┘           │ dashboard: React + deck.gl / Cesium    │
                                         │ MLOps: MLflow, Triton, Grafana         │
                                         └────────────────────────────────────────┘
```

* **Edge ↔ ground messaging:** ROS 2 (DDS) on the drone; Zenoh or NATS across
  unreliable links; MAVLink / MAVSDK to the autopilot (PX4 / ArduPilot).
* **Backend:** keep FastAPI for the dashboard; move high-rate detection
  streams to gRPC or NATS; Redis for live state; TimescaleDB for history.
* **Dashboard:** current React/Vite app + deck.gl or CesiumJS for 3D/geo
  layers; role-based access via Keycloak/OIDC (the current `X-KAAL-Token` is
  a single-operator stopgap).
* **Simulation for development:** PX4 SITL + Gazebo (Harmonic), AirSim /
  Colosseum (existing `SCRIPTS/`), NVIDIA Isaac Sim + Pegasus Simulator for
  photoreal multi-drone scenes.

---

## 8. Phased roadmap (each phase has a measurable exit criterion)

| Phase | Scope | Exit criterion |
|---|---|---|
| 5a | ONNX/TensorRT export + parity test; backend loads the exported engine | ≤ 1 mAP50:95 drop; ≥ 2× speed-up on the RTX 4060 |
| 5b | Jetson Orin deployment (DeepStream or TensorRT Python) | ≥ 15 FPS at 512 px, FFT + temporal included |
| 5c | Ego-motion-compensated temporal cue | clean false-flag rate ≤ 1% on moving-camera clips, attack detection ≥ 90% |
| 5d | RF-DETR-Medium at 640 px on kaal_v4 (+ tiled inference) | VisDrone mAP50:95 ≥ 0.30; ground any-class mAP50 ≥ 0.80 |
| 5e | Fine-grained vehicle head | top-1 ≥ 80% on Russian-military crops |
| 6a | Counter-UAS `drone` class + tracking | AP50 ≥ 0.6 on a held-out anti-UAV split |
| 6b | Multi-view consistency across 3+ simulated drones | patch flag rate ≥ 95% with no single-view false alarms |
| 6c | MAPPO swarm policy in simulation | beats boids on coverage/loss-tolerance metrics |
| 7 | Publication (roadmap Phase 7) | reproducible artefacts from tagged commits |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Non-commercial / unknown-licence data in the training set | registry filter + retrain without them before commercial use |
| VLM-generated labels (DVIDS) carry systematic errors | spot-checks, FiftyOne mistakenness, relabel high-loss images |
| Benchmark leakage | cross-source dHash grouping (kaal_v4), independent `kaal_ground_test`, `--dedup-against` |
| Attack drift (new patch types) | red-team set refreshed each release; temporal + multi-view cues do not depend on patch appearance |
| Edge accelerators lacking DETR operators | keep the YOLO backend as the T1 fallback (ADR-001) |
| Over-trust in automation | human-in-the-loop UI, flags shown as "predictions unreliable", audit log |
