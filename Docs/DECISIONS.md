# KAAL — Architecture Decision Records

Each record states the context, the decision, the evidence and the
consequences. Numbers refer to `MODEL/RESULTS.md`.

---

## ADR-001 — RF-DETR is the production detector; YOLO is an optional backend

**Status:** accepted (2026-10-11)

**Context.** The original KAAL detector was YOLOv8 (`MODEL/kaal_eye_v2.pt`,
59 classes). The goal was a detector that is more accurate, more robust under
adversarial patches and that generalises to imagery it was not trained on.
The budget allows training and maintaining **one** model.

**Options considered.**

| Option | Notes |
|---|---|
| YOLO26s (Ultralytics, AGPL-3.0) | CNN detector with an end-to-end (NMS-free) head, fastest, mature tooling. |
| **RF-DETR** (Roboflow, Apache-2.0, ICLR 2026) | Real-time DETR on a DINOv2 backbone, NMS-free; best published transfer on the RF100-VL benchmark; `pip install rfdetr`; detection sizes nano → large, segmentation nano → 2xlarge; ONNX export (TensorRT via `trtexec`). |
| RT-DETRv2 / D-FINE / DEIM-family | Strong COCO numbers; weaker fine-tuning tooling, no single maintained package with the same export/seg story. Revisit in FUTURE_SCALING. |

**Evidence (same training data, same GPU, `MODEL/RESULTS.md` §6).**

| | RF-DETR-Small | YOLO26s |
|---|---|---|
| kaal_v3 test mAP50:95 (clean / attacked) | **0.552 / 0.516** | 0.517 / 0.472 |
| … after removing near-duplicates of train | **0.552** | 0.516 |
| Unseen ground-vehicle photos, any-class mAP50 | **0.722** | 0.015 |
| Latency, RTX 4060, FP32, batch 1 | 17.1 ms | 7.5 ms |
| Training time used | 1.99 h (12 epochs) | 1.44 h (30 epochs, plateaued) |

**Decision.** RF-DETR (Small) is the default detector everywhere
(`train_detector`, `compare_detectors`, backend, docs). YOLO remains
selectable — `KAAL_DETECTOR=yolo` / `.pt` weights — for edge targets where
the 2.3× speed matters more than accuracy and generalisation.

**Consequences.** All new training data and fine-tuning go to RF-DETR.
YOLO receives no new training runs; its kaal_v3 baseline weights are kept.
The DINOv2 backbone needs more VRAM to train (batch 4 + grad-accum 4 on
8 GB) and RF-DETR inference is slower; ONNX/TensorRT export
(`optimize_for_inference`, FP16) is the planned mitigation.

---

## ADR-002 — FFT anomaly score: spectral power ratio in cycles/pixel

**Status:** accepted (2026-10-11)

**Context.** The original Stream 2 score (`log1p|F|` energy outside an
index-space circle) tracked crop **shape**: the same photo scored 0.50 as a
square and 0.91 as a 600×100 strip, and white noise scored the same as a
photo (§0). Per-box scoring flagged every tall/wide detection.

**Decision.** Score = share of AC spectral **power** above
`0.5·(1−high_freq_ratio)` cycles/pixel on a mean-removed, Hann-windowed
crop. `DEFAULT_FFT_THRESHOLD = 0.08` (≈ p93 of clean kaal_v3 boxes), one
shared constant for model, backend and dashboard.

**Evidence.** AUROC vs printed-style patches on real photos 0.59 → 0.89;
shape spread 0.50–0.91 → 0.068–0.079. On real kaal_v3 boxes per-box AUROC
is 0.74 (many boxes are tiny) — FFT alone is a weak per-object signal.

**Consequences.** Calibrate per camera (`python -m MODEL.calibration
--labels`). Do not rely on FFT alone; fuse with Stream 3 (ADR-003).

---

## ADR-003 — Streaming temporal cue (Stream 3) fused into the live defense

**Status:** accepted (2026-10-11)

**Context.** FFT is blind to smooth, low-frequency perturbations (§1, 0%
detection) and a frame-level score dilutes small patches.

**Decision.** `StreamingTemporalCue`: per-cell EMA background on a 32×32
grid, median illumination normalisation, warm-up calibration (p99.9 × 1.5,
floor 0.01), a frame is flagged when a cell stays hot for 5 frames; hot
cells are not absorbed into the background, cells hot for too long are
(parked vehicle), a whole-view change re-baselines.

**Evidence.** Low-frequency attacks 0% (FFT) → 100% (temporal), clean FPR
0%. Live demo with trained RF-DETR (220-frame clip, patch on one of seven
tanks): 0 false alarms on 110 clean frames, attack flagged 4 frames
(0.2 s) after onset on 76/80 patched frames, cleared on the first clean
frame; the per-box FFT did not fire on that patch.

**Consequences.** Assumes a static or slowly moving camera; a moving drone
needs ego-motion compensation (FUTURE_SCALING §2).

---

## ADR-004 — The defense flags, it does not repair

**Status:** accepted. The streams decide *whether to trust* a frame or box;
they do not modify detections. "mAP recovery" is therefore not a metric —
the defense metric is the flag rate on attacked inputs vs the false-flag
rate on clean inputs. Evaluation code reports exactly that.

---

## ADR-005 — Data policy

**Status:** accepted (2026-10-11)

1. **One coarse 9-class taxonomy** (`aircraft, helicopter, tank,
   armored_vehicle, artillery, engineering_vehicle, soldier, vehicle,
   person`) shared by every source and both architectures.
2. **People are `person`, never `soldier`, unless a source explicitly labels
   military personnel.** Mislabelling civilians as combatants is the error we
   refuse to train in. Open Images tank photos with people are dropped rather
   than labelling crews `person` (conflicting with DVIDS `soldier`) or leaving
   them unlabelled.
3. **Unlabelled ≠ empty.** Images without annotations are not used as
   negatives (254 unlabelled legacy strike-footage frames dropped).
4. **No photo in two splits — across all sources.** Source-photo grouping
   (Roboflow `.rf.<hash>`, `_noisy/_occlude/_patch`), flip-aware 256-bit dHash
   near-duplicate grouping across every source, test > val > train priority.
5. **Independent test data is never trained on.** `kaal_ground_test`
   (Orion / Russian-military / Search_2) is test-only and rebuilt against each
   new training set to exclude near-duplicates.
6. **Every dataset is registered** in `MODEL/datasets_registry.yaml` with
   licence, status and which model used it; a test keeps it in sync.
7. **Licences are carried, not hidden.** VisDrone and the Simuletic sample
   are non-commercial; the legacy `mil_*` origin is unknown. A commercial
   deployment must retrain without them (DATASETS.md).

---

## ADR-006 — Backend security defaults

**Status:** accepted (Phase 4.5). Bind `127.0.0.1`; CORS allow-list that also
gates WebSocket `Origin`; optional `X-KAAL-Token` on `PUT /api/settings`
compared with `secrets.compare_digest`; Pydantic finite-number validation and
a NaN-safe 422 handler; drop-oldest broadcast queue so a slow client cannot
grow memory.

---

## ADR-007 — Continue from the trained checkpoint instead of retraining

**Status:** accepted (2026-10-11). With ~3 GPU-hours available, `kaal_v4`
training warm-starts RF-DETR-Small from the kaal_v3 checkpoint
(`train_detector --init-weights`) — same taxonomy, so the detection head is
reused — rather than starting from COCO. A larger model (RF-DETR-Medium,
higher resolution for small objects) is the next step when more GPU time is
available (FUTURE_SCALING §1).
