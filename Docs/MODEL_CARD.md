# Model Card — KAAL production detector

| | |
|---|---|
| **Model** | RF-DETR-Small fine-tuned for KAAL — `rfdetr-small-v4` |
| **Status** | production (`MODEL/datasets_registry.yaml` → `models:`; loaded by the backend by default) |
| **Weights** | `MODEL/runs/release/rfdetr-small-v4/rfdetr-small-v4.pth` (127 MB, SHA-256 `fb2a2f0532c138d72ef02c3ddb3ca9ab5bb82e2459a51e0707be71a0b79cc8f0`) — release checkpoint written by `python -m MODEL.publish_model strip` from the training checkpoint `MODEL/runs/train/rfdetr-small-v4/checkpoint_best_total.pth` (SHA-256 `d38e4896dbcf80809d9f635f754d2be5dfa5bb7fd267281caad0a82af9c13d39`); identical predictions, no training state or local paths; not committed to git |
| **Distribution** | Hugging Face Hub: [`ak-2468/kaal-rfdetr-small`](https://huggingface.co/ak-2468/kaal-rfdetr-small), tag `v4` = commit `496fd0e909bce8eb4300aa60aa36b4bfd9b2356a` (pinned in the registry's `hub:` block); public since 2026-10-12 and gated (sign in and accept the use conditions; approval is automatic), CC BY-NC 4.0 — `python -m MODEL.fetch_weights` restores and verifies the files (`Docs/DECISIONS.md` ADR-008) |
| **ONNX** | `MODEL/runs/export/rfdetr-small-v4/rfdetr-small-v4.onnx` (+ `.onnx.json`; SHA-256 `f0c6fe9aab09af6240ffc9766e188b19acb64ef970570b52a67a67b24c288e58`) — run with `KAAL_MODEL_PATH=<file>.onnx`, no torch needed |
| **Architecture** | RF-DETR (Roboflow, Apache-2.0): DINOv2-windowed-small backbone, deformable-attention decoder, 300 queries, NMS-free; input 512 × 512 |
| **Classes (9)** | aircraft, helicopter, tank, armored_vehicle, artillery, engineering_vehicle, soldier, vehicle, person |
| **Owner** | KAAL project (aryank2212/KAAL_v2) |

## Intended use

Perception for drone and ground-camera imagery inside the KAAL dual-stream
defense: detections feed the FFT and temporal tamper checks, and a human
operator sees detections plus "predictions unreliable" flags on the
dashboard.

**Out of scope:** autonomous engagement or weapon targeting; identifying or
tracking individuals; deciding whether a person is a combatant (`soldier` is
a visual label learned from sources that mark military personnel, not a
status determination); any use where a missed or false detection would
cause harm without human review.

## Training

| Item | Value |
|---|---|
| Data | `kaal_v4` train, 16,683 images — legacy KAAL 5,000, VisDrone2019-DET 6,471, military aircraft 3,263, DVIDS military photos 963, Open Images tanks 853, Simuletic UAV tanks 133 (`Docs/DATASETS.md`) |
| Data hygiene | near-duplicates grouped across all sources before splitting (627 cross-split duplicates removed); unlabeled frames dropped; independent ground-vehicle test kept out of training |
| Init | `rfdetr-small-v3` (COCO → kaal_v3, 12 epochs) — warm start, same head |
| Schedule | 10 epochs, lr 5e-5, batch 4 × grad-accum 4, EMA, early stopping patience 4 (not triggered) |
| Hardware / time | RTX 4060 Laptop 8 GB, 2.43 h |
| Validation | mAP50:95 0.462 / mAP50 0.650 on kaal_v4 val (last epoch, still rising) |
| Environment | Python 3.12.14, torch 2.14.1 (CUDA 13.0), rfdetr 1.11.2, pytorch-lightning 2.6.6, supervision 0.30.9, pycocotools 2.0.11; export: onnx 1.23.2, onnxruntime 1.31.0 |

## Evaluation (`MODEL/RESULTS.md` §7)

| Test set | mAP50 | mAP50:95 | attacked mAP50:95 | v3 mAP50:95 |
|---|---|---|---|---|
| kaal_v4 test (1,428) | 0.565 | **0.405** | 0.379 | 0.313 |
| kaal_ground_test (798 unseen photos; tank/APC/artillery/engineering) | 0.172 | 0.119 | 0.121 | 0.090 |
| … any-class (did it find the vehicle?) | **0.915** | 0.672 | — | 0.720 / 0.530 |

Per class on kaal_v4 test (mAP50:95): aircraft 0.81, helicopter 0.68,
tank 0.66, vehicle 0.37, soldier 0.30, artillery 0.26, person 0.09,
armored_vehicle 0.05. Latency 16.9 ms median per image (RTX 4060, FP32,
batch 1, PyTorch, unoptimised); ONNX Runtime on CPU ≈ 160–300 ms.

## Robustness and defense

A noise/pattern patch inside every ground-truth box costs 0.026 mAP50:95 on
kaal_v4 test and nothing on the unseen ground set (0.119 → 0.121). The model is not
adversarially trained; runtime protection comes from the model-free streams:
per-box FFT (27.8% of patched boxes flagged at 5.2% clean false flags,
threshold 0.08) and the temporal cue (attack demo: flagged 4 frames after the
patch appeared on 76/80 patched frames, 0 false alarms on 110 clean frames).

## Limitations

* **Fine-grained ground classes:** APC/IFV (0.05 on kaal_v4 test), artillery
  and engineering vehicles on unseen photos (≈ 0.03 / 0.00) remain weak —
  the model finds the vehicle (any-class 0.915) more reliably than it names
  it. Some tanks are labelled armored_vehicle (unseen-tank AP 0.29 → 0.24 vs v3).
* **Small objects:** VisDrone people from drone altitude (person 0.09).
* **Label noise:** DVIDS labels are VLM-assisted (≈ 1 in 10–20 boxes
  questionable); Open Images includes toy/model tanks.
* **Licences:** trained on non-commercial (VisDrone, Simuletic) and
  unknown-licence (legacy `mil_*`, military aircraft) data — **not cleared for
  commercial use**; the weights are released under CC BY-NC 4.0. Retrain
  from the licence-clean registry entries first for any commercial use.
* Evaluated on still images plus one synthetic video clip; real flight video
  (motion blur, compression, moving camera) is not yet benchmarked, and the
  temporal cue assumes a static camera.

## Ethical considerations

* People are `person` unless a source explicitly marks military personnel;
  Open Images tank photos with people are excluded rather than guessed
  (ADR-005). Mislabelling civilians as combatants is the error the data policy
  is built to avoid; operators must treat `soldier` as a low-confidence visual
  cue.
* Outputs are advisory; the dashboard shows flags, never actions.

## Versions

| Version | Data | Status | Key numbers |
|---|---|---|---|
| `rfdetr-small-v4` | kaal_v4, warm start from v3 | **production** | kaal_v4 test 0.405; unseen any-class AP50 0.915 |
| `rfdetr-small-v5` | kaal_v4, warm start from v4, cosine tail (8 epochs) | rejected (`MODEL/RESULTS.md` §8) | kaal_v4 test 0.412 but unseen ground test 0.107 vs 0.119 (APC 0.210 → 0.150) |
| `rfdetr-small-v3` | kaal_v3 | superseded | kaal_v3 test 0.552 (old split); kaal_v4 test 0.313; unseen 0.720 |
| `yolo26s-v3` | kaal_v3 | baseline, optional backend | kaal_v3 test 0.517; unseen 0.015 |
