# Model Card — KAAL production detector

| | |
|---|---|
| **Model** | RF-DETR-Small fine-tuned for KAAL (`rfdetr-small-v3`) |
| **Status** | production (registry `models:`), until `rfdetr-small-v4` passes its gates |
| **Weights** | `MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth` (127 MB; not committed) |
| **Framework** | `rfdetr` 1.11 (Roboflow, Apache-2.0), PyTorch; DINOv2-windowed-small backbone, 512 px input, NMS-free |
| **Classes (9)** | aircraft, helicopter, tank, armored_vehicle, artillery, engineering_vehicle, soldier, vehicle, person |
| **Owner** | KAAL project (aryank2212/KAAL_v2) |

## Intended use

Perception for drone imagery inside the KAAL dual-stream defense:
detections feed the FFT and temporal tamper checks, and a human operator sees
detections plus "predictions unreliable" flags on the dashboard.

**Out of scope:** autonomous engagement or weapon targeting; identifying
individuals; deciding whether a person is a combatant (the `soldier` class is
a visual label, not a status determination); any use where a missed or false
detection would cause harm without human review.

## Training

| Item | Value |
|---|---|
| Data | `kaal_v3` (11,633 train images): legacy KAAL set, VisDrone2019-DET (3,000), military aircraft subset (3,500), Simuletic UAV tanks (133) — see `docs/DATASETS.md` |
| Init | COCO-pretrained RF-DETR-Small |
| Schedule | 12 epochs, batch 4 × grad-accum 4, lr 1e-4, EMA, early stopping (patience 5) |
| Hardware / time | RTX 4060 Laptop 8 GB, 1.99 h |
| Best validation | mAP50:95 0.604 (epoch 9, RF-DETR evaluator) |

## Evaluation (`MODEL/RESULTS.md` §6)

| Test set | mAP50 | mAP50:95 | attacked mAP50:95 | Notes |
|---|---|---|---|---|
| kaal_v3 test (1,216) | 0.708 | 0.552 | 0.516 | aircraft 0.81, tank 0.80, helicopter 0.73, vehicle 0.35, person 0.07 |
| kaal_v3 test, de-duplicated (1,148) | 0.708 | 0.552 | 0.525 | 68 near-duplicates of train removed |
| kaal_ground_test (798 unseen photos) | 0.132 | 0.091 | 0.075 | any-class mAP50 **0.722** — finds vehicles, confuses tank/APC/artillery/engineering |

Latency: 17.1 ms median per image (RTX 4060, FP32, batch 1, unoptimised).

## Robustness and defense

Adversarial patches inside every ground-truth box cost 0.036 mAP50:95
(YOLO26s: 0.045). The model is not adversarially trained; robustness at
runtime comes from the independent streams: per-box FFT (28.8% of patched
boxes flagged at 5.4% clean false flags, threshold 0.08) and the temporal cue
(live demo: flagged 0.2 s after a patch appeared, no false alarms on clean
frames).

## Limitations

* Ground-vehicle sub-class confusion (see above); `soldier`,
  `engineering_vehicle` and `artillery` have few distinct training scenes.
* Small objects from drone altitude (VisDrone person 0.07 mAP50:95).
* Training data includes non-commercial (VisDrone, Simuletic) and
  unknown-licence sources — **not cleared for commercial use**.
* Evaluated on still images; video-domain shifts (motion blur, compression)
  are only covered by the demo clip.

## Ethical considerations

* People are labelled `person` unless a source explicitly marks military
  personnel; Open Images photos with people near tanks are excluded rather
  than guessed (ADR-005).
* Outputs are advisory; the dashboard presents flags, not actions.

## Versions

| Version | Data | Status |
|---|---|---|
| `rfdetr-small-v3` | kaal_v3 | production (this card) |
| `yolo26s-v3` | kaal_v3 | baseline, optional backend |
| `rfdetr-small-v4` | kaal_v4, warm start from v3 | in training — card updated when it passes |
