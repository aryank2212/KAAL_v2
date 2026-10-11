---
license: cc-by-nc-4.0
pipeline_tag: object-detection
tags:
  - object-detection
  - rf-detr
  - detr
  - onnx
  - pytorch
  - drone
  - aerial-imagery
  - uav
  - adversarial-robustness
  - military
datasets:
  - llama-farm/military-labeled-yolo
  - a2015003713/military-aircraft-detection-dataset
  - Simuletic/UAV-Aerial-View-Battle-Tank-Detection-Dataset
extra_gated_prompt: >-
  KAAL RF-DETR-Small detects military vehicles, aircraft and personnel in drone
  and ground imagery. It is released for research on adversarial-patch defence
  and perception robustness under CC BY-NC 4.0. Access is granted automatically
  once you accept the conditions below.
extra_gated_fields:
  I will use this model for research or other non-commercial purposes only (CC BY-NC 4.0): checkbox
  I will not use this model for autonomous weapon targeting or engagement: checkbox
  I will not use this model to identify, track or profile individual people: checkbox
  A human will review any decision informed by this model's outputs: checkbox
extra_gated_button_content: Accept and access
model-index:
  - name: {{MODEL_ID}}
    results:
      - task:
          type: object-detection
          name: Object Detection
        dataset:
          name: kaal_v4 test split (1,428 images)
          type: kaal_v4
          split: test
        metrics:
          - type: mAP
            name: mAP50:95
            value: 0.405
          - type: mAP@0.5
            name: mAP50
            value: 0.565
      - task:
          type: object-detection
          name: Object Detection
        dataset:
          name: kaal_ground_test (798 unseen ground-vehicle photos)
          type: kaal_ground_test
          split: test
        metrics:
          - type: mAP@0.5
            name: any-class mAP50 (vehicle found, class ignored)
            value: 0.915
          - type: mAP
            name: mAP50:95 (tank, armored_vehicle, artillery, engineering_vehicle)
            value: 0.119
---

# KAAL RF-DETR-Small (`{{MODEL_ID}}`)

The perception model of [KAAL](https://github.com/aryank2212/KAAL_v2), a
research system that defends drone object detection against adversarial
patches. It is an [RF-DETR](https://github.com/roboflow/rf-detr)-Small
detector (DINOv2 backbone, NMS-free, 31.8 M parameters, 512 × 512 input)
fine-tuned to find 9 classes in drone and ground imagery:

`aircraft`, `helicopter`, `tank`, `armored_vehicle`, `artillery`,
`engineering_vehicle`, `soldier`, `vehicle`, `person`

Inside KAAL the detector is never trusted on its own: every frame also goes
through model-free tamper checks (per-box FFT spectrum, temporal consistency),
and a human operator sees the detections together with "predictions
unreliable" flags. Read [Intended use](#intended-use) before downloading.

## Files (tag `{{TAG}}`)

| File | Use | SHA-256 |
|---|---|---|
| `{{WEIGHTS_FILE}}` | PyTorch checkpoint for `rfdetr` (GPU or CPU) | `{{WEIGHTS_SHA256}}` |
| `{{ONNX_FILE}}` | ONNX graph for ONNX Runtime / TensorRT / OpenVINO (no torch) | `{{ONNX_SHA256}}` |
| `{{ONNX_FILE}}.json` | Class names, input size and export parity report for the ONNX graph | — |

Pin the tag (`revision="{{TAG}}"`) so a later release cannot change what you
download. The checkpoint holds only model weights, class names and the
architecture config: training state (optimizer, EMA copy, loop counters) and
machine-specific paths were removed. It loads with
`torch.load(weights_only=True)`, and its predictions are identical to the
training checkpoint's (100 test images, 21,714 detections, zero difference in
any box or score). The ONNX graph matches the PyTorch model within
+0.005 mAP50:95 on 200 validation images.

## Quick start

### ONNX Runtime (no PyTorch)

```python
# pip install onnxruntime opencv-python numpy huggingface_hub
import json

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo, rev = "{{REPO_ID}}", "{{TAG}}"
onnx_path = hf_hub_download(repo, "{{ONNX_FILE}}", revision=rev)
with open(hf_hub_download(repo, "{{ONNX_FILE}}.json", revision=rev)) as f:
    names = json.load(f)["class_names"]
session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
inp = session.get_inputs()[0]
H, W = inp.shape[2], inp.shape[3]  # 512 x 512

image = cv2.imread("frame.jpg")  # BGR
h0, w0 = image.shape[:2]
x = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_LINEAR)
x = (x.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.transpose(2, 0, 1)[None].astype(np.float32)

# boxes: (1, 300, 4) normalised cx, cy, w, h; logits: (1, 300, 10)
boxes, logits = session.run(["dets", "labels"], {inp.name: x})
prob = 1.0 / (1.0 + np.exp(-logits[0]))  # 9 classes + 1 unused slot
top = np.argsort(-prob.ravel(), kind="stable")[:300]
for query, cls, score in zip(top // prob.shape[1], top % prob.shape[1], prob.ravel()[top]):
    if score < 0.25 or cls >= len(names):
        continue
    cx, cy, bw, bh = boxes[0, query] * [w0, h0, w0, h0]
    xyxy = [round(float(v), 1) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)]
    print(names[cls], round(float(score), 3), xyxy)
```

### PyTorch (`rfdetr`)

```python
# pip install "rfdetr>=1.11" huggingface_hub
from huggingface_hub import hf_hub_download
from PIL import Image
from rfdetr import RFDETR

weights = hf_hub_download("{{REPO_ID}}", "{{WEIGHTS_FILE}}", revision="{{TAG}}")
model = RFDETR.from_checkpoint(weights)
detections = model.predict(Image.open("frame.jpg").convert("RGB"), threshold=0.25)
for box, score, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
    print(model.class_names[cls], round(float(score), 3), [round(float(v), 1) for v in box])
```

### In KAAL (dashboard + dual-stream defence)

```bash
git clone https://github.com/aryank2212/KAAL_v2 && cd KAAL_v2
pip install -e ".[onnx,server,hub]"      # or ".[ml,server,hub]" for the PyTorch model on a GPU
hf auth login                           # once, after accepting the terms on this page
python -m MODEL.fetch_weights --format onnx   # pinned revision, SHA-256 checked against the registry
python Frontend_Backend/backend/server.py     # loads the production model from MODEL/datasets_registry.yaml
```

## Evaluation

All numbers are COCO-style (pycocotools, 300 detections per image), from
[`MODEL/RESULTS.md` §7](https://github.com/aryank2212/KAAL_v2/blob/main/MODEL/RESULTS.md).
"Attacked" places a noise/pattern patch inside every ground-truth box.

**`kaal_v4` test split** (1,428 images; near-duplicates of training images
removed across all sources before splitting):

| Model | mAP50 | mAP50:95 | mAP50:95 attacked | Latency |
|---|---|---|---|---|
| previous version (v3) | 0.418 | 0.313 | 0.291 | 17.0 ms |
| **this model (v4)** | **0.565** | **0.405** | **0.379** | 16.9 ms |

Latency: median per image, RTX 4060 Laptop, FP32, batch 1, PyTorch,
unoptimised. ONNX Runtime on a laptop CPU: about 160–300 ms per image.

| Class (mAP50:95) | aircraft | helicopter | tank | vehicle | soldier | artillery | person | armored_vehicle |
|---|---|---|---|---|---|---|---|---|
| v4 | 0.813 | 0.682 | 0.664 | 0.373 | 0.301 | 0.264 | 0.085 | 0.052 |

**`kaal_ground_test`**: 798 public photos of ground vehicles from sources
never used in training (Orion web images, Russian-military-annotated, TNO
Search_2):

| Model | mAP50 | mAP50:95 | any-class mAP50 | any-class mAP50:95 |
|---|---|---|---|---|
| previous version (v3) | 0.131 | 0.090 | 0.720 | 0.530 |
| **this model (v4)** | **0.172** | **0.119** | **0.915** | **0.672** |

"Any-class" asks whether the vehicle was found at all, ignoring which class
was predicted. The model finds unseen vehicles far more reliably than it
names them: per class (mAP50:95) armored_vehicle 0.210, tank 0.237,
artillery 0.029, engineering_vehicle 0.001.

## Training

| Item | Value |
|---|---|
| Initialisation | RF-DETR-Small (COCO, Apache-2.0) → KAAL v3 (12 epochs) → this model |
| Data | `kaal_v4` train, 16,683 images (sources below) |
| Schedule | 10 epochs, lr 5e-5, batch 4 × gradient accumulation 4, EMA |
| Hardware | 1 × RTX 4060 Laptop (8 GB), 2.43 h |
| Validation | mAP50:95 0.462 on `kaal_v4` val at the last epoch (still rising) |
| Software | Python 3.12, torch 2.14, rfdetr 1.11.2; ONNX export with onnx 1.23, onnxruntime 1.31 |

| Source | Train images | Licence |
|---|---|---|
| VisDrone2019-DET | 6,471 | academic, non-commercial |
| Legacy KAAL set (aircraft, aerial cars, military video frames) | 5,000 | project data; military frames' upstream licence unknown |
| Military Aircraft Detection (Kaggle mirror) | 3,263 | none declared upstream |
| DVIDS military photos (`llama-farm/military-labeled-yolo`) | 963 | images U.S. public domain; labels CC0-1.0 |
| Open Images V7, images with a Tank box | 853 | annotations CC BY 4.0 (Google); images CC BY 2.0 |
| Simuletic UAV battle tanks (synthetic sample) | 133 | CC BY-NC 4.0 |

Labelling policy: people are `person` unless a source explicitly marks
military personnel, so `soldier` comes only from sources that label it.
Open Images tank photos that contain people are excluded rather than guessed.

## Intended use

Research and evaluation of perception robustness: adversarial-patch
detection, dual-stream tamper checks, benchmarking detectors on drone and
ground imagery, with a human reviewing every output.

**Out of scope:**

- autonomous weapon targeting or engagement, or any fire-control use;
- identifying, tracking or profiling individual people;
- deciding whether a person is a combatant. `soldier` is a visual label
  learned from a few sources, not a status determination;
- any use where a missed or false detection could cause harm without human
  review;
- commercial use (see the licence below).

The access conditions at the top of this page commit every user to the
first two points, to human review, and to non-commercial use.

## Limitations

- **Fine-grained ground classes are weak.** APC/IFV scores 0.05 mAP50:95 on
  `kaal_v4` test. On unseen photos, artillery and engineering vehicles are
  near zero, and some tanks are labelled `armored_vehicle`.
- **Small objects.** People seen from drone altitude (VisDrone) score
  0.09 mAP50:95.
- **Label noise.** DVIDS labels are VLM-assisted with human correction:
  roughly 1 box in 10–20 is questionable. Open Images includes toy and model
  tanks.
- **Not adversarially trained.** A patch inside every box costs
  0.026 mAP50:95. KAAL's runtime defence flags patched inputs; it does not
  correct the detections.
- **Still images only.** Evaluated on still images plus one synthetic video
  clip. Real flight video (motion blur, compression, moving camera) is not
  yet benchmarked.

## Licence

The weights are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) because the
training data includes non-commercial (VisDrone, Simuletic) and
unknown-licence sources. **They are not cleared for commercial use.** RF-DETR
code and its COCO-pretrained weights are Apache-2.0 (Roboflow).

Attribution: Open Images Dataset V7 annotations by Google LLC (CC BY 4.0).
DVIDS imagery: the appearance of U.S. Department of Defense visual
information does not imply or constitute DoD endorsement. VisDrone2019 by
the AISKYEYE team, Tianjin University.

## Provenance

| | |
|---|---|
| Registry entry | `{{MODEL_ID}}` in [`MODEL/datasets_registry.yaml`](https://github.com/aryank2212/KAAL_v2/blob/main/MODEL/datasets_registry.yaml) |
| Training checkpoint | `checkpoint_best_total.pth`, SHA-256 `d38e4896dbcf80809d9f635f754d2be5dfa5bb7fd267281caad0a82af9c13d39` (EMA weights) |
| Release checkpoint | written from it by `python -m MODEL.publish_model strip` |
| Full model card | [`Docs/MODEL_CARD.md`](https://github.com/aryank2212/KAAL_v2/blob/main/Docs/MODEL_CARD.md) |

## Citation

If you use this model, cite RF-DETR as its
[repository](https://github.com/roboflow/rf-detr) asks, and reference this
model as:

```bibtex
@misc{kaal_rfdetr_small,
  title  = {KAAL RF-DETR-Small: a drone and ground-imagery detector for adversarial-patch defence research},
  author = {KAAL project},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/{{REPO_ID}}}}
}
```
