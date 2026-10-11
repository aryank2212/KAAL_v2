# KAAL — Datasets

Human-readable view of [`MODEL/datasets_registry.yaml`](../MODEL/datasets_registry.yaml),
the machine-readable record that `tests/test_datasets_registry.py` keeps in
sync with the builders. **Update the registry in the same commit as any data
change.** Everything under `MODEL/Datasets/` is git-ignored; this page says
how to recreate it.

---

## 1. Taxonomy (9 classes, shared by every source and model)

| id | class | covers | never |
|---|---|---|---|
| 0 | `aircraft` | fixed-wing military/civil aircraft, tilt-rotors, large UAVs | — |
| 1 | `helicopter` | all rotorcraft | — |
| 2 | `tank` | main battle tanks | APCs/IFVs |
| 3 | `armored_vehicle` | APCs, IFVs | light armoured cars (ignored in tests) |
| 4 | `artillery` | towed/self-propelled guns, MLRS, missile launchers | — |
| 5 | `engineering_vehicle` | armoured engineering / recovery vehicles | — |
| 6 | `soldier` | people a source explicitly labels as military personnel | civilians, VisDrone/Open Images people |
| 7 | `vehicle` | civilian / unspecified road vehicles, military trucks | — |
| 8 | `person` | people with no military label | — |

## 2. Status at a glance

| Dataset | Role | Status | Licence | kaal_v3 | kaal_v4 | Test-only |
|---|---|---|---|---|---|---|
| Legacy KAAL (`legacy`) | training source | on disk, 3.5 GB | project data; `mil_*` upstream unknown | ✓ | ✓ (unlabeled frames dropped) | |
| VisDrone2019-DET | training source | on disk, 1.9 GB | academic, **non-commercial** | 3,000 train | all 6,471 train | |
| Military aircraft (subset) | training source | on disk, 1.3 GB | none declared | ✓ | ✓ | |
| Simuletic UAV tank | training source | on disk, 166 MB | CC BY-NC 4.0 (**non-commercial**) | ✓ | ✓ | |
| DVIDS military (llama-farm) | training source | on disk, 9.6 GB | public domain imagery, CC0 labels | | ✓ | |
| Open Images tanks | training source | on disk, ~410 MB | CC BY 4.0 / CC BY 2.0 | | ✓ | |
| Orion web images | test only | on disk | MIT annotations, web-scraped images | | | `kaal_ground_test` |
| Russian-military-annotated | test only | on disk | CC BY 4.0 | | | `kaal_ground_test` |
| TNO Search_2 | test only | on disk | CC BY 4.0 | | | `kaal_ground_test` |
| `kaal_v3_1` (experimental re-split) | built | **trashed** | — | | | |

Rejected candidates (registry `role: candidate`): AMOD-V1.0 (647 GB),
DRC CARLA camouflage (174 GB, civilian cars), MVRSD (Baidu-only),
leibnitz-lab military_vehicles (classification only), and account-gated
sources (Roboflow Universe, Kaggle, xView, DOTA).

## 3. Sources in detail

### 3.1 Legacy KAAL set — `MODEL/runs/adversarial/augmented_dataset`

Built before Phase 0 by `MODEL/merge_all.py` → `MODEL/dataset.py`:

| Prefix | Upstream | Legacy classes |
|---|---|---|
| `air_*` | Roboflow `aircraft.v1i.yolov8` | 0–39 aircraft types (0 = "A10") |
| `car_*` | `aerial_cars` | 48–51 aerial car 0–3 |
| `mil_*` | `military-artillery-and-tanks` CVAT export (HF `shreyas231219/…`, licence unknown) | 40 APC, 41 tank, 42 artillery, 43 airplane, 44 IFV, 45 helicopter, 46 engineering, 47 soldier |

`dataset.py` added `_noisy`, `_occlude` and `_patch` copies.

**Problems found (2026-10-11) and how the builder handles them**

| Problem | Evidence | Handling |
|---|---|---|
| Class 0 "A10" is a catch-all mixing cars and generic aircraft | visual inspection; 1,459 val boxes | any image carrying it is dropped (dropping only the box would make real objects background) |
| No labelled ground-vehicle/soldier image outside train | `merge_all.py` split `mil_*` 90/10 by file order; 181 of 198 val images unlabeled; none in test | ground classes are evaluated on `kaal_v4` test (DVIDS / Open Images) and the independent `kaal_ground_test` |
| `mil_*` are near-identical video frames under unrelated numbers | `mil_image_2789` = `mil_image_2790` pixel-for-pixel; 1,977 originals ≈ 615 scenes | flip-aware 256-bit dHash grouping (`merge_near_duplicate_groups`) so copies never straddle splits |
| 254 `mil_*` originals have no labels | 73 train + 181 val; they are aerial strike-footage frames | dropped in kaal_v4 (`--keep-unlabeled-legacy` restores kaal_v3 behaviour) |
| `air_*` shares photos with the military-aircraft source | 124 near-duplicate test/train pairs across the two in kaal_v3 | cross-source grouping in kaal_v4 |
| Roboflow copies (`*_jpg.rf.<hash>`) and augmented variants | many names per photo | `legacy_source()` maps them to one group; augmented variants only in train, one image per photo in val/test |

### 3.2 VisDrone2019-DET — `MODEL/Datasets/visdrone`

`gh release download v0.0.0 --repo ultralytics/assets --pattern 'VisDrone2019-DET-*.zip' --dir MODEL/Datasets/visdrone` (then unzip).
Mapping: pedestrian, people → `person` (never `soldier`); bicycle, car, van,
truck, tricycle, awning-tricycle, bus, motor → `vehicle`; ignored regions and
"others" dropped. Splits: train / val / test-dev → train / val / test.

### 3.3 Military aircraft — `MODEL/Datasets/military_aircraft`

`python -m MODEL.fetch_datasets aircraft --train 3500 --val 400 --test 400`
(rarity-weighted per split from the COCO files; 429-aware downloader).
Rotorcraft types (AH64, CH47, Mi24, …) → `helicopter`, others → `aircraft`.
The upstream also repeats images across its own splits (88 near-duplicate
test/train pairs in kaal_v3) — handled by kaal_v4 cross-source grouping.

### 3.4 Simuletic UAV tank — `MODEL/Datasets/simuletic_uav_tank`

`python -m MODEL.fetch_datasets uav_tank` — 164 synthetic drone-view tanks,
split by a stable hash (70/15/15).

### 3.5 DVIDS military — `MODEL/Datasets/dvids_military`

`python -m MODEL.fetch_datasets dvids` — 1,185 high-resolution U.S. DoD
photos (public domain); labels CC0, VLM-assisted with human correction.

| DVIDS class | KAAL class |
|---|---|
| soldier | soldier |
| tank | tank |
| apc | armored_vehicle |
| artillery, mlrs, missile_launcher | artillery |
| military_truck, car, truck | vehicle |
| helicopter | helicopter |
| aircraft | aircraft |
| warship | *(box dropped — KAAL has no maritime class)* |

Single `train` folder upstream → split 80/10/10 by blocks of 20 DVIDS ids
(one upload ≈ one event), so an event does not straddle splits.

Usable after mapping: 1,035+ labelled photos (847 / 84 / 104 at the first
count) — soldier 2,734, vehicle 476, armored_vehicle 418, tank 337,
artillery 298, helicopter 42, aircraft 30 boxes. It is the only clean-licence
source of real soldier and artillery photos in the set.

Known label noise (spot-check of 16 photos): camouflage netting boxed as
`tank`; the same M109 self-propelled howitzer labelled `artillery` in one
photo and `apc` in another; a gun-barrel close-up labelled `apc`; close-up
portraits of service members (incl. a diver) labelled `soldier`. Roughly one
questionable box in 10–20 — acceptable for training, worth relabelling
(FUTURE_SCALING §4).

### 3.6 Open Images tanks — `MODEL/Datasets/openimages_tank`

`python -m MODEL.fetch_datasets openimages_tank` — every Open Images V7
image with a `Tank` box (train 1,062 / validation 50 / test 134), plus the
other leaf-class boxes in those images. The 2.26 GB box CSV is downloaded to
`_csv/` and was trashed after filtering.

* Kept classes: Tank → tank; Helicopter → helicopter; Airplane → aircraft;
  Car, Truck, Van, Bus, Motorcycle, Bicycle → vehicle.
* Dropped: 64 images with a *group-of* box; 208 images with a Person box
  (crews/soldiers next to tanks would conflict with DVIDS `soldier`;
  `--keep-openimages-person` maps them to `person` instead). 974 remain
  (853 / 32 / 89), 1,575 tank boxes.
* Known label noise (spot-check): toy and board-game tanks, a few APCs
  labelled Tank, occasional unlabelled people.

### 3.7 Independent ground-vehicle test — `MODEL/Datasets/kaal_ground_test`

Built by `python -m MODEL.build_ground_test` from the three
`jonasrenault/orion` release assets (download commands in the module
docstring); archives were trashed after extraction.

| Source | Images | Mapping |
|---|---|---|
| Orion web (AFV/APC/MEV/LAV) | 655 | AFV in a tank-model folder (T-72, Leopard, M1 …) → tank, other AFVs (Lynx, Boxer, ZBD-04, AMX-10RC) → armored_vehicle; APC → armored_vehicle; MEV → engineering_vehicle; LAV → **ignore** |
| Russian-military-annotated | 99 (100 photos × ~10 Roboflow copies, one kept) | T-64/72/80 → tank; BMP-1/2, BMD-2, BTR-70/80, MT-LB → armored_vehicle; BM-21 → artillery |
| TNO Search_2 | 44 | 1024 px crop around the target: M1, M60, T72 → tank; M3, BMP, BTR, M113 → armored_vehicle; HVS/HVT → **ignore** |

Totals: tank 263, armored_vehicle 428, engineering_vehicle 64,
artillery 19, ignore regions 199. Near-duplicates of the training set are
excluded (`--train-ref`; rebuild against every new training set).

## 4. Built datasets

| Dataset | Command | Train / val / test | Notes |
|---|---|---|---|
| `kaal_v3` | git `6581dc5`: `python -m MODEL.build_dataset` (now: `--out MODEL/Datasets/kaal_v3 --caps <V3_CAPS> --cross-dup-dist -1 --keep-unlabeled-legacy`) | 11,633 / 1,215 / 1,216 | trained RF-DETR-S v3 and YOLO26s; val/test lack 4 ground classes; 68 test images near-duplicate train |
| `kaal_v4` | `python -m MODEL.build_dataset` | see `MODEL/Datasets/kaal_v4/manifest.json` and `MODEL/RESULTS.md` §7 | adds DVIDS, Open Images tanks, all VisDrone; cross-source de-duplication; unlabeled legacy frames dropped |
| `kaal_ground_test` | `python -m MODEL.build_ground_test` | test only, 798 | never trained on |

**Known issue (kaal_v4):** the builder guarantees no photo spans two splits,
but a photo can appear twice *within* one split via two sources or two
legacy frames: 14 of 1,428 test images (1.0%) and 16 of 1,331 val images
(1.2%) are near-duplicates of another image in the same split (dHash ≤ 20).
All models are scored on the same images, so comparisons stay fair; those
photos simply weigh twice. The next dataset version should keep one image
per near-duplicate group in val/test across sources (as `legacy_samples`
already does within the legacy source).

Every built dataset writes `manifest.json` (arguments, per-split image and
instance counts, licences, duplicates removed) and a `data.yaml` usable by
both RF-DETR (`dataset_file="yolo"`) and Ultralytics.

## 5. Which model used which data

| Model | Trained on | Status |
|---|---|---|
| `rfdetr-small-v3` | kaal_v3 | see registry `models:` |
| `yolo26s-v3` | kaal_v3 | baseline (optional backend) |
| `rfdetr-small-v4` | kaal_v4 (warm start from v3) | see registry `models:` |

## 6. Licences and commercial use

Non-commercial: **VisDrone** (academic), **Simuletic UAV tank** (CC BY-NC).
Unknown: the legacy `mil_*` upstream and the military-aircraft set. Attribution
required: Open Images (CC BY), Russian-military-annotated (CC BY 4.0),
Search_2 (CC BY 4.0). Public domain / CC0: DVIDS. Web-scraped Orion images are
for local evaluation only. A commercial model must be retrained from the
registry entries whose licence allows it.

## 7. Adding a dataset (checklist)

1. Fetch it reproducibly — a `fetch_datasets` sub-command or a documented
   `gh release download` / URL.
2. Write a `<name>_samples()` adapter in `MODEL/build_dataset.py` mapping to
   the taxonomy (document every mapping and every dropped class), add it to
   `LICENSES` and the `loaders` dict, and give it caps in `V4_CAPS` (or a new
   version constant).
3. Add a registry entry (status, role, licence, size, `used_in`) — the
   registry test fails until you do.
4. Unit-test the adapter on a few synthetic files (`tests/test_build_dataset.py`).
5. Spot-check labels visually before training.
6. Rebuild, rebuild `kaal_ground_test` with `--train-ref` against the new
   train split, retrain, and accept only if `compare_detectors` improves on
   both test sets.
