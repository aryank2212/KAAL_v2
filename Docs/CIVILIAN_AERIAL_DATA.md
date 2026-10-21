# KAAL — civilian aerial data (AU-AIR)

Preflight record for an **isolated, civilian-only** experiment built on the
AU-AIR dataset. This page is acquisition provenance and scope guardrails. It is
not a dataset registry entry: `MODEL/datasets_registry.yaml` and the
nine-class production taxonomy are **not** touched by this experiment.

Every statement below was checked against upstream documentation on
**2026-10-21**. Anything that upstream does not state is marked
*unresolved* rather than guessed.

---

## 1. Scope

| | |
|---|---|
| Purpose | A separate civilian aerial-traffic dataset and baseline, for general-purpose scene understanding and robustness research. |
| Not in scope | Military-target detection, target prioritisation, tracking of people, autonomous engagement. |
| Human review | Consequential decisions stay human-reviewed. The civilian baseline is never promoted to production. |
| Isolation | Separate importer, separate eight-class taxonomy, separate output directory. The production detector, its weights and `MODEL/build_dataset.py` are untouched. |

---

## 2. Upstream source

| | |
|---|---|
| Paper | Bozcan & Kayacan, *AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance*, ICRA 2020 — <https://arxiv.org/abs/2001.11737> |
| Official project page | <https://bozcani.github.io/auairdataset> — **now redirects to <https://odtum.de/auairdataset/> and returns HTTP 404.** The site is gone as of 2026-10-21. |
| Archived official page | <https://web.archive.org/web/20220125151322/https://bozcani.github.io/auairdataset> |
| Official API repository | <https://github.com/bozcani/auairdataset> — **HTTP 404; the repository no longer exists.** |
| Live mirrors of that API | <https://github.com/sunw71/auairdataset> (MIT, 21★, last push 2022-03-03) · <https://github.com/freeridering/auair-dataset> (MIT, 5★, last push 2020-03-17) |

The mirror repositories hold the **API code only**. They are the only surviving
source for the annotation file's schema, because the API source is what reads
that schema (`auairtools/auair.py`: `dataset['annotations']`, `dataset['categories']`,
`bbox['left' / 'top' / 'width' / 'height' / 'class']`).

> **Consequence:** the download links below are still the only official source
> for the data itself, and they are hosted on Google Drive by the authors. They
> are publicly shared files — no credentials, no access control to bypass.

---

## 3. Download locations

| What | Where | Published size |
|---|---|---|
| Frames (images) | Google Drive id `1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ` — <https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ> | **2.2 GB** |
| Annotations (JSON) | Google Drive id `1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA` — <https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA> | **3.9 MB** |

Archive names are **not published** — upstream distributes loose files through
Google Drive, not a named archive. Sizes are the figures from the official
project page; the actual byte counts are recorded in the dataset manifest at
build time and are not asserted here.

Reachability check on 2026-10-21 (headers only, no file downloaded): both Drive
ids returned HTTP 200.

Fetch with `gdown` against the public ids. No account, no key, no cookie.

---

## 4. Licence — **conflicting upstream statements, unresolved**

This is the single most important item in this preflight. The upstream sources
do **not** agree, and the most restrictive reading must be assumed until the
authors answer.

| Source | Statement | Exact wording |
|---|---|---|
| Official project page (archived 2022-01-25) | "CC-BY" | "You are free to use AU-AIR dataset and all empirical data and metadata under the CC-BY licence, requesting appropriate credit if you do." |
| FMI Data Index | CC BY 4.0 | "Dataset licenced under the CC BY 4.0 licence." |
| **The annotation file's own `licenses` block** | **CC BY-NC-SA 2.0 and CC BY-NC 2.0** | `[{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2, "name": "Attribution-NonCommercial License"}]` |
| `auairdataset` GitHub `LICENSE` | MIT | Copyright (c) 2019 İlker Bozcan — **this covers the API code, not the dataset.** |

**Unresolved.** The dataset's own embedded metadata lists only **NonCommercial**
variants, while the project page advertises plain CC-BY. The annotation file is
the artefact that actually travels with the data, so its metadata is at least as
authoritative as a marketing sentence on a page that is now offline.

**Operating rules until this is clarified**

1. Treat AU-AIR as **non-commercial**. Do not publish weights trained on it, or
   any derivative dataset, under terms that permit commercial use.
2. **Do not redistribute** raw frames or raw annotations.
3. Do not assert a specific licence version in any manifest or report. Record
   the *conflict* and the sources, not a single resolved answer.
4. Cite the paper and credit the dataset as the authors request.
5. Open question, unresolved: whether ShareAlike (`by-nc-sa/2.0`) attaches to
   adapted forms. Upstream does not say which of the two embedded licence ids
   applies to the dataset as a whole.
6. Official contact, from the project page: `ilker@eng.au.dk`. Confirm before
   any publication or release.

---

## 5. Expected layout and schema

Verified against the upstream sample annotation file
(`examples/auair_subset/annotations.json`, 80 frames, from the `sunw71` mirror)
and against `auairtools/auair.py`.

### Frames

* Flat directory of JPEG files, no sub-folders.
* File name: `frame_<YYYYMMDDHHMMSS>_<x+>_<7 digits>.jpg`, e.g.
  `frame_20190829091111_x_0001973.jpg`. All 80 sample names match this pattern.
* Resolution **1920×1080** (`image_width` / `image_height` are `1920.0` / `1080.0`).
* Extracted at **5 frames per second** from 30 fps video (paper §III-B), so
  consecutive frames are 0.2 s apart.
* Eight source videos, ~2 hours total.

### Annotations

One JSON file. Top level is an object:

| Key | Content |
|---|---|
| `info` | `description`, `url`, `version`, `year`, `contributor`, `date_created` |
| `licenses` | list of `{url, id, name}` — see §4 |
| `categories` | **list of 8 strings**, indexed by the `class` field of a box |
| `annotations` | list of per-frame records |

Each `annotations[i]`:

| Field | Type | Meaning |
|---|---|---|
| `image_name` | str | JPEG file name, same directory as the frames |
| `image_width`, `image_height` | float | frame size in pixels |
| `platform` | str | UAV platform, e.g. `Parrot Bebop 2` |
| `time` | object | `year, month, day, hour, min, sec, ms` |
| `longtitude`, `latitude` | float | GPS (note upstream spelling `longtitude`) |
| `altitude` | float | millimetres |
| `linear_x`, `linear_y`, `linear_z` | float | body-frame velocity, m/s |
| `angle_phi`, `angle_theta`, `angle_psi` | float | roll / pitch / yaw, radians |
| `bbox` | list | object boxes |

Each `bbox[j]`: `top`, `left`, `height`, `width` in **pixels, top-left origin**
and `class`, an **index into `categories`**.

### Category order — authoritative

The `categories` array in the annotation file is the single source of truth:

| id | upstream name | civilian name (this repo) |
|---:|---|---|
| 0 | `Human` | `human` |
| 1 | `Car` | `car` |
| 2 | `Truck` | `truck` |
| 3 | `Van` | `van` |
| 4 | `Motorbike` | `motorbike` |
| 5 | `Bicycle` | `bicycle` |
| 6 | `Bus` | `bus` |
| 7 | `Trailer` | `trailer` |

> **Do not use the README's class list.** The API README lists
> "human, car, van, truck, bike, motorbike, bus, trailar" — a different order
> *and* a misspelling. Reading it as the label order would silently permute
> four classes. Only the JSON `categories` array is authoritative.

This eight-class civilian taxonomy is **separate** from the nine-class
production taxonomy (`aircraft, helicopter, tank, armored_vehicle, artillery,
engineering_vehicle, soldier, vehicle, person`). `human` is deliberately not
named `person`: it is a different label with a different meaning, and keeping
the names apart prevents the two taxonomies being conflated. No AU-AIR label is
mapped into the production taxonomy.

---

## 6. Split policy — why the upstream split is not reused

Upstream splits **frames** at 60 / 10 / 30 (paper §IV-A) and describes the
dataset as 30,000 train-validation + 2,823 test frames (paper §I). With frames
extracted 5-per-second from eight continuous videos, adjacent frames are
0.2 s apart and depict the same vehicles at almost the same place. A random
frame split therefore leaks the test set into training.

This experiment splits **by source video**, so no video contributes frames to
more than one split.

**Sequence identity is recoverable**, which is what makes this possible:

* The file name carries a `YYYYMMDDHHMMSS` prefix identifying the source video.
* It is cross-checkable against the per-frame `time` field: in the upstream
  80-frame sample, all 80 names matched the `time` field and there were exactly
  **8 distinct prefixes** — matching the eight source videos.
* The `_x_` / `_xx_` marker is *not* a sequence discriminator: four of the eight
  prefixes contain both variants.

The importer must fail closed if this key cannot be recovered, rather than fall
back to a random frame split.

---

## 7. Destination and Git-ignore status

| | |
|---|---|
| Raw frames + annotations | `MODEL/Datasets/auair/` |
| Converted dataset + manifest | `MODEL/Datasets/auair_civilian/` |
| Ignored by | `.gitignore` line 101 — `MODEL/Datasets/` |

Verified on 2026-10-21:

```
$ git check-ignore -v MODEL/Datasets/auair/annotations.json
.gitignore:101:MODEL/Datasets/    MODEL/Datasets/auair/annotations.json
```

No raw data, no checkpoint and no credential is ever committed.

---

## 8. Published counts — and one inconsistency

| Quantity | Value | Source |
|---|---|---|
| Source videos | 8 | project page, paper §III-B |
| Extracted, labelled frames | **32,823** | project page, paper §I, §III-B |
| — the API README says | **32,283** | `auairdataset` README — inconsistent with the two sources above; treated as a transposition typo |
| Annotated object instances | 132,034 | paper §I |
| Images | 1920×1080 | paper §III-A |
| Frame extraction rate | 5 fps from 30 fps | paper §III-B |
| Flight altitude range | 5–30 m | paper §III-B |
| Camera angle range | 45°–90° | paper §III-B |
| Annotation method | 3 crowd workers per frame, boxes merged at IoU > 0.75, then manually corrected | paper §III-C |

Upstream's own AP table (paper §IV-C) shows heavy class imbalance, with
`motorbike` and `bicycle` near zero for both baselines. Expect those two
classes to be the weak point of any baseline here; do not read a low number for
them as a bug in the pipeline.

---

## 9. Explicitly unresolved

* Which licence actually applies — see §4. The conflict between "CC-BY" on the
  project page and the annotation file's `CC BY-NC-SA 2.0` / `CC BY-NC 2.0`
  metadata is unresolved, and no upstream source was found that picks one.
* Whether ShareAlike attaches to adapted forms.
* Archive names and checksums — upstream publishes none; they are recorded at
  build time instead.
* Whether the released annotation file matches the sample schema exactly, in
  particular the `categories` order. Verified against the upstream sample only.
* Whether all 32,823 frames are present in the released archive, and whether
  the released annotation file covers every frame.

---

## 10. Next step

`MODEL/build_civilian_auair.py` — the isolated importer (Task 3), then
`MODEL/audit_civilian_dataset.py` — the read-only audit (Task 4).
