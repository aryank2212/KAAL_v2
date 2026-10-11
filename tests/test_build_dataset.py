"""
test_build_dataset.py — Label remapping, VisDrone parsing and split hygiene.
"""

import json

import cv2
import pytest
import numpy as np

from MODEL.build_dataset import (
    CLS,
    LEGACY_MAP,
    Sample,
    dvids_samples,
    legacy_samples,
    legacy_source,
    merge_near_duplicate_groups,
    openimages_tank_samples,
    parse_caps,
    parse_visdrone_annotation,
    parse_yolo_file,
    pixel_boxes_to_yolo,
    rarity_sample,
    resolve_leakage,
    stratified_group_split,
    write_dataset,
)


def test_legacy_class0_drops_whole_image(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("41 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n")
    assert parse_yolo_file(lbl, LEGACY_MAP) is None


def test_legacy_mapping_and_clipping(tmp_path):
    lbl = tmp_path / "b.txt"
    lbl.write_text("45 0.5 0.5 0.2 0.2\n47 0.95 0.5 0.2 0.2\nbad line\n")
    boxes = parse_yolo_file(lbl, LEGACY_MAP)
    assert [b[0] for b in boxes] == [CLS["helicopter"], CLS["soldier"]]
    _, cx, _, w, _ = boxes[1]
    assert cx + w / 2 <= 1.0 + 1e-9  # clipped to the image


def test_visdrone_people_are_person_never_soldier(tmp_path):
    ann = tmp_path / "v.txt"
    ann.write_text(
        "10,10,20,40,1,1,0,0\n"  # pedestrian
        "50,50,30,20,1,4,0,0\n"  # car
        "0,0,100,100,0,0,0,0\n"  # ignored region
        "5,5,10,10,1,11,0,0\n"  # others
    )
    boxes = parse_visdrone_annotation(ann)
    assert [b[0] for b in boxes] == [CLS["person"], CLS["vehicle"]]
    assert CLS["soldier"] not in [b[0] for b in boxes]


def test_pixel_boxes_normalized():
    ((cls, cx, cy, w, h),) = pixel_boxes_to_yolo([(2, 10.0, 20.0, 30.0, 40.0)], h=100, w=200)
    assert cls == 2
    assert (cx, cy, w, h) == pytest.approx((0.125, 0.4, 0.15, 0.4))


def _s(split, group, classes=(0,)):
    return Sample("src", split, None, group, lambda h, w: [], list(classes))


def test_resolve_leakage_keeps_group_in_one_split():
    kept, removed = resolve_leakage([_s("train", "g1"), _s("test", "g1"), _s("val", "g2")])
    assert removed == 1
    assert {(s.split, s.group) for s in kept} == {("test", "g1"), ("val", "g2")}


def test_rarity_sample_prefers_rare_classes():
    pool = [_s("train", f"c{i}", (0,)) for i in range(200)] + [_s("train", "rare", (5,))]
    picks = [any(s.group == "rare" for s in rarity_sample(pool, 20, seed)) for seed in range(20)]
    assert sum(picks) >= 18


def test_parse_caps():
    assert parse_caps("legacy=10:2:3,visdrone=-1:-1:-1") == {
        "legacy": (10, 2, 3),
        "visdrone": (-1, -1, -1),
    }


def test_write_dataset_resizes_and_writes_yaml(tmp_path):
    img_path = tmp_path / "big.jpg"
    cv2.imwrite(str(img_path), np.full((800, 1600, 3), 127, dtype=np.uint8))
    sample = Sample(
        "visdrone",
        "train",
        img_path,
        "g",
        lambda h, w: pixel_boxes_to_yolo([(7, 100, 100, 200, 100)], h, w),
        [7],
    )
    out = tmp_path / "ds"
    stats = write_dataset([sample], out, max_side=400, min_box_px=2, workers=1)

    written = cv2.imread(str(out / "train" / "images" / "visdrone_big.jpg"))
    assert written.shape[:2] == (200, 400)
    label = (out / "train" / "labels" / "visdrone_big.txt").read_text().split()
    assert label[0] == "7"
    assert stats["images"] == {"train/visdrone": 1}
    assert "names" in (out / "data.yaml").read_text()
    json.dumps(stats)  # manifest-serializable


def test_legacy_source_groups_all_copies_of_one_photo():
    base = "air_0003f5_jpg"
    assert legacy_source(f"{base}.rf.2b6a03") == "air_0003f5"
    assert legacy_source(f"{base}.rf.73270a_noisy") == "air_0003f5"
    assert legacy_source(f"{base}.rf.2b6a03_patch") == "air_0003f5"
    assert legacy_source("mil_image_2723_occlude") == "mil_image_2723"
    assert legacy_source("mil_image_2723") == "mil_image_2723"


def test_stratified_split_puts_rare_classes_in_val_and_test():
    # Mimics the legacy set: the rare class exists only in "train".
    pool = [_s("train", f"common{i}", (0,)) for i in range(300)]
    pool += [_s("train", f"rare{i}", (0, 6)) for i in range(30)]
    pool += [_s("train", "rare0", (6,))]  # second copy of one rare photo
    out = stratified_group_split(pool, (0.8, 0.1, 0.1), seed=0)

    rare = [s for s in out if 6 in s.classes]
    assert {s.split for s in rare} == {"train", "val", "test"}
    assert len({s.group for s in rare if s.split == "test"}) == 3  # 10% of 30 photos
    homes = {}
    for s in out:
        assert homes.setdefault(s.group, s.split) == s.split  # groups never straddle


def _legacy_tree(root, split, stems, cls=41):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"")
        (root / split / "labels" / f"{stem}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n")


def test_legacy_resplit_keeps_augmented_copies_in_train_only(tmp_path):
    stems = []
    for i in range(40):
        stems += [f"mil_{i}", f"mil_{i}_noisy", f"mil_{i}_occlude", f"mil_{i}_patch"]
    _legacy_tree(tmp_path, "train", stems)
    samples = legacy_samples(tmp_path, patch_frac=1.0, seed=0, resplit=(0.8, 0.1, 0.1))

    assert not any(s.image.stem.endswith("_noisy") for s in samples)
    evals = [s for s in samples if s.split != "train"]
    assert len(evals) == 8  # 10% + 10% of 40 photos, one image each
    assert not any(s.augmented for s in evals)
    eval_groups = {s.group for s in evals}
    assert not any(s.group in eval_groups for s in samples if s.split == "train")


def test_legacy_original_split_is_kept_on_request(tmp_path):
    _legacy_tree(tmp_path, "train", [f"t{i}" for i in range(10)])
    _legacy_tree(tmp_path, "test", ["x0", "x0_occlude"], cls=1)
    samples = legacy_samples(tmp_path, patch_frac=0.0, seed=0)
    assert sorted((s.split, s.image.stem) for s in samples if s.split == "test") == [("test", "x0")]
    assert sum(s.split == "train" for s in samples) == 10


def test_near_duplicates_under_unrelated_names_share_a_group(tmp_path):
    rng = np.random.default_rng(0)
    photo = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (9, 9), 0)
    other = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (9, 9), 0)
    images = {
        "mil_image_1": photo,
        "mil_image_9": np.clip(photo.astype(int) + 12, 0, 255).astype(np.uint8),  # brighter copy
        "mil_image_5": photo[:, ::-1],  # mirrored copy
        "mil_image_2": other,
    }
    samples = []
    for stem, img in images.items():
        path = tmp_path / f"{stem}.jpg"
        cv2.imwrite(str(path), img)
        samples.append(Sample("legacy", "train", path, f"legacy:{stem}", lambda h, w: [], [2]))

    groups = {s.image.stem: s.group for s in merge_near_duplicate_groups(samples, max_dist=10)}
    assert groups["mil_image_1"] == groups["mil_image_9"] == groups["mil_image_5"]
    assert groups["mil_image_2"] != groups["mil_image_1"]


def test_legacy_unlabeled_images_dropped_unless_requested(tmp_path):
    _legacy_tree(tmp_path, "train", ["mil_1", "mil_2"])
    (tmp_path / "train" / "labels" / "mil_2.txt").write_text("")  # unannotated frame
    kept = legacy_samples(tmp_path, patch_frac=0.0, seed=0)
    assert [s.image.stem for s in kept] == ["mil_1"]
    both = legacy_samples(tmp_path, patch_frac=0.0, seed=0, keep_unlabeled=True)
    assert sorted(s.image.stem for s in both) == ["mil_1", "mil_2"]


def test_dvids_mapping_and_event_blocks(tmp_path):
    (tmp_path / "data.yaml").write_text(
        "names:\n  0: soldier\n  1: tank\n  2: apc\n  3: artillery\n  4: mlrs\n"
        "  5: military_truck\n  6: helicopter\n  7: aircraft\n  8: warship\n"
        "  9: missile_launcher\n  10: car\n  11: truck\n"
    )
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    for stem, label in (
        ("dvids_image_100", "0 0.5 0.5 0.1 0.2\n4 0.3 0.3 0.1 0.1\n8 0.7 0.7 0.2 0.2\n"),
        ("dvids_image_101", "2 0.5 0.5 0.2 0.2\n"),
        ("dvids_image_102", ""),  # labelled, no objects
    ):
        (tmp_path / "train" / "images" / f"{stem}.jpg").write_bytes(b"")
        (tmp_path / "train" / "labels" / f"{stem}.txt").write_text(label)
    (tmp_path / "train" / "images" / "dvids_image_103.jpg").write_bytes(b"")  # no label file

    by_stem = {s.image.stem: s for s in dvids_samples(tmp_path)}
    assert set(by_stem) == {"dvids_image_100", "dvids_image_101", "dvids_image_102"}
    assert by_stem["dvids_image_100"].classes == [
        CLS["soldier"],
        CLS["artillery"],
    ]  # warship dropped
    assert by_stem["dvids_image_101"].classes == [CLS["armored_vehicle"]]
    assert len({s.group for s in by_stem.values()}) == 1  # ids 100-102 are one upload block
    assert len({s.split for s in by_stem.values()}) == 1


def test_openimages_tank_filters_group_of_and_people(tmp_path):
    rows = {
        "a": [["Tank", 0.1, 0.5, 0.2, 0.6, 0], ["Car", 0.6, 0.9, 0.6, 0.9, 0]],
        "b": [["Tank", 0.1, 0.5, 0.2, 0.6, 1]],  # group-of box
        "c": [["Tank", 0.1, 0.5, 0.2, 0.6, 0], ["Person", 0.6, 0.7, 0.1, 0.9, 0]],
    }
    import json as _json

    (tmp_path / "train_boxes.json").write_text(_json.dumps(rows))
    (tmp_path / "train").mkdir()
    for image_id in rows:
        (tmp_path / "train" / f"{image_id}.jpg").write_bytes(b"")

    samples = openimages_tank_samples(tmp_path)
    assert [s.image.stem for s in samples] == ["a"]
    assert samples[0].classes == [CLS["tank"], CLS["vehicle"]]
    (_, cx, cy, w, h), _ = samples[0].boxes(1, 1)
    assert (cx, cy, w, h) == pytest.approx((0.3, 0.4, 0.4, 0.4))
    with_people = openimages_tank_samples(tmp_path, drop_person_images=False)
    assert sorted(s.image.stem for s in with_people) == ["a", "c"]
    assert with_people[1].classes == [CLS["tank"], CLS["person"]]  # never soldier


def _photo(seed):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8), (7, 7), 0)


def _v4_sources(tmp_path):
    """One photo shared by two sources, landing in test (uav_tank) and train (dvids)."""
    from MODEL import build_dataset as bd

    uav_stem = next(f"tank_{i}" for i in range(500) if bd._stable_split(f"tank_{i}") == "test")
    block = next(
        k for k in range(500) if bd._stable_split(f"dvids:{k}", (0.8, 0.1, 0.1)) == "train"
    )
    shared, other = _photo(1), _photo(2)

    uav = tmp_path / "uav"
    (uav / "images").mkdir(parents=True)
    (uav / "labels").mkdir()
    (uav / "data.yaml").write_text("names: {0: tank}\n")
    for stem, img in ((uav_stem, shared), ("tank_other", other)):
        cv2.imwrite(str(uav / "images" / f"{stem}.jpg"), img)
        (uav / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    dvids = tmp_path / "dvids"
    (dvids / "train" / "images").mkdir(parents=True)
    (dvids / "train" / "labels").mkdir(parents=True)
    (dvids / "data.yaml").write_text("names: {0: soldier, 1: tank}\n")
    dvids_stem = f"dvids_image_{block * 20}"
    cv2.imwrite(str(dvids / "train" / "images" / f"{dvids_stem}.jpg"), shared)
    (dvids / "train" / "labels" / f"{dvids_stem}.txt").write_text("1 0.5 0.5 0.3 0.3\n")
    return uav, dvids, uav_stem, dvids_stem


def _run_build(monkeypatch, tmp_path, uav, dvids, *extra):
    import sys

    from MODEL import build_dataset as bd

    out = tmp_path / "out"
    argv = ["build_dataset", "--out", str(out), "--uav-tank", str(uav), "--dvids", str(dvids),
            "--caps", "uav_tank=-1:-1:-1,dvids=-1:-1:-1", "--workers", "1", *extra]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    bd.main()
    names = {p.name: p.parent.parent.name for p in out.glob("*/images/*.jpg")}
    return names, json.loads((out / "manifest.json").read_text())


def test_v4_cross_source_duplicate_lives_in_one_split(tmp_path, monkeypatch):
    uav, dvids, uav_stem, dvids_stem = _v4_sources(tmp_path)
    names, manifest = _run_build(monkeypatch, tmp_path, uav, dvids)
    assert names[f"uav_tank_{uav_stem}.jpg"] == "test"
    assert f"dvids_{dvids_stem}.jpg" not in names  # its train copy was removed
    assert manifest["cross_split_duplicates_removed"] == 1


def test_v4_cross_source_dedup_can_be_disabled(tmp_path, monkeypatch):
    uav, dvids, uav_stem, dvids_stem = _v4_sources(tmp_path)
    names, manifest = _run_build(monkeypatch, tmp_path, uav, dvids, "--cross-dup-dist", "-1")
    assert names[f"uav_tank_{uav_stem}.jpg"] == "test"
    assert names[f"dvids_{dvids_stem}.jpg"] == "train"  # the leak kaal_v3 had
    assert manifest["cross_split_duplicates_removed"] == 0
