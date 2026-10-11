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
    parse_caps,
    parse_visdrone_annotation,
    parse_yolo_file,
    pixel_boxes_to_yolo,
    rarity_sample,
    resolve_leakage,
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
