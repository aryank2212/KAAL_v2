"""
test_compare_detectors.py — Evaluation math used for the RF-DETR vs YOLO report.
"""

import numpy as np
import pytest

from MODEL.compare_detectors import _auroc, coco_map, patched_copy, read_gt

pytest.importorskip("pycocotools")

CLASSES = ["aircraft", "tank"]


def _det(cls, box, score=0.9):
    x, y, w, h = box
    return {
        "bbox": [x, y, x + w, y + h],
        "confidence": score,
        "class_id": cls,
        "class_name": CLASSES[cls],
    }


def test_perfect_detections_score_one():
    gts = [[(1, [10.0, 10.0, 50.0, 40.0])], [(0, [100.0, 80.0, 60.0, 30.0])]]
    dets = [[_det(1, gts[0][0][1])], [_det(0, gts[1][0][1])]]
    out = coco_map(gts, dets, [(200, 200), (200, 200)], CLASSES)
    assert out["mAP50"] == pytest.approx(1.0)
    assert out["mAP50_95"] == pytest.approx(1.0)
    assert set(out["per_class_mAP50_95"]) == {"aircraft", "tank"}


def test_wrong_class_scores_zero():
    gts = [[(1, [10.0, 10.0, 50.0, 40.0])]]
    out = coco_map(gts, [[_det(0, gts[0][0][1])]], [(200, 200)], CLASSES)
    assert out["mAP50_95"] == pytest.approx(0.0)


def test_no_detections():
    out = coco_map([[(0, [1.0, 1.0, 5.0, 5.0])]], [[]], [(50, 50)], CLASSES)
    assert out == {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_mAP50_95": {}}


def test_auroc():
    assert _auroc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(1.0)
    assert _auroc([0.8, 0.9], [0.1, 0.2]) == pytest.approx(0.0)
    assert _auroc([0.1, 0.5], [0.3, 0.7]) == pytest.approx(0.75)


def test_read_gt_and_deterministic_patch(tmp_path):
    lbl = tmp_path / "x.txt"
    lbl.write_text("1 0.5 0.5 0.5 0.5\n")
    gt = read_gt(lbl, h=100, w=200)
    assert gt == [(1, [50.0, 25.0, 100.0, 50.0])]
    img = np.full((100, 200, 3), 120, dtype=np.uint8)
    a, b = patched_copy(img, gt, seed=4), patched_copy(img, gt, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, img)
