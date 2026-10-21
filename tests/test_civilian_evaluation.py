"""CPU-only fixture tests for the civilian baseline evaluator's metrics.

The detector itself is not exercised here - these cover the parts that decide
what gets reported: the image-quality transforms, the COCO metric plumbing,
the precision/recall matching and the size buckets.
"""

import numpy as np
import pytest
from MODEL.evaluate_civilian_baseline import (
    CONDITIONS,
    coco_evaluate,
    dim,
    gaussian_blur,
    jpeg_compress,
    low_light,
    percentiles,
    read_split,
    size_bucket,
    size_bucket_ap,
)

CLASS_NAMES = ["human", "car", "truck", "van", "motorbike", "bicycle", "bus", "trailer"]


def test_transforms_change_the_image_in_the_documented_way():
    image = np.full((32, 32, 3), 200, dtype=np.uint8)

    compressed = jpeg_compress(image, 40)
    assert compressed.shape == image.shape
    assert not np.array_equal(compressed, image)

    blurred = gaussian_blur(image, 9, 0.0)
    assert blurred.shape == image.shape

    darker = dim(image, 0.6)
    assert darker.max() <= image.max()
    assert darker.mean() < image.mean()

    low = low_light(image, 2.2)
    assert low.mean() < image.mean()

    # A uniform image is unchanged by blur, so the transform is a no-op there.
    assert np.array_equal(blurred, image)


def test_conditions_are_documented():
    assert set(CONDITIONS) == {
        "clean",
        "jpeg_q40",
        "blur_k9",
        "dim_x0.6",
        "low_light_gamma2.2",
    }
    for name, spec in CONDITIONS.items():
        assert spec["description"], name


def test_size_bucket_thresholds():
    assert size_bucket(0.0005) == "tiny"
    assert size_bucket(0.005) == "small"
    assert size_bucket(0.02) == "medium"
    assert size_bucket(0.5) == "large"


def test_percentiles_of_empty_input_are_null():
    assert percentiles([], (50, 95)) == {"p50": None, "p95": None}


def build_case():
    """One image, 100x100 px, one ground-truth car and one matching detection."""
    truth = [[(1, 0.5, 0.5, 0.2, 0.2)]]  # per image: (class_id, cx, cy, w, h)
    detections = [
        [
            {
                "bbox": [40.0, 40.0, 60.0, 60.0],
                "confidence": 0.9,
                "class_id": 1,
                "class_name": "car",
            }
        ]
    ]
    sizes = [(100, 100)]
    return truth, detections, sizes


def test_coco_evaluate_reports_perfect_detection():
    truth, detections, sizes = build_case()
    metrics = coco_evaluate(truth, detections, sizes, CLASS_NAMES, operating_conf=0.25)
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP50_95"] == pytest.approx(1.0)
    assert metrics["per_class_mAP50_95"]["car"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0


def test_coco_evaluate_counts_a_miss_and_a_false_positive():
    truth, detections, sizes = build_case()
    detections[0].append(
        {
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "confidence": 0.8,
            "class_id": 6,
            "class_name": "bus",
        }
    )
    metrics = coco_evaluate(truth, detections, sizes, CLASS_NAMES, operating_conf=0.25)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["missed_ground_truth"] == 0
    assert metrics["false_positives_per_image"] == pytest.approx(1.0)


def test_coco_evaluate_counts_a_missed_ground_truth_box():
    truth, detections, sizes = build_case()
    metrics = coco_evaluate(truth, [], sizes, CLASS_NAMES, operating_conf=0.25)
    assert metrics["true_positives"] == 0
    assert metrics["missed_ground_truth"] == 1
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["precision"] is None


def test_coco_evaluate_handles_a_class_with_no_ground_truth():
    truth, detections, sizes = build_case()
    metrics = coco_evaluate(truth, detections, sizes, CLASS_NAMES, operating_conf=0.25)
    # Classes with no ground truth and no detection are absent, not zero.
    assert "human" not in metrics["per_class_mAP50_95"]
    assert "bus" not in metrics["per_class_mAP50_95"]


def test_size_buckets_split_by_normalised_area():
    truth = [
        [
            (1, 0.5, 0.5, 0.02, 0.02),  # tiny:  area 0.0004
            (1, 0.25, 0.25, 0.05, 0.05),  # small: area 0.0025
            (1, 0.75, 0.75, 0.15, 0.15),  # medium: area 0.0225
        ]
    ]
    detections = [
        [
            {
                "bbox": [49.0, 49.0, 51.0, 51.0],
                "confidence": 0.9,
                "class_id": 1,
                "class_name": "car",
            }
        ]
    ]
    sizes = [(100, 100)]
    buckets = size_bucket_ap(truth, detections, sizes, CLASS_NAMES, operating_conf=0.25)
    assert buckets["tiny"]["ground_truth"] == 1
    assert buckets["small"]["ground_truth"] == 1
    assert buckets["medium"]["ground_truth"] == 1
    assert buckets["tiny"]["true_positives"] == 1
    assert buckets["small"]["missed"] == 1
    assert buckets["medium"]["missed"] == 1
    assert buckets["tiny"]["recall"] == pytest.approx(1.0)
    assert buckets["small"]["recall"] == pytest.approx(0.0)


def test_read_split_reads_labels_and_names(tmp_path):
    import yaml

    data = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (data / split / "images").mkdir(parents=True, exist_ok=True)
        (data / split / "labels").mkdir(parents=True, exist_ok=True)
    image = data / "test" / "images" / "frame_20190906150731_x_0000001.jpg"
    image.write_bytes(b"not really a jpeg")
    (data / "test" / "labels" / "frame_20190906150731_x_0000001.txt").write_text(
        "1 0.5 0.5 0.2 0.3\n", encoding="utf-8"
    )
    (data / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(data),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 8,
                "names": dict(enumerate(CLASS_NAMES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    images, class_names, truth = read_split(data, "test")
    assert [path.name for path in images] == ["frame_20190906150731_x_0000001.jpg"]
    assert class_names == CLASS_NAMES
    assert truth["frame_20190906150731_x_0000001.jpg"] == [(1, 0.5, 0.5, 0.2, 0.3)]
