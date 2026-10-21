"""CPU-only fixture tests for the read-only civilian dataset audit."""

import json
from pathlib import Path

import numpy as np
import pytest
from MODEL.audit_civilian_dataset import audit, near_duplicate_pairs, percentiles, size_bucket
from MODEL.build_civilian_auair import CIVILIAN_CLASSES


def write_image(path: Path, value: int = 40, width: int = 64, height: int = 48):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(abs(hash(path.name)) % 2**31)
    image = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def build_dataset(root: Path, broken: bool = False) -> Path:
    """A tiny YOLO dataset; with ``broken`` it contains deliberate defects."""
    data = root / "dataset"
    names = {index: name for index, name in enumerate(CIVILIAN_CLASSES)}
    (data / "train" / "images").mkdir(parents=True, exist_ok=True)
    (data / "val" / "images").mkdir(parents=True, exist_ok=True)
    (data / "test" / "images").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (data / split / "labels").mkdir(parents=True, exist_ok=True)

    good = "0 0.5 0.5 0.2 0.3\n"
    for split, sequence in (
        ("train", "20190829091111"),
        ("val", "20190905103112"),
        ("test", "20190906150731"),
    ):
        name = f"frame_{sequence}_x_0000001.jpg"
        write_image(data / split / "images" / name)
        (data / split / "labels" / f"{Path(name).stem}.txt").write_text(good, encoding="utf-8")

    if broken:
        # An exact duplicate of a train frame, placed in the test split.
        duplicate = "frame_20190829091111_x_0000001.jpg"
        source = data / "train" / "images" / duplicate
        (data / "test" / "images" / duplicate).write_bytes(source.read_bytes())
        (data / "test" / "labels" / "frame_20190829091111_x_0000001.txt").write_text(
            "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
        )

        # A frame with no label file at all.
        write_image(data / "test" / "images" / "frame_20190906150731_x_0000009.jpg")

        # A frame with an empty label file.
        write_image(data / "test" / "images" / "frame_20190906150731_x_0000010.jpg")
        (data / "test" / "labels" / "frame_20190906150731_x_0000010.txt").write_text(
            "", encoding="utf-8"
        )

        # A frame with malformed and out-of-range labels.
        write_image(data / "val" / "images" / "frame_20190905103112_x_0000011.jpg")
        (data / "val" / "labels" / "frame_20190905103112_x_0000011.txt").write_text(
            "0 0.5 0.5 0.2\n"  # too few fields
            "99 0.5 0.5 0.2 0.3\n"  # unknown class id
            "1 0.5 0.5 0 0.3\n"  # zero width
            "2 1.4 0.5 0.2 0.3\n"  # centre outside the frame
            "3 0.5 0.5 1.4 0.3\n"  # size outside the frame
            "4 0.05 0.5 0.2 0.3\n",  # box crosses the left edge
            encoding="utf-8",
        )

        # A corrupt image.
        (data / "train" / "images" / "frame_20190829091111_x_0000012.jpg").write_bytes(
            b"not a jpeg"
        )

    (data / "data.yaml").write_text(
        "path: {}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 8\nnames:\n".format(
            data
        )
        + "".join(f"  {index}: {name}\n" for index, name in names.items()),
        encoding="utf-8",
    )
    return data


def test_audit_reports_nothing_for_a_clean_dataset(tmp_path):
    report = audit(build_dataset(tmp_path), near_duplicates=False, max_distance=5, max_images=1000)
    assert report["read_only"] is True
    assert report["counts"]["images"] == 3
    assert report["counts"]["boxes"] == 3
    assert report["images"]["corrupt_count"] == 0
    assert report["images"]["missing_label_file_count"] == 0
    assert report["labels"]["rejected"] == {}
    assert report["leakage"]["groups_in_multiple_splits"] == []
    assert report["duplicates"]["groups"] == 0


def test_audit_flags_broken_examples(tmp_path):
    report = audit(
        build_dataset(tmp_path, broken=True), near_duplicates=False, max_distance=5, max_images=1000
    )
    assert report["images"]["corrupt_count"] == 1
    assert report["images"]["missing_label_file_count"] == 1
    rejected = report["labels"]["rejected"]
    assert rejected["empty_label_file"] == 1
    assert rejected["malformed_line"] == 1
    assert rejected["unknown_class_id"] == 1
    assert rejected["non_positive_size"] == 1
    assert rejected["centre_out_of_range"] == 1
    assert rejected["size_out_of_range"] == 1
    assert rejected["box_crosses_top_left"] == 1
    # The exact duplicate is found and it crosses a split boundary.
    assert report["duplicates"]["groups"] == 1
    assert len(report["duplicates"]["cross_split_groups"]) == 1


def test_audit_detects_group_leakage(tmp_path):
    """A group that appears in two splits is reported."""
    data = build_dataset(tmp_path)
    # Copy a train frame into the test split under the same sequence name.
    source = data / "train" / "images" / "frame_20190829091111_x_0000001.jpg"
    (data / "test" / "images" / "frame_20190829091111_x_0000002.jpg").write_bytes(
        source.read_bytes()
    )
    (data / "test" / "labels" / "frame_20190829091111_x_0000002.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
    )
    report = audit(data, near_duplicates=False, max_distance=5, max_images=1000)
    assert "20190829091111" in report["leakage"]["groups_in_multiple_splits"]


def test_audit_reports_the_same_image_name_in_two_splits(tmp_path):
    """The same frame name in two splits is leakage, and must be listed."""
    data = build_dataset(tmp_path)
    source = data / "train" / "images" / "frame_20190829091111_x_0000001.jpg"
    (data / "test" / "images" / "frame_20190829091111_x_0000001.jpg").write_bytes(
        source.read_bytes()
    )
    (data / "test" / "labels" / "frame_20190829091111_x_0000001.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
    )
    report = audit(data, near_duplicates=False, max_distance=5, max_images=1000)
    assert "frame_20190829091111_x_0000001.jpg" in report["leakage"]["images_in_multiple_splits"]
    assert len(report["duplicates"]["cross_split_groups"]) == 1


def test_near_duplicates_across_splits_are_reported(tmp_path):
    """A near-duplicate pair that straddles a split must be flagged."""
    data = build_dataset(tmp_path)
    source = data / "train" / "images" / "frame_20190829091111_x_0000001.jpg"
    (data / "test" / "images" / "frame_20190906150731_x_0000002.jpg").write_bytes(
        source.read_bytes()
    )
    (data / "test" / "labels" / "frame_20190906150731_x_0000002.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
    )
    report = audit(data, near_duplicates=True, max_distance=5, max_images=1000)
    assert report["near_duplicates"]["pair_count"] >= 1
    assert report["near_duplicates"]["cross_split_pairs"], (
        "a cross-split near-duplicate must be flagged"
    )
    assert report["near_duplicates"]["within_split_pairs"] == []


def test_pair_counts_are_totals_not_the_capped_listing(tmp_path):
    """The reported counts must not be the truncated listing length."""
    data = build_dataset(tmp_path)
    source = data / "train" / "images" / "frame_20190829091111_x_0000001.jpg"
    for index in range(4):
        name = f"frame_20190829091111_x_000100{index}.jpg"
        (data / "train" / "images" / name).write_bytes(source.read_bytes())
        (data / "train" / "labels" / f"frame_20190829091111_x_000100{index}.txt").write_text(
            "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
        )
    report = audit(data, near_duplicates=True, max_distance=5, max_images=1000)
    near = report["near_duplicates"]
    assert near["within_split_pair_count"] == near["within_split_pair_count"]
    assert near["within_split_pair_count"] >= 6
    assert near["cross_split_pair_count"] == len(near["cross_split_pairs"])
    assert "capped" in near["listing_note"]
    assert near["images_in_cross_split_pairs"] == 0
    assert near["groups_in_cross_split_pairs"] == []


def test_audit_is_read_only(tmp_path):
    """The audit must not modify the dataset it inspects."""
    data = build_dataset(tmp_path, broken=True)
    before = {path: path.stat().st_mtime_ns for path in sorted(data.rglob("*")) if path.is_file()}
    audit(data, near_duplicates=False, max_distance=5, max_images=1000)
    after = {path: path.stat().st_mtime_ns for path in sorted(data.rglob("*")) if path.is_file()}
    assert before == after


def test_audit_output_is_deterministic(tmp_path):
    data = build_dataset(tmp_path, broken=True)
    first = json.dumps(
        audit(data, near_duplicates=False, max_distance=5, max_images=1000),
        indent=2,
        sort_keys=True,
    )
    second = json.dumps(
        audit(data, near_duplicates=False, max_distance=5, max_images=1000),
        indent=2,
        sort_keys=True,
    )
    assert first == second


def test_class_balance_and_object_size_are_reported(tmp_path):
    report = audit(build_dataset(tmp_path), near_duplicates=False, max_distance=5, max_images=1000)
    balance = report["class_balance"]["train"]
    assert balance["instances"]["human"] == 1
    assert balance["rarest_class_count"] == 1
    assert balance["ratio_to_rarest"]["human"] == 1.0
    sizes = report["object_size"]["human"]
    assert sizes["instances"] == 3
    assert sizes["normalised_area"]["p50"] == pytest.approx(0.2 * 0.3)
    assert sum(sizes["buckets"].values()) == 3


def test_size_buckets():
    assert size_bucket(0.0005) == "tiny"
    assert size_bucket(0.005) == "small"
    assert size_bucket(0.02) == "medium"
    assert size_bucket(0.5) == "large"


def test_percentiles_of_an_empty_sample_are_null():
    values = percentiles([])
    assert values["p50"] is None
    assert values["min"] is None


def test_near_duplicate_pairs_finds_identical_hashes():
    """Identical hashes pair up; a different hash is not a duplicate at distance 0."""
    hashes = np.zeros((4, 8), dtype=np.uint8)
    hashes[3] = 255  # every bit differs from the others
    pairs = list(near_duplicate_pairs(hashes, max_distance=0))
    assert pairs == [(0, 1), (0, 2), (1, 2)]
    # With a permissive distance the different hash joins in.
    assert len(list(near_duplicate_pairs(hashes, max_distance=64))) == 6


def test_near_duplicate_limits_are_explicit(tmp_path):
    report = audit(build_dataset(tmp_path), near_duplicates=False, max_distance=5, max_images=1000)
    limits = report["near_duplicates"]["limits"]
    assert limits["analysis_performed"] is False
    assert limits["max_images_hashed"] == 1000
    assert "not run" in limits["note"]


def test_near_duplicate_stage_records_its_limits(tmp_path):
    report = audit(build_dataset(tmp_path), near_duplicates=True, max_distance=5, max_images=1000)
    limits = report["near_duplicates"]["limits"]
    assert limits["analysis_performed"] is True
    assert limits["images_hashed"] == 3
    assert limits["truncated"] is False
    assert report["near_duplicates"]["max_hamming_distance"] == 5
