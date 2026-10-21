"""CPU-only fixture tests for the isolated AU-AIR civilian importer."""

import json
from pathlib import Path

import pytest
from MODEL.build_civilian_auair import (
    CIVILIAN_CLASSES,
    UPSTREAM_CATEGORIES,
    CivilianDatasetError,
    assign_splits,
    build_manifest,
    civilian_class_ids,
    frame_width,
    parse_boxes,
    read_frames,
    sequence_from_name,
    sequence_from_time,
    write_dataset,
)


def make_annotation(
    image_name="frame_20190829091111_x_0000001.jpg",
    sequence="20190829091111",
    boxes=None,
    width=1920.0,
    height=1080.0,
    time_field=None,
):
    """A minimal AU-AIR annotation record, shaped like the released file."""
    if boxes is None:
        boxes = [{"top": 100, "left": 200, "height": 50, "width": 60, "class": 1}]
    return {
        "image_name": image_name,
        "image_width:": width,
        "image_height": height,
        "platform": "Parrot Bebop 2",
        "time": time_field
        or {"year": 2019, "month": 8, "day": 29, "hour": 9, "min": 11, "sec": 11, "ms": 0.0},
        "bbox": boxes,
    }


def make_document(records, categories=UPSTREAM_CATEGORIES):
    return {
        "info": {"description": "fixture"},
        "licenses": [
            {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2, "name": "nc"}
        ],
        "categories": list(categories),
        "annotations": records,
    }


def write_images(directory: Path, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"\xff\xd8\xff\xd9 fixture jpeg")


# ── category mapping ─────────────────────────────────────────────────────


def test_civilian_class_ids_map_upstream_names_in_any_order():
    forward = civilian_class_ids(list(UPSTREAM_CATEGORIES))
    assert forward == {index: index for index in range(8)}
    # The mapping is by name, so a permuted upstream array cannot remap labels.
    permuted = list(reversed(UPSTREAM_CATEGORIES))
    assert civilian_class_ids(permuted) == {
        index: CIVILIAN_CLASSES.index(name.lower()) for index, name in enumerate(permuted)
    }


def test_civilian_class_ids_reject_an_unknown_category():
    with pytest.raises(CivilianDatasetError, match="Unknown upstream category"):
        civilian_class_ids(["Human", "Car", "Tracked"])


def test_civilian_taxonomy_is_eight_classes_and_disjoint_from_production():
    assert len(CIVILIAN_CLASSES) == 8
    assert set(CIVILIAN_CLASSES).isdisjoint(
        {
            "aircraft",
            "helicopter",
            "tank",
            "armored_vehicle",
            "artillery",
            "engineering_vehicle",
            "soldier",
            "vehicle",
            "person",
        }
    )


def test_frame_width_accepts_both_upstream_spellings():
    assert frame_width({"image_width": 1920.0}) == 1920.0
    assert frame_width({"image_width:": 1920.0}) == 1920.0
    assert frame_width({"image_width": "wide"}) is None


# ── sequence identity ────────────────────────────────────────────────────


def test_sequence_is_read_from_the_name_and_the_time_field():
    assert sequence_from_name("frame_20190829091111_x_0001973.jpg") == "20190829091111"
    assert sequence_from_name("frame_20190829091111_xx_0001973.jpg") == "20190829091111"
    assert sequence_from_name("not_a_frame.jpg") is None
    stamp = {"year": 2019, "month": 8, "day": 29, "hour": 9, "min": 11, "sec": 11}
    assert sequence_from_time(stamp) == "20190829091111"
    assert sequence_from_time({"year": 2019}) is None


# ── box validation ───────────────────────────────────────────────────────


def test_valid_box_becomes_a_normalised_yolo_box():
    rejected: dict = {}
    from collections import Counter

    rejected = Counter()
    boxes = parse_boxes(make_annotation(), {0: 0, 1: 1}, rejected, [])
    assert len(boxes) == 1
    class_id, cx, cy, w, h = boxes[0]
    assert class_id == 1
    assert cx == pytest.approx((200 + 30) / 1920)
    assert cy == pytest.approx((100 + 25) / 1080)
    assert w == pytest.approx(60 / 1920)
    assert h == pytest.approx(50 / 1080)
    assert not rejected


@pytest.mark.parametrize(
    ("box", "kind"),
    [
        ({"top": 10, "left": 10, "height": 0, "width": 10, "class": 1}, "non_positive_size"),
        ({"top": 10, "left": 10, "height": -5, "width": 10, "class": 1}, "non_positive_size"),
        ({"top": -5, "left": 10, "height": 10, "width": 10, "class": 1}, "negative_origin"),
        ({"top": 10, "left": 1900, "height": 400, "width": 100, "class": 1}, "out_of_bounds"),
        ({"top": 10, "left": 10, "height": 10, "class": 1}, "malformed_box"),
    ],
)
def test_malformed_boxes_are_dropped_and_reported(box, kind):
    from collections import Counter

    rejected = Counter()
    examples: list = []
    boxes = parse_boxes(make_annotation(boxes=[box]), {0: 0, 1: 1}, rejected, examples)
    assert boxes == []
    assert rejected[kind] == 1
    assert examples and examples[0]["kind"] == kind


def test_unknown_class_index_is_dropped_not_remapped():
    from collections import Counter

    rejected = Counter()
    examples: list = []
    boxes = parse_boxes(
        make_annotation(boxes=[{"top": 1, "left": 1, "height": 2, "width": 2, "class": 99}]),
        {0: 0, 1: 1},
        rejected,
        examples,
    )
    assert boxes == []
    assert rejected["unknown_class_index"] == 1
    assert examples[0]["kind"] == "unknown_class_index"


# ── frame reading ────────────────────────────────────────────────────────


def test_read_frames_drops_frames_whose_image_is_missing(tmp_path):
    from collections import Counter

    images = tmp_path / "images"
    write_images(images, ["frame_20190829091111_x_0000001.jpg"])
    rejected = Counter()
    examples: list = []
    document = make_document(
        [
            make_annotation("frame_20190829091111_x_0000001.jpg"),
            make_annotation("frame_20190829091111_x_0000002.jpg"),
        ]
    )
    frames = read_frames(document, images, rejected, examples)
    assert [frame.image_name for frame in frames] == ["frame_20190829091111_x_0000001.jpg"]
    assert rejected["missing_image_file"] == 1


def test_read_frames_drops_a_frame_whose_sequence_key_disagrees(tmp_path):
    from collections import Counter

    images = tmp_path / "images"
    name = "frame_20190829091111_x_0000001.jpg"
    write_images(images, [name])
    rejected = Counter()
    examples: list = []
    document = make_document(
        [
            make_annotation(
                name,
                time_field={"year": 2020, "month": 1, "day": 1, "hour": 0, "min": 0, "sec": 0},
            )
        ]
    )
    frames = read_frames(document, images, rejected, examples)
    assert frames == []
    assert rejected["sequence_key_mismatch"] == 1


# ── grouped splitting ────────────────────────────────────────────────────


def build_sequences():
    """Eight sequences with the real upstream class mix."""
    mix = {
        "20190829091111": [1, 1, 1, 0, 7, 2],
        "20190905091750": [1, 1, 3, 2, 6],
        "20190905103112": [1, 1, 1, 4, 5],
        "20190905111947": [1, 6, 5],
        "20190905112522": [1, 3, 2, 6],
        "20190905142119": [1, 1, 5, 4],
        "20190905143505": [1, 3, 6],
        "20190906150731": [1, 0, 2, 3, 4, 5, 6, 7],
    }
    frames = []
    for sequence, classes in mix.items():
        for index, class_id in enumerate(classes):
            frames.append(
                type(
                    "F",
                    (),
                    {
                        "image_name": f"frame_{sequence}_x_{index:07d}.jpg",
                        "sequence": sequence,
                        "boxes": [(class_id, 0.5, 0.5, 0.1, 0.1)],
                        "split": "",
                    },
                )()
            )
    return frames, mix


def test_assign_splits_is_deterministic():
    frames, _ = build_sequences()
    first = assign_splits(frames, (0.7, 0.15, 0.15), 42)
    second = assign_splits(frames, (0.7, 0.15, 0.15), 42)
    assert first == second


def test_assign_splits_keeps_every_sequence_whole():
    frames, mix = build_sequences()
    assignment = assign_splits(frames, (0.7, 0.15, 0.15), 42)
    assert set(assignment) == set(mix)
    assert set(assignment.values()) <= {"train", "val", "test"}
    # A sequence that lands in test must not also appear in train or val.
    assert len(set(assignment.values())) >= 3


def test_assign_splits_fails_closed_with_a_single_sequence():
    frames, _ = build_sequences()
    one = [frame for frame in frames if frame.sequence == "20190906150731"]
    with pytest.raises(CivilianDatasetError, match="Refusing to fall back"):
        assign_splits(one, (0.7, 0.15, 0.15), 42)


def test_assign_splits_prefers_coverage_of_the_rare_classes():
    frames, _ = build_sequences()
    assignment = assign_splits(frames, (0.7, 0.15, 0.15), 42)
    held_out = {key for key, split in assignment.items() if split in {"val", "test"}}
    assert held_out, "val and test must not be empty"
    classes = {
        class_id for frame in frames if frame.sequence in held_out for class_id, *_ in frame.boxes
    }
    assert 4 in classes and 5 in classes


def test_assign_splits_seed_decides_between_equally_good_sequences():
    """Two sequences with the same class mix must be separable by seed alone."""

    frames = []
    for sequence, classes in (
        ("20190829091111", [1, 1, 5]),
        ("20190905091750", [1, 1, 5]),
        ("20190905103112", [1, 1]),
    ):
        for index, class_id in enumerate(classes):
            frames.append(
                type(
                    "F",
                    (),
                    {
                        "image_name": f"frame_{sequence}_x_{index:07d}.jpg",
                        "sequence": sequence,
                        "boxes": [(class_id, 0.5, 0.5, 0.1, 0.1)],
                        "split": "",
                    },
                )()
            )
    assignments = {seed: assign_splits(frames, (0.7, 0.15, 0.15), seed) for seed in (1, 2, 3, 4, 5)}
    # The two equivalent sequences swap between test and val across seeds.
    swapped = {
        seed
        for seed, assignment in assignments.items()
        if assignment["20190829091111"] != assignments[1]["20190829091111"]
    }
    assert swapped, "the seed must be able to swap equally good sequences"
    for assignment in assignments.values():
        assert assignment["20190905103112"] == "train"
        assert {assignment["20190829091111"], assignment["20190905091750"]} == {"test", "val"}


# ── dataset writing and manifest ─────────────────────────────────────────


def test_write_dataset_and_manifest(tmp_path):
    from collections import Counter

    images = tmp_path / "images"
    out = tmp_path / "auair_civilian"
    sequences = {
        "20190829091111": (1, 7),
        "20190905091750": (3, 6),
        "20190905103112": (2, 5),
        "20190906150731": (0, 4),
    }
    records = []
    names = []
    for sequence, (first_class, second_class) in sequences.items():
        stamp = {
            "20190829091111": {
                "year": 2019,
                "month": 8,
                "day": 29,
                "hour": 9,
                "min": 11,
                "sec": 11,
            },
            "20190905091750": {"year": 2019, "month": 9, "day": 5, "hour": 9, "min": 17, "sec": 50},
            "20190905103112": {
                "year": 2019,
                "month": 9,
                "day": 5,
                "hour": 10,
                "min": 31,
                "sec": 12,
            },
            "20190906150731": {"year": 2019, "month": 9, "day": 6, "hour": 15, "min": 7, "sec": 31},
        }[sequence]
        for index, class_id in enumerate((first_class, second_class)):
            name = f"frame_{sequence}_x_{index:07d}.jpg"
            names.append(name)
            records.append(
                make_annotation(
                    name,
                    boxes=[{"top": 0, "left": 0, "height": 1080, "width": 1920, "class": class_id}],
                    time_field=stamp,
                )
            )
    write_images(images, names)
    document = make_document(records)
    rejected = Counter()
    examples: list = []
    frames = read_frames(document, images, rejected, examples)
    assert len(frames) == len(names)
    assignment = assign_splits(frames, (0.7, 0.15, 0.15), 42)
    for frame in frames:
        frame.split = assignment[frame.sequence]

    stats = write_dataset(frames, images, out)
    assert sum(stats["frames"].values()) == len(names)
    data_yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 8" in data_yaml
    for name in CIVILIAN_CLASSES:
        assert name in data_yaml

    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(document), encoding="utf-8")
    manifest = build_manifest(
        frames=frames,
        annotations_path=annotations_path,
        images_dir=images,
        out=out,
        seed=42,
        fractions=(0.7, 0.15, 0.15),
        split_assignment=assignment,
        rejected=rejected,
        examples=examples,
        inputs=[
            {
                "label": "annotations.json",
                "path": str(annotations_path),
                "sha256": "0" * 64,
                "bytes": 1,
            }
        ],
        categories=document["categories"],
    )
    assert manifest["source"]["name"] == "AU-AIR"
    assert manifest["source"]["license"]["status"] == "conflicting - unresolved"
    assert manifest["seed"] == 42
    assert manifest["counts"]["frames_total"] == len(names)
    assert manifest["counts"]["instances_per_class"]["car"] == 1
    assert manifest["counts"]["instances_per_class"]["human"] == 1
    assert manifest["inputs"][0]["sha256"] == "0" * 64
    assert manifest["split_policy"]["grouping"] == "source_video"
    assert manifest["rejected"] == {}
    assert {entry["key"] for entry in manifest["sequences"]} == set(sequences)
    # Every sequence is recorded with exactly one split.
    for entry in manifest["sequences"]:
        assert entry["split"] in {"train", "val", "test"}
    # The manifest records the upstream category order for traceability.
    assert manifest["upstream_categories"] == list(UPSTREAM_CATEGORIES)


def test_manifest_reports_rejected_annotations(tmp_path):
    from collections import Counter

    images = tmp_path / "images"
    out = tmp_path / "auair_civilian"
    sequences = {
        "20190829091111": {"year": 2019, "month": 8, "day": 29, "hour": 9, "min": 11, "sec": 11},
        "20190906150731": {"year": 2019, "month": 9, "day": 6, "hour": 15, "min": 7, "sec": 31},
        "20190905103112": {"year": 2019, "month": 9, "day": 5, "hour": 10, "min": 31, "sec": 12},
        "20190905091750": {"year": 2019, "month": 9, "day": 5, "hour": 9, "min": 17, "sec": 50},
    }
    records = []
    names = []
    for sequence, stamp in sequences.items():
        name = f"frame_{sequence}_x_0000001.jpg"
        names.append(name)
        records.append(
            make_annotation(
                name,
                boxes=[
                    {"top": 0, "left": 0, "height": 1080, "width": 1920, "class": 1},
                    {"top": 0, "left": 0, "height": 0, "width": 10, "class": 1},
                ],
                time_field=stamp,
            )
        )
    write_images(images, names)
    document = make_document(records)
    rejected = Counter()
    examples: list = []
    frames = read_frames(document, images, rejected, examples)
    assignment = assign_splits(frames, (0.7, 0.15, 0.15), 42)
    for frame in frames:
        frame.split = assignment[frame.sequence]
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text("{}", encoding="utf-8")
    manifest = build_manifest(
        frames=frames,
        annotations_path=annotations_path,
        images_dir=images,
        out=out,
        seed=42,
        fractions=(0.7, 0.15, 0.15),
        split_assignment=assignment,
        rejected=rejected,
        examples=examples,
        inputs=[],
        categories=document["categories"],
    )
    assert manifest["rejected"]["non_positive_size"] == len(sequences)
    assert manifest["counts"]["instances_total"] == len(sequences)
    assert manifest["rejected_examples"][0]["kind"] == "non_positive_size"


def test_source_block_keeps_the_licence_conflict_unresolved():
    from MODEL.build_civilian_auair import SOURCE

    assert SOURCE["license"]["status"] == "conflicting - unresolved"
    assert len(SOURCE["license"]["statements"]) >= 3
    assert any("NonCommercial" in item["text"] for item in SOURCE["license"]["statements"])
