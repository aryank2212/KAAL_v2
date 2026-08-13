"""
test_parse_yolo_labels.py — YOLO label file parser unit tests.

Covers empty files, malformed lines, normalized coordinate conversion, and
missing-file handling.
"""

from MODEL.adversarial_patch import parse_yolo_labels


class TestEmptyAndMissing:
    def test_missing_file_returns_empty(self, tmp_path):
        cls, boxes = parse_yolo_labels(str(tmp_path / "nope.txt"), 100, 100)
        assert cls == [] and boxes == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        cls, boxes = parse_yolo_labels(str(f), 100, 100)
        assert cls == [] and boxes == []


class TestMalformed:
    def test_short_lines_skipped(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("0 0.5 0.5\n1 0.5\n")
        cls, boxes = parse_yolo_labels(str(f), 100, 100)
        assert cls == [] and boxes == []

    def test_non_numeric_lines_skipped(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("abc def\n0 0.5 0.5 0.1 notanumber\n")
        cls, boxes = parse_yolo_labels(str(f), 100, 100)
        assert cls == [] and boxes == []

    def test_extra_tokens_ignored(self, tmp_path):
        f = tmp_path / "extra.txt"
        f.write_text("1 0.5 0.5 0.1 0.1 0.99 0.99\n")
        cls, boxes = parse_yolo_labels(str(f), 100, 100)
        assert cls == [1] and boxes == [(50, 50, 10, 10)]


class TestNormalizedConversion:
    def test_normalized_to_pixels(self, tmp_path):
        f = tmp_path / "labels.txt"
        f.write_text("0 0.25 0.5 0.1 0.2\n")
        cls, boxes = parse_yolo_labels(str(f), 200, 100)
        assert cls == [0]
        assert boxes == [(50, 50, 20, 20)]

    def test_full_image_box(self, tmp_path):
        f = tmp_path / "full.txt"
        f.write_text("2 0.5 0.5 1.0 1.0\n")
        cls, boxes = parse_yolo_labels(str(f), 320, 240)
        assert cls == [2]
        assert boxes == [(160, 120, 320, 240)]

    def test_multiple_objects(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("0 0.5 0.5 0.1 0.1\n1 0.1 0.9 0.2 0.05\n")
        cls, boxes = parse_yolo_labels(str(f), 1000, 1000)
        assert cls == [0, 1]
        assert boxes == [(500, 500, 100, 100), (100, 900, 200, 50)]
