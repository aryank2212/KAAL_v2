"""
test_benchmark.py — Unit tests for MODEL/benchmark.py (Phase 3).
"""

import csv

import numpy as np
import cv2
import pytest

import MODEL.benchmark as bench
from MODEL.benchmark import (
    apply_defense,
    apply_attack,
    generate_universal_perturbation,
    voc_map,
    BenchmarkRunner,
)


class TestDefenses:
    def test_identity(self, smooth_image):
        out = apply_defense("none", smooth_image)
        np.testing.assert_array_equal(out, smooth_image)

    def test_jpeg_roundtrip_shape_and_dtype(self, smooth_image):
        out = apply_defense("jpeg", smooth_image, jpeg_quality=30)
        assert out.shape == smooth_image.shape
        assert out.dtype == np.uint8

    def test_blur_no_crash_and_smooths(self, smooth_image):
        out = apply_defense("blur", smooth_image, blur_kernel=7, blur_sigma=2.0)
        assert out.shape == smooth_image.shape
        assert out.dtype == np.uint8
        assert float(out.std()) <= float(smooth_image.std())

    def test_bitdepth_quantizes_to_levels(self, smooth_image):
        out = apply_defense("bitdepth", smooth_image, bit_depth=1)
        assert out.dtype == np.uint8
        assert set(np.unique(out)).issubset({0, 85, 170, 255})  # 2 levels per channel

    def test_featuresqueeze_runs(self, smooth_image):
        out = apply_defense("featuresqueeze", smooth_image, bit_depth=3, blur_kernel=5)
        assert out.shape == smooth_image.shape
        assert out.dtype == np.uint8

    def test_unknown_defense_raises(self, smooth_image):
        with pytest.raises(ValueError):
            apply_defense("bogus", smooth_image)

    def test_policy_defenses_pass_through(self, smooth_image):
        for name in ("fft", "fft_jpeg"):
            np.testing.assert_array_equal(apply_defense(name, smooth_image), smooth_image)

    def test_jpeg_kills_high_frequency_signature(self):
        rng = np.random.RandomState(0)
        noise = rng.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        from MODEL.fft_scorer import compute_fft_anomaly_score

        before = compute_fft_anomaly_score(noise)
        after = compute_fft_anomaly_score(apply_defense("jpeg", noise, jpeg_quality=10))
        assert after < before


class TestAttacks:
    def test_universal_perturbation_deterministic(self, smooth_image):
        a = generate_universal_perturbation([smooth_image], seed=7)
        b = generate_universal_perturbation([smooth_image], seed=7)
        np.testing.assert_array_equal(a, b)
        assert a.shape == smooth_image.shape
        assert np.abs(a).max() <= 40.0 + 1e-3

    def test_universal_low_freq_is_low_frequency(self, smooth_image):
        from MODEL.fft_scorer import compute_fft_anomaly_score

        pert = generate_universal_perturbation([smooth_image], seed=1, low_freq=True)
        perturbed = np.clip(smooth_image.astype(np.float32) + pert, 0, 255).astype(np.uint8)
        sharp = generate_universal_perturbation([smooth_image], seed=1, low_freq=False)
        sharped = np.clip(smooth_image.astype(np.float32) + sharp, 0, 255).astype(np.uint8)
        assert compute_fft_anomaly_score(perturbed) < compute_fft_anomaly_score(sharped)

    def test_apply_attack_patch_needs_bbox(self, smooth_image):
        rng = np.random.RandomState(0)
        out = apply_attack("patch", smooth_image, rng, bboxes=[(80, 80, 40, 40)])
        assert out.shape == smooth_image.shape
        assert not np.array_equal(out, smooth_image)

    def test_apply_attack_universal_requires_perturbation(self, smooth_image):
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError):
            apply_attack("universal", smooth_image, rng)

    def test_apply_attack_lowfreq_changes_image(self, smooth_image):
        rng = np.random.RandomState(0)
        out = apply_attack("lowfreq", smooth_image, rng)
        assert out.shape == smooth_image.shape
        assert not np.array_equal(out, smooth_image)

    def test_unknown_attack_raises(self, smooth_image):
        with pytest.raises(ValueError):
            apply_attack("bogus", smooth_image, np.random.RandomState(0))


class TestVOCMap:
    def test_perfect_detection(self):
        gt = [[(0, [10, 10, 40, 40])]]
        preds = [[(0, 0.9, [10, 10, 40, 40])]]
        assert voc_map(gt, preds) == pytest.approx(1.0)

    def test_no_detections(self):
        gt = [[(0, [10, 10, 40, 40])]]
        preds = [[]]
        assert voc_map(gt, preds) == 0.0

    def test_misplaced_box_scores_zero(self):
        gt = [[(0, [10, 10, 40, 40])]]
        preds = [[(0, 0.9, [100, 100, 130, 130])]]
        assert voc_map(gt, preds) == 0.0

    def test_class_ignored_when_no_gt(self):
        gt = [[(0, [10, 10, 40, 40])]]
        preds = [[(1, 0.9, [10, 10, 40, 40])]]
        assert voc_map(gt, preds) == 0.0


class _StubDetector:
    """Returns a fixed detection in the benchmark detector contract."""

    def __init__(self, box=(70, 70, 90, 90), conf=0.9, cls=0):
        self._box = list(box)
        self._conf = conf
        self._cls = cls

    def predict(self, image):
        return [{"bbox": self._box, "confidence": self._conf, "class_id": self._cls}]


def _make_dataset(root, n=6, seed=0):
    """Clean BGR images with one centered GT box each."""
    rng = np.random.RandomState(seed)
    img_dir = root / "images"
    lab_dir = root / "labels"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    for i in range(n):
        img = cv2.GaussianBlur(rng.randint(0, 255, (160, 160, 3), dtype=np.uint8), (21, 21), 0)
        cv2.rectangle(img, (70, 70), (90, 90), (200, 200, 200), -1)
        cv2.imwrite(str(img_dir / f"img{i:02d}.png"), img)
        cx, cy, w, h = 80, 80, 20, 20
        with open(lab_dir / f"img{i:02d}.txt", "w") as f:
            f.write(f"0 {cx / 160:.6f} {cy / 160:.6f} {w / 160:.6f} {h / 160:.6f}\n")
    return img_dir, lab_dir


@pytest.fixture
def dataset(tmp_path):
    img_dir, lab_dir = _make_dataset(tmp_path)
    return str(img_dir), str(lab_dir)


class TestBenchmarkRunner:
    def test_detectability_matrix_model_free(self, dataset, tmp_path):
        img_dir, lab_dir = dataset
        runner = BenchmarkRunner(img_dir, lab_dir, output_dir=str(tmp_path / "out"), n_images=6)
        summary = runner.run()
        det = summary["detectability"]
        assert det["n_images"] == 6
        rows = det["rows"]
        # one row per (defense, attack)
        assert len(rows) == len(bench.DEFAULT_DEFENSES) * len(bench.DEFAULT_ATTACKS)
        for row in rows:
            assert 0.0 <= row["fft_flag_rate"] <= 1.0
            assert 0.0 <= row["clean_fpr"] <= 1.0
        # outputs written
        assert (tmp_path / "out" / "benchmark_detectability.csv").exists()
        assert (tmp_path / "out" / "benchmark_detectability.json").exists()

    def test_map_matrix_with_detector(self, dataset, tmp_path):
        img_dir, lab_dir = dataset
        runner = BenchmarkRunner(
            img_dir,
            lab_dir,
            output_dir=str(tmp_path / "out"),
            detector=_StubDetector(box=[70, 70, 90, 90]),
            n_images=6,
        )
        summary = runner.run()
        rows = summary["map_matrix"]["rows"]
        assert len(rows) == len(bench.DEFAULT_DEFENSES) * len(bench.DEFAULT_ATTACKS)
        for row in rows:
            assert 0.0 <= row["mAP50"] <= 1.0
            assert row["latency_ms"] >= 0.0
        md = (tmp_path / "out" / "benchmark_map.md").read_text()
        assert "| Defense |" in md
        assert "| FFT (ours)" in md or "| fft |" in md

    def test_detectability_csv_has_header(self, dataset, tmp_path):
        img_dir, lab_dir = dataset
        runner = BenchmarkRunner(img_dir, lab_dir, output_dir=str(tmp_path / "out"), n_images=6)
        runner.run()
        with open(tmp_path / "out" / "benchmark_detectability.csv") as f:
            header = next(csv.reader(f))
        assert header == ["defense", "attack", "fft_flag_rate", "clean_fpr"]

    def test_empty_images_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        runner = BenchmarkRunner(str(empty), str(empty), n_images=6)
        with pytest.raises(ValueError):
            runner.run()
