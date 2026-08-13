"""
test_end_to_end.py — GPU end-to-end integration test (opt-in).

Trains the model for 1 epoch on a tiny dataset, runs the full evaluation
pass, and asserts the report file exists with valid numbers.

This test requires ultralytics + torch + a model checkpoint, so it only runs
when KAAL_RUN_GPU_TESTS=1 (see .github/workflows/integration.yml). It is
skipped in the fast unit-test CI job.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KAAL_RUN_GPU_TESTS") != "1",
    reason="GPU integration test — set KAAL_RUN_GPU_TESTS=1 to run",
)


def _make_dataset(root):
    """Build a minimal YOLO dataset: 3 train images, 2 val images."""
    import cv2
    import numpy as np

    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        img_dir = os.path.join(root, split, "images")
        lab_dir = os.path.join(root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for i in range(3 if split == "train" else 2):
            img = cv2.GaussianBlur(rng.randint(0, 255, (160, 160, 3), dtype=np.uint8), (21, 21), 0)
            cv2.rectangle(img, (60, 60), (100, 100), (200, 200, 200), -1)
            cv2.imwrite(os.path.join(img_dir, f"{split}{i}.png"), img)
            with open(os.path.join(lab_dir, f"{split}{i}.txt"), "w") as f:
                f.write("0 0.5 0.5 0.25 0.25\n")

    data_yaml = os.path.join(root, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"train: {root}/train/images\nval: {root}/val/images\nnc: 1\nnames: ['person']\n")
    return data_yaml


def test_train_one_epoch_and_evaluate(tmp_path):
    pytest.importorskip("ultralytics")
    pytest.importorskip("torch")

    data_yaml = _make_dataset(str(tmp_path / "data"))
    report_path = os.path.join(str(tmp_path), "runs", "adversarial", "evaluation_report.txt")

    cmd = [
        sys.executable,
        "-m",
        "MODEL.pipeline",
        "--data",
        data_yaml,
        "--train",
        "--epochs",
        "1",
        "--batch",
        "2",
        "--model",
        "yolov8n.pt",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    assert os.path.exists(report_path), "evaluation report not written"
    with open(report_path) as f:
        content = f.read()

    for marker in ("mAP@0.5", "Clean", "Adversarial", "Adversarial + Defense"):
        assert marker in content, f"report missing {marker}"
