"""
test_make_demo_video.py — The demo clip goes clean -> patched -> clean.
"""

import cv2
import numpy as np

from MODEL.make_demo_video import largest_label_box, render


def test_largest_label_box(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img = tmp_path / "images" / "a.jpg"
    (tmp_path / "labels" / "a.txt").write_text("0 0.2 0.2 0.1 0.1\n2 0.5 0.5 0.4 0.2\n")
    assert largest_label_box(img, w=100, h=100) == (30, 40, 70, 60)


def test_render_three_acts(tmp_path):
    rng = np.random.RandomState(0)
    still = cv2.GaussianBlur(rng.randint(0, 255, (120, 160, 3), dtype=np.uint8), (15, 15), 0)
    out = tmp_path / "demo.mp4"
    n = render(still, (40, 30, 120, 90), out, fps=10, seconds=(1.0, 1.0, 1.0))
    assert n == 30

    cap = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float32))
    assert len(frames) == 30
    box = (slice(30, 90), slice(40, 120))  # the target region
    clean_diff = np.abs(frames[2][box] - frames[5][box]).mean()
    attack_diff = np.abs(frames[5][box] - frames[15][box]).mean()
    assert attack_diff > 3 * clean_diff  # the patch is present in act 2 only
    assert np.abs(frames[5][box] - frames[25][box]).mean() < attack_diff / 3
