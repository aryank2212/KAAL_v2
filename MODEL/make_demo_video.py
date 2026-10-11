"""
make_demo_video.py — Build a reproducible attack-demo clip for the live server.

Takes one still image (e.g. a drone-view frame from the kaal_v3 test split)
and renders a static-camera clip in three acts:

    clean (sensor noise only) -> adversarial patch on the target -> clean

so the dashboard can be exercised end to end without a webcam:

    python -m MODEL.make_demo_video --image MODEL/Datasets/kaal_v3/test/images/uav_tank_x.jpg \\
        --box 400 300 520 380 --out demo.mp4
    KAAL_CAMERA_SOURCE=demo.mp4 python Frontend_Backend/backend/server.py

Without ``--box`` the box is read from the image's YOLO label (largest object).
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .adversarial_patch import AdversarialPatchInjector
from .config import PatchType


def largest_label_box(image_path: Path, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label.exists():
        return None
    best = None
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        if best is None or bw * bh > best[2] * best[3]:
            best = (cx, cy, bw, bh)
    if best is None:
        return None
    cx, cy, bw, bh = best
    return (
        int((cx - bw / 2) * w),
        int((cy - bh / 2) * h),
        int((cx + bw / 2) * w),
        int((cy + bh / 2) * h),
    )


def render(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    out: Path,
    fps: int = 20,
    seconds: Tuple[float, float, float] = (4.0, 4.0, 3.0),
    patch: PatchType = PatchType.NOISE,
    seed: int = 0,
) -> int:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    x1, y1, x2, y2 = box
    injector = AdversarialPatchInjector(patch_type=patch, scale_range=(0.35, 0.5))
    attacked, _ = injector.apply_patch(
        image, [((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1)], prob=1.0
    )

    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    frames = 0
    for base, secs in ((image, seconds[0]), (attacked, seconds[1]), (image, seconds[2])):
        for _ in range(int(secs * fps)):
            noisy = base.astype(np.float32) + rng.normal(0, 1.5, base.shape)  # sensor noise
            writer.write(np.clip(noisy, 0, 255).astype(np.uint8))
            frames += 1
    writer.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a clean -> attack -> clean demo clip.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--out", default="demo.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--patch", choices=[t.value for t in (PatchType.NOISE, PatchType.PATTERN)], default="noise"
    )
    parser.add_argument("--width", type=int, default=960, help="Resize the still to this width")
    args = parser.parse_args()

    path = Path(args.image)
    image = cv2.imread(str(path))
    if image is None:
        parser.error(f"cannot read {path}")
    h, w = image.shape[:2]
    scale = args.width / w
    image = cv2.resize(image, (args.width, round(h * scale)))
    if args.box:
        box = tuple(int(v * scale) for v in args.box)
    else:
        found = largest_label_box(path, w, h)
        if found is None:
            parser.error("no --box given and no YOLO label found next to the image")
        box = tuple(int(v * scale) for v in found)
    n = render(image, box, Path(args.out), fps=args.fps, patch=PatchType(args.patch))  # type: ignore[arg-type]
    print(f"wrote {args.out}: {n} frames, attack on box {box}")


if __name__ == "__main__":
    main()
