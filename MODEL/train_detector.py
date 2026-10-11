"""
train_detector.py — Fine-tune RF-DETR or a YOLO baseline on a KAAL dataset.

Both architectures read the same YOLO-format dataset produced by
``python -m MODEL.build_dataset`` so their results are directly comparable
(``python -m MODEL.compare_detectors``).

Examples:
    # RF-DETR Small (default resolution 512) — the recommended KAAL detector
    python -m MODEL.train_detector --arch rfdetr-small --epochs 12

    # Matched YOLO baseline on the same data and budget
    python -m MODEL.train_detector --arch yolo26s --epochs 12

    # Continue training an existing KAAL checkpoint on a newer dataset
    python -m MODEL.train_detector --arch rfdetr-small --data MODEL/Datasets/kaal_v4 \
        --init-weights MODEL/runs/train/rfdetr-small/checkpoint_best_total.pth \
        --epochs 10 --lr 5e-5 --name rfdetr-small-v4

    # Extend a model with a cosine tail (constant-LR runs that were still improving)
    python -m MODEL.train_detector --arch rfdetr-small --data MODEL/Datasets/kaal_v4 \
        --init-weights MODEL/runs/release/rfdetr-small-v4/rfdetr-small-v4.pth \
        --epochs 12 --lr 5e-5 --lr-scheduler cosine --lr-min-factor 0.05 --name rfdetr-small-v5

Outputs land in ``MODEL/runs/train/<name>``:
  RF-DETR  checkpoint_best_total.pth (inference checkpoint), metrics, tensorboard
  YOLO     weights/best.pt, results.csv, plots
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
YOLO_ARCHS = {"yolo26n": "yolo26n.pt", "yolo26s": "yolo26s.pt", "yolo26m": "yolo26m.pt",
              "yolov8n": "yolov8n.pt", "yolov8m": "yolov8m.pt"}  # fmt: skip
RFDETR_ARCHS = {"rfdetr-nano": "nano", "rfdetr-small": "small", "rfdetr-medium": "medium"}


def train_rfdetr(args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    import rfdetr

    from .detectors import RFDETR_SIZES

    model_cls = getattr(rfdetr, RFDETR_SIZES[RFDETR_ARCHS[args.arch]])
    # Warm start from a previous KAAL checkpoint (same taxonomy) instead of COCO.
    model = model_cls(pretrain_weights=args.init_weights) if args.init_weights else model_cls()
    kwargs: Dict[str, Any] = dict(
        dataset_dir=str(args.data),
        dataset_file="yolo",
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=str(out),
        num_workers=args.workers,
        early_stopping=True,
        early_stopping_patience=args.patience,
        lr_scheduler=args.lr_scheduler,
        checkpoint_interval=max(1, args.epochs // 4),
        tensorboard=True,
        progress_bar=None,
    )
    if args.lr_scheduler == "cosine":
        kwargs["lr_scheduler_kwargs"] = {"min_factor": args.lr_min_factor}
    if args.imgsz:
        kwargs["resolution"] = args.imgsz
    model.train(**kwargs)
    return {"weights": str(out / "checkpoint_best_total.pth")}


def train_yolo(args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(args.init_weights or YOLO_ARCHS[args.arch])
    model.train(
        data=str(Path(args.data) / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz or 640,
        batch=args.batch * args.grad_accum,  # same effective batch as RF-DETR; YOLO fits it in VRAM
        workers=args.workers,
        patience=args.patience,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        seed=0,
        plots=True,
        verbose=False,
    )
    return {"weights": str(out / "weights" / "best.pt")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR or YOLO on a KAAL dataset.")
    parser.add_argument("--arch", required=True, choices=[*RFDETR_ARCHS, *YOLO_ARCHS])
    parser.add_argument("--data", default=str(HERE / "Datasets" / "kaal_v3"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4, help="Per-step batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="RF-DETR learning rate")
    parser.add_argument(
        "--lr-scheduler",
        choices=("step", "cosine"),
        default="step",
        help="RF-DETR schedule: step (10x drop after 100 epochs, i.e. constant for short "
        "fine-tunes) or cosine (anneal to --lr-min-factor x lr over the run)",
    )
    parser.add_argument("--lr-min-factor", type=float, default=0.05, help="cosine floor")
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Input resolution (model default if unset)"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--name", default=None, help="Run name (default: <arch>)")
    parser.add_argument(
        "--init-weights",
        default=None,
        help="Start from this checkpoint instead of the COCO weights (fine-tune an existing "
        "KAAL model; must match --arch)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.data = str(Path(args.data).resolve())
    if args.init_weights:
        args.init_weights = str(Path(args.init_weights).resolve())  # before the chdir below
    out = HERE / "runs" / "train" / (args.name or args.arch)
    out.mkdir(parents=True, exist_ok=True)
    # Both frameworks download starter weights into the CWD; keep them in one
    # ignored cache dir instead of the repo root.
    weights_cache = HERE / "runs" / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_cache)
    start = time.time()
    result = train_rfdetr(args, out) if args.arch in RFDETR_ARCHS else train_yolo(args, out)
    result.update(arch=args.arch, data=str(args.data), epochs=args.epochs,
                  init_weights=args.init_weights, lr=args.lr, lr_scheduler=args.lr_scheduler,
                  train_hours=round((time.time() - start) / 3600, 2))  # fmt: skip
    (out / "kaal_train_summary.json").write_text(json.dumps(result, indent=2))
    logger.info("Done: %s", json.dumps(result))


if __name__ == "__main__":
    main()
