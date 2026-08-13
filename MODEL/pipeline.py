import os
import logging
import argparse
from .config import Config, DefenseMode
from .trainer import AdversarialTrainer
from .evaluator import RobustnessEvaluator
from ultralytics import YOLO
import yaml

logger = logging.getLogger(__name__)


def run_pipeline(args):
    # 1. Initialize Config
    config = Config(
        data_yaml=args.data,
        model_weights=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        defense_mode=DefenseMode(args.defense),
        fft_threshold=args.fft_threshold,
        fft_high_freq_ratio=args.high_freq_ratio,
        fft_region_mode=not args.frame_fft,
        fft_region_min_size=args.region_min_size,
        calibration_profile=args.calibration,
    )

    # 2. Dataset Preparation & Training
    best_model_path = args.model
    if args.train:
        logger.info("Starting Training Phase")
        # Handle resume logic
        if args.resume:
            # Look for checkpoint in multiple possible nested locations
            possible_checkpoints = [
                os.path.join(
                    config.output_dir, "train", "adversarial_finetune", "weights", "last.pt"
                ),
                os.path.join(
                    "runs",
                    "detect",
                    config.output_dir,
                    "train",
                    "adversarial_finetune",
                    "weights",
                    "last.pt",
                ),
                "runs/detect/runs/adversarial/train/adversarial_finetune/weights/last.pt",
            ]
            checkpoint = None
            for cp in possible_checkpoints:
                if os.path.exists(cp):
                    checkpoint = os.path.abspath(cp)
                    break

            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint}")
                config.model_weights = checkpoint
                config.resume = True
            else:
                logger.warning("No checkpoint found in expected locations. Starting from scratch.")

        trainer = AdversarialTrainer(config)

        best_model_path = trainer.train()
        logger.info(f"Training complete. Best model: {best_model_path}")

    # 3. Load Trained Model
    model = YOLO(best_model_path)

    # 4. Evaluation
    logger.info("Starting Evaluation Phase")
    evaluator = RobustnessEvaluator(model, config)

    # Resolve validation paths from yaml
    with open(config.data_yaml, "r") as f:
        data = yaml.safe_load(f)

    val_images_dir = data["val"]
    if not os.path.isabs(val_images_dir):
        val_images_dir = os.path.join(os.path.dirname(config.data_yaml), val_images_dir)

    # Infer label dir
    val_labels_dir = os.path.join(os.path.dirname(val_images_dir), "labels")
    if not os.path.exists(val_labels_dir):
        # Try parent/labels/val
        val_labels_dir = os.path.join(
            os.path.dirname(os.path.dirname(val_images_dir)),
            "labels",
            os.path.basename(val_images_dir),
        )

    evaluator.run_full_evaluation(val_images_dir, val_labels_dir)

    report_path = os.path.join(config.output_dir, "evaluation_report.txt")
    evaluator.save_report(report_path)
    logger.info(f"Full pipeline execution complete. Report at {report_path}")


def main():
    parser = argparse.ArgumentParser(description="KAAL: Robust ML Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Pretrained or starting weights"
    )
    parser.add_argument("--train", action="store_true", help="Run training before evaluation")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="dual_stream",
        choices=[d.value for d in DefenseMode],
        help="Defense strategy",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--fft-threshold",
        type=float,
        default=None,
        help="FFT anomaly threshold (default: calibration profile or 0.3)",
    )
    parser.add_argument(
        "--high-freq-ratio",
        type=float,
        default=0.25,
        help="Fraction of spectrum considered high-frequency",
    )
    parser.add_argument(
        "--frame-fft",
        action="store_true",
        help="Use whole-frame FFT only (disable per-region mode)",
    )
    parser.add_argument(
        "--region-min-size",
        type=int,
        default=16,
        help="Skip per-region FFT for detections smaller than this",
    )
    parser.add_argument(
        "--calibration", type=str, default=None, help="Path to a saved calibration profile (JSON)"
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
