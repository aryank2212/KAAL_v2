import os
import argparse
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_robust_inference(model_path: str, source: str, output_dir: str):
    """
    Runs YOLOv8 inference with explicit NMS and confidence thresholding
    tuned to fix fragmented output (parts of objects detected as full objects).

    Args:
        model_path: Path to the trained YOLOv8 weights (.pt file).
        source: Path to an image, video, or directory to run inference on.
        output_dir: Directory to save the predicted output.
    """
    if not os.path.exists(model_path):
        logger.error("Model not found at %s.", model_path)
        return

    logger.info("Loading model from %s...", model_path)
    model = YOLO(model_path)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("\n--- Running Quality-Tuned Inference ---")
    logger.info("Source: %s", source)
    logger.info("Applying: Conf=0.45, IoU=0.40, Agnostic NMS (to merge fragments)")

    # Key settings for fragmented bounding boxes:
    # 1. conf=0.45: Drops low confidence weak edge detections (like lone wings).
    # 2. iou=0.40: A stricter NMS. If two boxes overlap by > 40%, it drops the weaker one.
    # 3. agnostic_nms=True: If model predicts a 'wing' as an 'A10' but the 'body' as 'F16'
    #    and they heavily overlap, it ignores class labels and merges them into one box.
    results = model.predict(
        source=source,
        conf=0.45,
        iou=0.40,
        agnostic_nms=True,
        max_det=30,
        project=output_dir,
        name="predict_tuned",
        save=True,
        save_txt=True,
    )

    logger.info("\nInference complete. Results saved in %s/predict_tuned/", output_dir)

    # Optionally print summary
    for r in results:
        logger.info("Detected %d objects in %s.", len(r.boxes), r.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fragment-Resistant YOLO Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument(
        "--source", type=str, required=True, help="Input media (e.g., test/images or single file)"
    )
    parser.add_argument(
        "--out", type=str, default="runs/detect/robust_inference", help="Output directory"
    )

    args = parser.parse_args()
    run_robust_inference(args.model, args.source, args.out)
