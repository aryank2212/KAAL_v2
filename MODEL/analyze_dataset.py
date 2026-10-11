"""
analyze_dataset.py — [legacy] Flag tiny / fragmented boxes in a YOLO dataset.

Counts boxes whose area is below ``--threshold`` of the image (default 0.03)
— used on the original aircraft set to find wings or edges annotated as
separate objects. KAAL datasets are now built and checked by
``MODEL/build_dataset.py`` (minimum box size, manifest counts).

    python -m MODEL.analyze_dataset --data <yolo dataset dir> --threshold 0.03

Status: legacy (see Docs/FILE_REFERENCE.md); default path is the old
``Datasets/aircraft.v1i.yolov8`` export.
"""

import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_bbox_sizes(dataset_path: str, split: str = "train", threshold: float = 0.05):
    """
    Analyzes YOLO format annotations to detect potentially fragmented bounding boxes
    (e.g., wings/edges annotated separately from the main body).

    Args:
        dataset_path: Root path to the YOLO dataset.
        split: "train" or "valid".
        threshold: Fractional area below which a bbox is considered "fragmented".
                   E.g., 0.05 means the box occupies less than 5% of the total image area.
    """
    labels_dir = os.path.join(dataset_path, split, "labels")
    if not os.path.exists(labels_dir):
        logger.error("%s does not exist.", labels_dir)
        return

    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not txt_files:
        logger.error("No label files found in %s.", labels_dir)
        return

    total_boxes = 0
    small_boxes = 0

    logger.info(
        "Analyzing %d label files in %s set for bounding box fragmentation...",
        len(txt_files),
        split,
    )

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    total_boxes += 1
                    # YOLO format: class_id x_center y_center width_normalized height_normalized
                    w = float(parts[3])
                    h = float(parts[4])
                    area = w * h

                    if area < threshold:
                        small_boxes += 1
                        logger.warning(
                            "Fragmented box detected in %s: w=%.3f, h=%.3f, area=%.4f",
                            os.path.basename(txt_file),
                            w,
                            h,
                            area,
                        )

    if total_boxes == 0:
        logger.info("No bounding boxes found.")
        return

    logger.info("\n--- Summary ---")
    logger.info("Total Bounding Boxes: %d", total_boxes)
    logger.info("Fragmented Boxes (Area < %.0f%%): %d", threshold * 100, small_boxes)
    logger.info("Fragmentation Ratio: %.2f%%", (small_boxes / total_boxes) * 100)
    logger.info("If the ratio is high, consider merging bounding boxes or cleaning the dataset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find fragmented bounding boxes in dataset.")
    parser.add_argument(
        "--data", type=str, default="Datasets/aircraft.v1i.yolov8", help="Path to YOLO dataset"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.03, help="Area threshold to consider fragmented"
    )
    args = parser.parse_args()

    analyze_bbox_sizes(args.data, "train", args.threshold)
