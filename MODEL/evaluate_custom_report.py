import json
import cv2
import logging
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_custom_eval(model_path, images_dir, coco_json_path):
    logger.info("--- Running Evaluation ---")
    logger.info("Model: %s", model_path)
    logger.info("Images: %s", images_dir)
    logger.info("Ground Truth: %s", coco_json_path)

    # 1. Load Model
    model = YOLO(model_path)
    model_names = model.names

    # 2. Load Ground Truth (COCO)
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Map image file names to their annotations
    image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        img_name = image_id_to_name[ann["image_id"]]
        cat_name = category_id_to_name[ann["category_id"]]
        # COCO bbox: [x, y, width, height]
        annotations[img_name].append(
            {
                "category_id": ann["category_id"],
                "category_name": cat_name,
                "bbox": ann["bbox"],  # [x, y, w, h]
            }
        )

    # 3. Run Inference
    images_dir_path = Path(images_dir)
    image_files = list(images_dir_path.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    results_summary = {
        "total_images": len(image_files),
        "total_detections": 0,
        "detections_by_class": defaultdict(int),
        "gt_by_class": defaultdict(int),
        "matches": 0,
        "mismatches": 0,
    }

    # Count GT classes
    for img_name, anns in annotations.items():
        for ann in anns:
            results_summary["gt_by_class"][ann["category_name"]] += 1

    logger.info("Processing %d images...", len(image_files))

    for img_f in image_files:
        img_name = img_f.name
        img = cv2.imread(str(img_f))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Run YOLO
        # Using the quality-tuned settings from our plan
        results = model.predict(
            source=str(img_f), conf=0.45, iou=0.40, agnostic_nms=True, verbose=False
        )

        yolo_dets = []
        for r in results:
            for box in r.boxes:
                # box.xyxy: [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu())
                conf = float(box.conf[0].cpu())
                yolo_dets.append(
                    {"cls_id": cls_id, "cls_name": model_names[cls_id], "conf": conf, "bbox": xyxy}
                )
                results_summary["total_detections"] += 1
                results_summary["detections_by_class"][model_names[cls_id]] += 1

        # Compare with GT if exists
        if img_name in annotations:
            gt_anns = annotations[img_name]
            # This is a loose "correctness" check since classes might not match
            # We check if anything was detected where something was supposed to be
            for gt in gt_anns:
                gx, gy, gw, gh = gt["bbox"]
                # Convert to xyxy
                g_xyxy = [gx, gy, gx + gw, gy + gh]

                matched = False
                for det in yolo_dets:
                    iou = compute_iou(det["bbox"], g_xyxy)
                    if iou > 0.5:
                        matched = True
                        break

                if matched:
                    results_summary["matches"] += 1
                else:
                    results_summary["mismatches"] += 1

    # 4. Generate Report
    report = f"""
    ROBUSTNESS EVALUATION REPORT (CUSTOM DATASET)
    =============================================
    Model: {model_path}
    Dataset: {images_dir}

    Summary Statistics:
    -------------------
    Total Images Tested:      {results_summary["total_images"]}
    Total Ground Truth Boxes: {sum(results_summary["gt_by_class"].values())}
    Total Model Detections:   {results_summary["total_detections"]}

    Accuracy (IOU > 0.5):
    ---------------------
    Objects correctly localized: {results_summary["matches"]}
    Objects missed (FN):         {results_summary["mismatches"]}
    Localization Success Rate:   {(results_summary["matches"] / (results_summary["matches"] + results_summary["mismatches"] + 1e-6)) * 100:.2f}%

    Model Class Distribution (What the model 'thinks' they are):
    ----------------------------------------------------------
    """
    for cls, count in sorted(
        results_summary["detections_by_class"].items(), key=lambda x: x[1], reverse=True
    ):
        report += f"    - {cls}: {count}\n"

    report += "\n    Ground Truth Class Distribution (What they actually are):\n"
    report += "    ------------------------------------------------------\n"
    for cls, count in sorted(
        results_summary["gt_by_class"].items(), key=lambda x: x[1], reverse=True
    ):
        report += f"    - {cls}: {count}\n"

    logger.info("%s", report)
    with open("custom_evaluation_report.txt", "w") as f:
        f.write(report)
    logger.info("Report saved to custom_evaluation_report.txt")


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-10)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--json", required=True)
    args = p.parse_args()
    run_custom_eval(args.model, args.source, args.json)
