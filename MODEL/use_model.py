"""
use_model.py — [legacy] One-off YOLO inference over a folder.

Hard-coded Windows paths (``D:\\...``) from the original setup; edit them
before running. For current use: ``python -m MODEL.compare_detectors`` or the
backend (``KAAL_CAMERA_SOURCE`` for video files).
"""

import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 1. Load the model (We're using YOLOv8m here, but you can swap this with your 'best.pt' once fully trained)
model_path = r"D:\Github\Hackathons\KAAL\MODEL\yolov8m.pt"
logger.info("Loading model: %s", model_path)
model = YOLO(model_path)

# 2. Specify the directory containing the images you want to test
source_dir = r"D:\Github\Hackathons\KAAL\MODEL\Datasets\military-artillery-and-tanks\2nd part int\2nd part int"

logger.info("Running inference on directory: %s...", source_dir)

# 3. Run the prediction
results = model.predict(
    source=source_dir,
    conf=0.25,  # Minimum confidence threshold stringency
    iou=0.45,  # Removes overlapping duplicate boxes
    show=False,  # Disabled UI popup since we are processing a large folder
    save=True,  # Saves the output image
    project=r"D:\Github\Hackathons\KAAL\MODEL\Processed_Outputs",
    name="military_artillery_2nd_part",  # Final folder: Processed_Outputs\military_artillery_2nd_part
    exist_ok=True,  # Overwrites if folder exists
)

# 4. How to process the raw output data yourself
logger.info("\n--- RESULTS DATA ---")
for result in results:
    # results comprises a list of objects (one for each image processed)
    boxes = result.boxes
    logger.info("Found %d objects in the image.\n", len(boxes))

    for box in boxes:
        # Extract Box Data
        cls_id = int(box.cls[0].item())  # Class ID (0-based index)
        conf = float(box.conf[0].item())  # Confidence score (0.0 to 1.0)
        xyxy = box.xyxy[0].cpu().numpy()  # Coordinates [x1, y1, x2, y2]

        # Get actual class name from the model's dictionary
        class_name = model.names[cls_id]

        logger.info("➤ Object:  %s", class_name)
        logger.info("  Conf:    %.1f%%", conf * 100)
        logger.info("  Coords:  [%.1f, %.1f, %.1f, %.1f]\n", *xyxy)

logger.info(
    "Processing complete! Check the D:\\Github\\Hackathons\\KAAL\\MODEL\\Processed_Outputs\\military_artillery_2nd_part folder for saved images."
)
