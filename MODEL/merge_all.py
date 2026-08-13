import os
import json
import yaml
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def coco_to_yolo(coco_json_path, image_dir, output_labels_dir, category_id_map):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    image_id_to_info = {img["id"]: img for img in coco["images"]}
    os.makedirs(output_labels_dir, exist_ok=True)

    for ann in coco["annotations"]:
        img_info = image_id_to_info[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]
        img_name = img_info["file_name"]

        # COCO bbox: [x, y, w, h]
        x, y, w, h = ann["bbox"]

        # YOLO: [cls, cx, cy, w, h] normalized
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        new_cls = category_id_map.get(ann["category_id"])
        if new_cls is None:
            continue

        label_path = os.path.join(output_labels_dir, Path(img_name).stem + ".txt")
        with open(label_path, "a") as f:
            f.write(f"{new_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def merge():
    output_dir = "Datasets/master_data"
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # 1. Aircraft Dataset (0-39)
    logger.info("Processing Aircraft Dataset...")
    aircraft_base = "Datasets/aircraft.v1i.yolov8"
    for split in ["train", "valid", "test"]:
        target_split = "val" if split == "valid" else split
        img_src = os.path.join(aircraft_base, split, "images")
        lbl_src = os.path.join(aircraft_base, split, "labels")
        if not os.path.exists(img_src):
            continue

        for img in tqdm(os.listdir(img_src), desc=f"Aircraft {split}"):
            shutil.copy2(
                os.path.join(img_src, img),
                os.path.join(output_dir, target_split, "images", f"air_{img}"),
            )
            lbl = Path(img).stem + ".txt"
            if os.path.exists(os.path.join(lbl_src, lbl)):
                shutil.copy2(
                    os.path.join(lbl_src, lbl),
                    os.path.join(output_dir, target_split, "labels", f"air_{lbl}"),
                )

    # 2. Military (Artillery/Tanks) - COCO
    logger.info("Processing Military Ground Dataset...")
    military_json = "Datasets/military-artillery-and-tanks/instances_default.json"
    military_img_dir = "Datasets/military-artillery-and-tanks/2nd part int/2nd part int"
    # COCO IDs in file: 1:APC, 2:Tank, 3:Artillery, 4:AIRPLANE, 5:IFV, 6:Helicopter, 7:Engineering vehicle, 8:Soldiers
    category_id_map = {1: 40, 2: 41, 3: 42, 4: 43, 5: 44, 6: 45, 7: 46, 8: 47}

    # We'll split military images into train/val (90/10)
    img_files = [f for f in os.listdir(military_img_dir) if f.lower().endswith(".jpg")]
    split_idx = int(len(img_files) * 0.9)
    train_imgs = img_files[:split_idx]
    val_imgs = img_files[split_idx:]

    # Convert all military labels to a temp YOLO dir
    temp_mil_labels = "Datasets/temp_mil_labels"
    coco_to_yolo(military_json, military_img_dir, temp_mil_labels, category_id_map)

    for img in tqdm(train_imgs, desc="Military Train"):
        shutil.copy2(
            os.path.join(military_img_dir, img),
            os.path.join(output_dir, "train", "images", f"mil_{img}"),
        )
        lbl = Path(img).stem + ".txt"
        if os.path.exists(os.path.join(temp_mil_labels, lbl)):
            shutil.copy2(
                os.path.join(temp_mil_labels, lbl),
                os.path.join(output_dir, "train", "labels", f"mil_{lbl}"),
            )

    for img in tqdm(val_imgs, desc="Military Val"):
        shutil.copy2(
            os.path.join(military_img_dir, img),
            os.path.join(output_dir, "val", "images", f"mil_{img}"),
        )
        lbl = Path(img).stem + ".txt"
        if os.path.exists(os.path.join(temp_mil_labels, lbl)):
            shutil.copy2(
                os.path.join(temp_mil_labels, lbl),
                os.path.join(output_dir, "val", "labels", f"mil_{lbl}"),
            )

    # 3. Aerial Cars (48-51)
    logger.info("Processing Aerial Cars Dataset...")
    cars_base = "Datasets/aerial_cars"
    for split in ["train", "valid", "test"]:
        target_split = "val" if split == "valid" else split
        # Path in zip was ../train/images relative to data.yaml, so they are in aerial_cars/train/images
        img_src = os.path.join(cars_base, split, "images")
        lbl_src = os.path.join(cars_base, split, "labels")
        if not os.path.exists(img_src):
            continue

        for img in tqdm(os.listdir(img_src), desc=f"Cars {split}"):
            shutil.copy2(
                os.path.join(img_src, img),
                os.path.join(output_dir, target_split, "images", f"car_{img}"),
            )
            lbl = Path(img).stem + ".txt"
            if os.path.exists(os.path.join(lbl_src, lbl)):
                with (
                    open(os.path.join(lbl_src, lbl), "r") as f_in,
                    open(
                        os.path.join(output_dir, target_split, "labels", f"car_{lbl}"), "w"
                    ) as f_out,
                ):
                    for line in f_in:
                        parts = line.split()
                        if parts:
                            new_cls = int(parts[0]) + 48
                            f_out.write(f"{new_cls} {' '.join(parts[1:])}\n")

    # 4. Create data.yaml
    aircraft_names = [
        "A10",
        "A400M",
        "AG600",
        "AV8B",
        "B1",
        "B2",
        "B52",
        "Be200",
        "C130",
        "C17",
        "C5",
        "E2",
        "EF2000",
        "F117",
        "F14",
        "F15",
        "F16",
        "F18",
        "F22",
        "F35",
        "F4",
        "J20",
        "JAS39",
        "MQ9",
        "Mig31",
        "Mirage2000",
        "RQ4",
        "Rafale",
        "SR71",
        "Su34",
        "Su57",
        "Tornado",
        "Tu160",
        "Tu95",
        "U2",
        "US2",
        "V22",
        "Vulcan",
        "XB70",
        "YF23",
    ]
    mil_names = [
        "APC",
        "Tank",
        "Artillery",
        "AIRPLANE",
        "IFV",
        "Helicopter",
        "Engineering vehicle",
        "Soldiers",
    ]
    car_names = ["Aerial Car 0", "Aerial Car 1", "Aerial Car 2", "Aerial Car 3"]

    all_names = aircraft_names + mil_names + car_names

    master_yaml = {
        "path": os.path.abspath(output_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(all_names),
        "names": all_names,
    }
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(master_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Master dataset created at %s with %d classes.", output_dir, len(all_names))


if __name__ == "__main__":
    merge()
