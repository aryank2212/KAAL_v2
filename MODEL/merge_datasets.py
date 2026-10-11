"""
merge_datasets.py — [legacy] Merge two YOLO datasets with class-id offsets.

Copies dataset 1 as-is and dataset 2 with its class ids shifted past
dataset 1's, prefixing file names (``aircraft_`` / ``ground_``). The
``__main__`` block has hard-coded ``D:/...`` paths from the original Windows
setup. Superseded by ``MODEL/build_dataset.py`` (taxonomy mapping, leakage
control, registry).
"""

import os
import shutil
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def merge_yolo_datasets(dataset1_yaml_path, dataset2_yaml_path, output_dir):
    # Load dataset 1 config
    with open(dataset1_yaml_path, "r") as f:
        ds1 = yaml.safe_load(f)

    # Load dataset 2 config
    with open(dataset2_yaml_path, "r") as f:
        ds2 = yaml.safe_load(f)

    # Unified names list
    ds1_names = ds1["names"]
    if isinstance(ds1_names, dict):
        ds1_names = [ds1_names[i] for i in range(len(ds1_names))]

    ds2_names = ds2["names"]
    if isinstance(ds2_names, dict):
        ds2_names = [ds2_names[i] for i in range(len(ds2_names))]

    merged_names = list(ds1_names) + list(ds2_names)
    ds2_class_offset = len(ds1_names)

    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    def copy_dataset(ds_config, yaml_path, offset, prefix):
        base_dir = os.path.dirname(os.path.abspath(yaml_path))
        for split in splits:
            if split not in ds_config:
                continue

            # Resolve relative path using base_dir
            split_rel = ds_config[split]
            if isinstance(split_rel, list):
                split_rel = split_rel[0]

            if not os.path.isabs(split_rel):
                images_dir = os.path.normpath(os.path.join(base_dir, split_rel))
            else:
                images_dir = split_rel

            labels_dir = images_dir.replace("images", "labels")
            if not os.path.exists(images_dir):
                continue

            img_files = [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
            ]
            for img_name in img_files:
                src_img = os.path.join(images_dir, img_name)
                dst_img = os.path.join(output_dir, split, "images", f"{prefix}_{img_name}")
                shutil.copy2(src_img, dst_img)

                label_name = os.path.splitext(img_name)[0] + ".txt"
                src_label = os.path.join(labels_dir, label_name)
                dst_label = os.path.join(output_dir, split, "labels", f"{prefix}_{label_name}")

                if os.path.exists(src_label):
                    with open(src_label, "r") as fr, open(dst_label, "w") as fw:
                        for line in fr:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                new_class = int(parts[0]) + offset
                                fw.write(f"{new_class} {' '.join(parts[1:])}\n")

    logger.info("Merging dataset 1 (Aircraft)...")
    copy_dataset(ds1, dataset1_yaml_path, offset=0, prefix="aircraft")
    logger.info("Merging dataset 2 (Ground)...")
    copy_dataset(ds2, dataset2_yaml_path, offset=ds2_class_offset, prefix="ground")

    # Determine the output data.yaml formatting required for Ultralytics YOLOv8
    merged_yaml = {
        "path": os.path.abspath(output_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(merged_names),
        "names": merged_names,
    }

    yaml_out_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_out_path, "w") as f:
        yaml.dump(merged_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("\nMerged dataset successfully created at: %s", output_dir)
    logger.info("Total classes: %d (0-%d)", len(merged_names), len(merged_names) - 1)
    logger.info("Unified data.yaml saved at: %s", yaml_out_path)


merge_yolo_datasets(
    "D:/Github/Hackathons/KAAL/MODEL/Datasets/aircraft.v1i.yolov8/data.yaml",
    "D:/Github/Hackathons/KAAL/MODEL/Datasets/a/archive/dataset.yaml",
    "D:/Github/Hackathons/KAAL/MODEL/Datasets/merged_aircraft_ground",
)
