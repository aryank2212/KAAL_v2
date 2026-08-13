"""
dataset.py — Dataset loading, augmentation, and adversarial patch integration.

Reads a YOLO-format dataset (images + labels), applies standard augmentations,
and optionally injects adversarial patches to create a mixed training set.
The augmented dataset is written to a temp directory so Ultralytics can
consume it via a modified data.yaml.
"""

import os
import cv2
import yaml
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .config import Config
from .adversarial_patch import AdversarialPatchInjector, parse_yolo_labels
from .augmentations import add_patch, add_noise, blur, occlude

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """
    Prepares a YOLO-format dataset with adversarial augmentation.

    Workflow:
      1. Parse data.yaml to locate image/label directories.
      2. Copy the original dataset to a working directory.
      3. Create adversarial copies of training images and add them to the set.
      4. Write an updated data.yaml pointing to the augmented directory.
    """

    def __init__(self, config: Config):
        self.config = config
        self.injector = AdversarialPatchInjector(
            patch_type=config.patch_type,
            scale_range=config.patch_scale_range,
            alpha=config.patch_alpha,
            patch_image_path=config.patch_image_path,
            use_perspective=config.patch_perspective,
            use_lighting=config.patch_lighting,
            grad_steps=config.patch_opt_steps,
            grad_lr=config.patch_opt_lr,
        )
        self.data_config: Dict = {}
        self.augmented_data_yaml: Optional[str] = None

    def load_data_config(self) -> Dict:
        """Parse the original data.yaml and resolve relative paths."""
        with open(self.config.data_yaml, "r") as f:
            self.data_config = yaml.safe_load(f)

        # Resolve paths relative to the data.yaml location
        base_dir = str(Path(self.config.data_yaml).resolve().parent)
        for key in ("train", "val", "test"):
            if key in self.data_config:
                path = self.data_config[key]
                if not os.path.isabs(path):
                    self.data_config[key] = os.path.join(base_dir, path)
        return self.data_config

    def prepare_augmented_dataset(self) -> str:
        """
        Build an augmented training set with adversarial patches injected.

        Returns:
            Path to the new data.yaml that includes both clean and patched images.
        """
        if not self.data_config:
            self.load_data_config()

        # Output directory for the augmented dataset
        aug_dir = os.path.join(self.config.output_dir, "augmented_dataset")
        os.makedirs(aug_dir, exist_ok=True)

        # Copy val/test splits unchanged, preserving images/labels structure
        for split in ("val", "test"):
            if split in self.data_config and os.path.exists(self.data_config[split]):
                src_path = self.data_config[split]
                dst = os.path.join(aug_dir, split)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                # Copy images
                dst_images = os.path.join(dst, "images")
                os.makedirs(dst_images, exist_ok=True)
                for f in os.listdir(src_path):
                    src_f = os.path.join(src_path, f)
                    if os.path.isfile(src_f):
                        shutil.copy2(src_f, os.path.join(dst_images, f))
                # Copy corresponding labels
                src_labels = self._infer_labels_dir(src_path)
                if src_labels and os.path.isdir(src_labels):
                    dst_labels = os.path.join(dst, "labels")
                    os.makedirs(dst_labels, exist_ok=True)
                    for f in os.listdir(src_labels):
                        src_f = os.path.join(src_labels, f)
                        if os.path.isfile(src_f):
                            shutil.copy2(src_f, os.path.join(dst_labels, f))

        # Process training split: keep originals + add adversarial copies
        train_src = self.data_config.get("train", "")
        if not os.path.exists(train_src):
            raise FileNotFoundError(f"Training image directory not found: {train_src}")

        train_dst = os.path.join(aug_dir, "train")
        images_dst = os.path.join(train_dst, "images")
        labels_dst = os.path.join(train_dst, "labels")
        os.makedirs(images_dst, exist_ok=True)
        os.makedirs(labels_dst, exist_ok=True)

        # Determine source label directory
        # YOLO convention: labels dir is a sibling of images dir
        label_src = self._infer_labels_dir(train_src)

        # Collect all image files
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = [f for f in os.listdir(train_src) if Path(f).suffix.lower() in img_extensions]

        logger.info(f"Processing {len(image_files)} training images...")
        patched_count = 0

        for idx, img_name in enumerate(image_files):
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(image_files)} images processed...")

            img_path = os.path.join(train_src, img_name)
            stem = Path(img_name).stem
            ext = Path(img_name).suffix
            label_name = stem + ".txt"
            label_path = os.path.join(label_src, label_name) if label_src else ""

            # ── Copy original image and label ─────────────────────────────
            shutil.copy2(img_path, os.path.join(images_dst, img_name))
            if label_path and os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(labels_dst, label_name))

            # ── Create adversarial copies using all 3 attack types ─────────
            image = cv2.imread(img_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            # Parse bounding boxes for patch placement
            _, bboxes = parse_yolo_labels(label_path, img_w, img_h)

            # Apply standard augmentations as base
            augmented = self._apply_standard_augmentations(image)

            # Attack 1: Adversarial patch (from test/clean)
            if np.random.random() < self.config.patch_prob:
                patched_img = add_patch(augmented)
                adv_name = f"{stem}_patch{ext}"
                cv2.imwrite(os.path.join(images_dst, adv_name), patched_img)
                if label_path and os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(labels_dst, f"{stem}_patch.txt"))
                patched_count += 1

            # Attack 2: Noise + blur (from test/patched)
            if np.random.random() < self.config.patch_prob:
                noisy_img = blur(add_noise(augmented))
                adv_name = f"{stem}_noisy{ext}"
                cv2.imwrite(os.path.join(images_dst, adv_name), noisy_img)
                if label_path and os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(labels_dst, f"{stem}_noisy.txt"))
                patched_count += 1

            # Attack 3: Occlusion (from test/moisy)
            if np.random.random() < self.config.patch_prob:
                occluded_img = occlude(augmented)
                adv_name = f"{stem}_occlude{ext}"
                cv2.imwrite(os.path.join(images_dst, adv_name), occluded_img)
                if label_path and os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(labels_dst, f"{stem}_occlude.txt"))
                patched_count += 1

        logger.info(
            f"Created {patched_count} adversarial copies "
            f"(total training images: {len(image_files) + patched_count})"
        )

        # ── Write updated data.yaml ───────────────────────────────────────
        new_config = self.data_config.copy()
        new_config["train"] = os.path.join(os.path.abspath(train_dst), "images")
        for split in ("val", "test"):
            if split in new_config:
                split_dir = os.path.join(os.path.abspath(aug_dir), split)
                split_img = os.path.join(split_dir, "images")
                if os.path.isdir(split_img):
                    new_config[split] = split_img
                elif os.path.isdir(split_dir):
                    new_config[split] = split_dir

        # Ensure val has proper label pairing for YOLO validator
        val_labels = os.path.join(os.path.abspath(aug_dir), "val", "labels")
        if os.path.isdir(val_labels):
            logger.info(f"Validation labels: {val_labels} ({len(os.listdir(val_labels))} files)")

        self.augmented_data_yaml = os.path.join(aug_dir, "data.yaml")
        with open(self.augmented_data_yaml, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        logger.info(f"Augmented data.yaml written to: {self.augmented_data_yaml}")
        return self.augmented_data_yaml

    # ── Standard Augmentations ────────────────────────────────────────────

    @staticmethod
    def _apply_standard_augmentations(image: np.ndarray) -> np.ndarray:
        """
        Apply standard augmentations using OpenCV:
          - Random horizontal flip
          - Random scale (90%–110%)
          - Random brightness / contrast adjustment
        """
        result = image.copy()

        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            result = cv2.flip(result, 1)

        # Random scale jitter
        scale = np.random.uniform(0.9, 1.1)
        h, w = result.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        result = cv2.resize(result, (new_w, new_h))
        # Crop or pad back to original size
        result = _resize_pad(result, w, h)

        # Random brightness and contrast
        alpha = np.random.uniform(0.8, 1.2)  # contrast
        beta = np.random.randint(-20, 21)  # brightness
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        return result

    @staticmethod
    def _infer_labels_dir(images_dir: str) -> Optional[str]:
        """
        Infer the labels directory from the images directory following YOLO convention.
        e.g., .../images/ → .../labels/
        """
        images_path = Path(images_dir)
        # Try replacing the last 'images' component with 'labels'
        parts = list(images_path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].lower() == "images":
                parts[i] = "labels"
                candidate = str(Path(*parts))
                if os.path.isdir(candidate):
                    return candidate
                break
        # Fallback: sibling "labels" directory
        sibling = images_path.parent / "labels"
        return str(sibling) if sibling.is_dir() else None


def _resize_pad(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to target dimensions, center-cropping or zero-padding as needed."""
    h, w = image.shape[:2]
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center crop if larger, center pad if smaller
    src_y = max(0, (h - target_h) // 2)
    src_x = max(0, (w - target_w) // 2)
    dst_y = max(0, (target_h - h) // 2)
    dst_x = max(0, (target_w - w) // 2)

    copy_h = min(h, target_h)
    copy_w = min(w, target_w)

    result[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = image[
        src_y : src_y + copy_h, src_x : src_x + copy_w
    ]

    return result
