"""
trainer.py — Fine-tuning YOLOv8 with adversarial robustness.

Wraps Ultralytics YOLO training to support three defense modes:
  - NONE:                Standard training on clean data.
  - ADVERSARIAL_TRAINING: Train on a mix of clean + adversarial-patched images.
  - DUAL_STREAM:         Same as ADVERSARIAL_TRAINING (defense applied at inference).
"""

import os
import logging
from ultralytics import YOLO

from .config import Config, DefenseMode
from .dataset import DatasetPreparer

logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """
    Manages the end-to-end training workflow:
      1. Optionally prepare adversarial-augmented dataset.
      2. Load pretrained YOLOv8 model.
      3. Fine-tune with configurable hyperparameters.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model: YOLO = None
        self.dataset_preparer = DatasetPreparer(config)
        self.training_data_yaml: str = config.data_yaml

    def setup(self):
        """Load the pretrained model and prepare the dataset."""
        logger.info(f"Loading pretrained model: {self.config.model_weights}")
        self.model = YOLO(self.config.model_weights)

        # Prepare adversarial dataset if defense mode requires it
        if self.config.defense_mode in (
            DefenseMode.ADVERSARIAL_TRAINING,
            DefenseMode.DUAL_STREAM,
        ):
            # Check for existing augmented dataset
            aug_yaml = os.path.join(self.config.output_dir, "augmented_dataset", "data.yaml")
            if os.path.exists(aug_yaml):
                logger.info(f"Using existing augmented dataset: {aug_yaml}")
                self.training_data_yaml = aug_yaml
            else:
                logger.info("Preparing adversarial-augmented dataset...")
                self.training_data_yaml = self.dataset_preparer.prepare_augmented_dataset()
        else:
            logger.info("Using clean dataset (no adversarial augmentation)")
            self.training_data_yaml = self.config.data_yaml

    def train(self) -> str:
        """
        Run the training loop.

        Returns:
            Path to the best model weights.
        """
        if self.model is None:
            self.setup()

        # Output project directory
        project_dir = os.path.join(self.config.output_dir, "train")
        os.makedirs(project_dir, exist_ok=True)

        logger.info(
            f"Training Configuration: data={self.training_data_yaml}, "
            f"model={self.config.model_weights}, epochs={self.config.epochs}, "
            f"batch={self.config.batch_size}, lr={self.config.lr}, "
            f"imgsz={self.config.imgsz}, device={self.config.device}, "
            f"defense={self.config.defense_mode.value}"
        )

        # Launch Ultralytics training (with robust parameters to prevent fragmentation)
        resume_training = self.config.resume or self.config.model_weights.endswith("last.pt")
        self.model.train(
            data=self.training_data_yaml,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            imgsz=self.config.imgsz,
            lr0=self.config.lr,
            device=self.config.device,
            project=project_dir,
            name="adversarial_finetune",
            exist_ok=True,
            verbose=True,
            optimizer="AdamW",
            cos_lr=True,
            mosaic=1.0,
            mixup=0.2,
            multi_scale=False,  # Disable to prevent ZeroDivisionError in some environments
            workers=0,  # Prevent Windows memory/paging errors
            resume=resume_training,
        )

        # Locate best weights
        best_weights = os.path.join(project_dir, "adversarial_finetune", "weights", "best.pt")
        if os.path.exists(best_weights):
            logger.info(f"Best weights saved to: {best_weights}")
        else:
            # Fallback to last weights
            best_weights = os.path.join(project_dir, "adversarial_finetune", "weights", "last.pt")
            logger.info(f"Weights saved to: {best_weights}")

        return best_weights

    def get_model(self) -> YOLO:
        """Return the YOLO model (for evaluation/inference)."""
        return self.model
