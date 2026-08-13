import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer

from ..attacks.pgd import pgd_attack

class RobustDetectionTrainer(DetectionTrainer):
    """
    Custom YOLOv8 Trainer designed for Adversarial Training.
    Intercepts the batch processing step to dynamically perturb images.
    """
    
    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        # Configurable attack parameters could be fetched from overrides or defaults
        self.adv_epsilon = overrides.get("adv_epsilon", 4/255) if overrides else 4/255
        self.adv_iters = overrides.get("adv_iters", 5) if overrides else 5
    
    def preprocess_batch(self, batch):
        """
        Preprocesses batch (normalization, etc.) and injects PGD attack.
        """
        # Call the standard YOLOv8 preprocessing first (scales to 0-1 range, device transfer)
        batch = super().preprocess_batch(batch)
        
        # Only inject if the model is ready and training
        if hasattr(self, 'model') and self.model is not None:
            # We temporarily switch model to eval mode to prevent disrupting 
            # batch normalization running stats during attack generation.
            training_mode = self.model.training
            self.model.eval()
            
            images = batch["img"].clone().detach()
            B = images.shape[0]
            
            # Apply attack to half of the batch to mix clean and adversarial samples
            if B > 1:
                split_idx = B // 2
                clean_half = images[:split_idx]
                adv_half_source = images[split_idx:]
                
                # Generate adversarial permutations
                with torch.enable_grad():
                    adv_half = pgd_attack(
                        self.model, 
                        adv_half_source, 
                        epsilon=self.adv_epsilon, 
                        alpha=self.adv_epsilon / 2.0, 
                        iters=self.adv_iters, 
                        target_type="suppress"
                    )
                
                # Re-integrate the adversarial halves back into the batch
                batch["img"] = torch.cat([clean_half, adv_half.detach()], dim=0)
            
            # Restore the original training mode
            if training_mode:
                self.model.train()
                
        return batch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Adversarial Training for YOLOv8")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to initial model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    trainer = RobustDetectionTrainer(overrides={
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": "runs/robustness",
        "name": "adv_training_run"
    })
    trainer.train()
