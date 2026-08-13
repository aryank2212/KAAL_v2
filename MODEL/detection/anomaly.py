import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, threshold=None, percentile=95):
        """
        Anomaly Detection via Confidence Entropy.
        
        Args:
            threshold: Pre-set threshold (optional).
            percentile: Percentile of calibration scores to set as threshold if calibrating.
        """
        self.threshold = threshold
        self.percentile = percentile
        self.calibration_scores = []

    def compute_entropy_score(self, preds):
        """
        Computes the entropy of the bounding box class confidences.
        Adversarial inputs often distribute confidence mass more untidily across classes
        or scatter false positive confidences across anchors.
        
        Args:
            preds: Raw tensor outputs from YOLOv8 model.
        Returns:
            Tensor of anomaly scores (higher = more anomalous) per image.
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
            
        # YOLOv8 format: (B, 4 + num_classes, anchors)
        class_preds = preds[:, 4:, :] 
        
        eps = 1e-6
        # Treat class predictions as independent Bernoulli distributions (since they are typically Sigmoid outputs)
        p = torch.clamp(class_preds, eps, 1 - eps)
        
        # Binary entropy per anchor per class
        entropy = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
        
        # Average entropy over all anchors and classes for each image
        # This acts as our anomaly score.
        score = entropy.mean(dim=(1, 2))
        return score

    def calibrate(self, dataloader, model):
        """
        Automatically determine threshold using a clean calibration set.
        """
        logger.info("Calibrating Anomaly Detector on clean data...")
        model.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                # Need images in (B, 3, H, W) normalized to [0,1]
                images = batch["img"].float() / 255.0
                images = images.to(model.device)
                
                preds = model(images)
                batch_scores = self.compute_entropy_score(preds)
                scores.extend(batch_scores.cpu().numpy())
                
        self.calibration_scores = scores
        self.threshold = np.percentile(scores, self.percentile)
        logger.info("Calibration complete. Threshold set at %dth percentile: %.4f", self.percentile, self.threshold)
        return self.threshold

    def is_anomalous(self, preds):
        """
        Checks if the predictions are anomalous.
        Returns boolean mask corresponding to batch.
        """
        if self.threshold is None:
            raise ValueError("Threshold is not set. Calibrate or set threshold manually first.")
            
        scores = self.compute_entropy_score(preds)
        return scores > self.threshold
