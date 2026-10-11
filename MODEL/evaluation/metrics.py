"""
metrics.py — [legacy] Container for clean / adversarial / defended mAP, TP/FP/FN
counts and anomaly-score AUROC used by ``MODEL/evaluation/evaluate.py``.
"""

import numpy as np

class RobustnessMetrics:
    def __init__(self):
        self.clean_map = 0.0
        self.adv_map = 0.0
        self.defended_map = 0.0
        
        self.clean_results = {"TP": 0, "FP": 0, "FN": 0}
        self.adv_results = {"TP": 0, "FP": 0, "FN": 0}
        self.defended_results = {"TP": 0, "FP": 0, "FN": 0}
        
        # Anomaly tracking
        self.anomaly_labels = [] # 1 for adv, 0 for clean
        self.anomaly_scores = []
        
    def add_anomaly_score(self, score, is_adv):
        self.anomaly_scores.append(score)
        self.anomaly_labels.append(1 if is_adv else 0)
        
    def compute_anomaly_metrics(self, threshold):
        scores = np.array(self.anomaly_scores)
        labels = np.array(self.anomaly_labels)
        
        preds = (scores > threshold).astype(int)
        
        TP = np.sum((preds == 1) & (labels == 1))
        FP = np.sum((preds == 1) & (labels == 0))
        TN = np.sum((preds == 0) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))
        
        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        detection_rate = recall
        
        return {
            "Precision": precision,
            "Recall": recall,
            "DetectionRate": detection_rate,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN
        }
        
    def summary(self):
        attack_drop = self.clean_map - self.adv_map
        recovery = self.defended_map - self.adv_map
        net_change = self.defended_map - self.clean_map
        
        report = f"""
=========================================================
  ROBUSTNESS EVALUATION REPORT
=========================================================

  Scenario                 | mAP@0.5 | TP   | FP   | FN  
  -------------------------+---------+------+------+-----
  Clean (baseline)         | {self.clean_map:.4f}  | {self.clean_results['TP']} | {self.clean_results['FP']} | {self.clean_results['FN']}
  Adversarial (attack)     | {self.adv_map:.4f}  | {self.adv_results['TP']} | {self.adv_results['FP']} | {self.adv_results['FN']}
  Defended (pipeline)      | {self.defended_map:.4f}  | {self.defended_results['TP']} | {self.defended_results['FP']} | {self.defended_results['FN']}

---------------------------------------------------------
  Metrics Change:
---------------------------------------------------------
  Attack Drop: {attack_drop:+.4f}
  Recovery:    {recovery:+.4f}
  Net Change:  {net_change:+.4f}
=========================================================
"""
        return report
