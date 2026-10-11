"""
evaluate.py — [legacy] Clean / PGD-attacked / defended evaluation of a YOLO model.

Validates the clean model with Ultralytics, builds a PGD-perturbed copy of the
val split (``attacks/pgd.py``), applies ``defenses/preprocess.py`` and scores
the confidence-entropy detector (``detection/anomaly.py``). Superseded by
``MODEL/evaluator.py`` (seeded physical patches) and
``MODEL/compare_detectors.py``.

    python -m MODEL.evaluation.evaluate --model <weights.pt> --data <data.yaml>
"""

import os
import shutil
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DATASETS_DIR

from ..attacks.pgd import pgd_attack
from ..defenses.preprocess import defense_pipeline
from ..detection.anomaly import AnomalyDetector
from .metrics import RobustnessMetrics

logger = logging.getLogger(__name__)

def create_modified_dataset(original_yaml_path, output_dir_name, mode="adv", model=None, attack_kwargs=None, defense_kwargs=None, anomaly_detector=None, metrics_tracker=None):
    """
    Copies the validation set, applies transformations (attack/defense), and returns a new yaml path.
    Also computes anomaly scores on the fly.
    """
    with open(original_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
        
    # Find validation images path
    val_path = data_dict.get('val', None)
    if not val_path:
        raise ValueError("Data YAML must specify a 'val' path.")
        
    base_dir = Path(original_yaml_path).parent
    full_val_path = base_dir / val_path if not os.path.isabs(val_path) else Path(val_path)
    
    # Create new output directory
    out_dir = base_dir / output_dir_name
    out_imgs_dir = out_dir / "images"
    out_labels_dir = out_dir / "labels"
    out_imgs_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Also we need to get the labels source. Usually it shares the same structure 
    # as images, replacing 'images' with 'labels' in path.
    orig_labels_dir = Path(str(full_val_path).replace("images", "labels"))
    
    # Process each image
    image_files = list(full_val_path.glob("*.*"))
    logger.info("Processing %d images for %s dataset...", len(image_files), mode)
    
    transforms_pt = getattr(model.model, 'pt', True) if model else True
    
    for img_path in tqdm(image_files):
        # Read and resize image strictly if necessary, but we'll try to just convert to tensor
        orig_img = Image.open(img_path).convert("RGB")
        # To tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        if model:
            img_tensor = img_tensor.to(model.device)
            # Need strict resizing if YOLO demands it (imgsz usually 640)
            # For simplicity in evaluation generator, we just pass what we have.
            # But YOLO models require size divisible by 32.
            h, w = img_tensor.shape[2:]
            new_h, new_w = (h // 32) * 32, (w // 32) * 32
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear')
            
            with torch.enable_grad() if mode == "adv" else torch.no_grad():
                processed_tensor = img_tensor
                if mode == "adv" and attack_kwargs:
                    processed_tensor = pgd_attack(model.model, img_tensor, **attack_kwargs)
                elif mode == "defended" and defense_kwargs:
                    processed_tensor = defense_pipeline(img_tensor, **defense_kwargs)
                    
            # Anomaly scoring
            if anomaly_detector and metrics_tracker and model:
                # Get preds on the processed tensor
                with torch.no_grad():
                    preds = model.model(processed_tensor)
                score = anomaly_detector.compute_entropy_score(preds).item()
                metrics_tracker.add_anomaly_score(score, is_adv=(mode=="adv"))
                
            # Convert back to PIL
            res_img = (processed_tensor[0].cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            res_img_pil = Image.fromarray(res_img)
            # Save rescaled to original dimensions to preserve bounding box compatibility
            res_img_pil = res_img_pil.resize((w, h))
            res_img_pil.save(out_imgs_dir / img_path.name)
        else:
            # Clean mode or no model just copies
            shutil.copy(img_path, out_imgs_dir / img_path.name)
            
        # Copy corresponding label
        label_name = img_path.stem + ".txt"
        label_src = orig_labels_dir / label_name
        if label_src.exists():
            shutil.copy(label_src, out_labels_dir / label_name)
            
    # Write new yaml
    new_yaml_dict = data_dict.copy()
    # Assuming standard structure where setting val to the new images dir works
    new_yaml_dict['val'] = str(out_imgs_dir.absolute())
    
    # Note: Sometimes train is required by YOLO validator schema
    new_yaml_path = base_dir / f"{output_dir_name}.yaml"
    with open(new_yaml_path, 'w') as f:
        yaml.safe_dump(new_yaml_dict, f)
        
    return str(new_yaml_path)


def run_evaluation(model_path, data_yaml, img_size=640):
    model = YOLO(model_path)
    metrics = RobustnessMetrics()
    detector = AnomalyDetector(percentile=95)
    
    logger.info("Evaluating CLEAN baseline...")
    clean_val = model.val(data=data_yaml, imgsz=img_size, split='val', plots=False, workers=0)
    metrics.clean_map = clean_val.box.map50
    # Clean stats approximation from val
    logger.info("Clean mAP@0.5: %.4f", metrics.clean_map)
    
    logger.info("Generating Adversarial Dataset...")
    import numpy as np # deferred import
    
    adv_model = YOLO(model_path)
    for p in adv_model.model.parameters():
        p.requires_grad_(True)
    
    adv_yaml = create_modified_dataset(
        data_yaml, "val_adv", mode="adv", 
        model=adv_model, attack_kwargs={"epsilon": 8/255, "iters": 5},
        anomaly_detector=detector, metrics_tracker=metrics
    )
    logger.info("Evaluating ADVERSARIAL...")
    adv_val = model.val(data=adv_yaml, imgsz=img_size, split='val', plots=False, workers=0)
    metrics.adv_map = adv_val.box.map50
    logger.info("Adversarial mAP@0.5: %.4f", metrics.adv_map)
    
    logger.info("Generating Defended Dataset from Adversarial...")
    # Actually we should apply defense on the adversarial images directly.
    # To do this right, we can just construct a "defended" sequence.
    defended_yaml = create_modified_dataset(
        adv_yaml, "val_defended", mode="defended",
        model=model, defense_kwargs={"use_jpeg": True, "use_bit_depth": True, "use_blur": True},
        anomaly_detector=detector, metrics_tracker=metrics
    )
    logger.info("Evaluating DEFENDED...")
    def_val = model.val(data=defended_yaml, imgsz=img_size, split='val', plots=False, workers=0)
    metrics.defended_map = def_val.box.map50
    logger.info("Defended mAP@0.5: %.4f", metrics.defended_map)
    
    logger.info("%s", metrics.summary())
    # Anomaly metrics
    if len(metrics.anomaly_scores) > 0:
        # We can dynamically set threshold using clean scores if we tracked them, 
        # or just use a fixed 95th percentile of all scores
        thresh = np.percentile(metrics.anomaly_scores[:len(metrics.anomaly_scores)//2], 95)
        anom_res = metrics.compute_anomaly_metrics(thresh)
        logger.info("Anomaly Detection:")
        logger.info("  Precision: %.4f", anom_res['Precision'])
        logger.info("  Recall/Detection Rate: %.4f", anom_res['Recall'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.data)
