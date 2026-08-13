import os
import shutil
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def generate_visualizations(model_path, data_yaml):
    """
    Runs YOLO prediction over the clean, adversarial, and defended validation sets
    to generate bounded outputs with confidence scores, exported to /results/.
    """
    model = YOLO(model_path)
    
    # Extract paths
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
        
    base_dir = Path(data_yaml).parent
    val_path = data_dict.get('val', '')
    
    clean_dir = base_dir / val_path if not os.path.isabs(val_path) else Path(val_path)
    adv_dir = base_dir / "val_adv" / "images"
    defended_dir = base_dir / "val_defended" / "images"
    
    if not adv_dir.exists() or not defended_dir.exists():
        logger.error("Missing dataset directories! Please run evaluate.py first to generate adversarial sets.")
        return
        
    # Setup output directories per user requirements
    results_dir = base_dir / "results"
    
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO predict creates project/name structure automatically.
    # We map 'project' to results_dir and 'name' to target folder
    logger.info("Running inference and visual overlay on CLEAN dataset...")
    # save=True draws bounding boxes and labels. save_conf=True adds confidence scores.
    model.predict(source=str(clean_dir), project=str(results_dir), name="clean", save=True, save_conf=True, save_txt=False, verbose=False)
    
    logger.info("Running inference and visual overlay on ADVERSARIAL dataset...")
    model.predict(source=str(adv_dir), project=str(results_dir), name="adversarial", save=True, save_conf=True, save_txt=False, verbose=False)
    
    logger.info("Running inference and visual overlay on DEFENDED dataset...")
    model.predict(source=str(defended_dir), project=str(results_dir), name="defended", save=True, save_conf=True, save_txt=False, verbose=False)

    logger.info("\n=========================================================")
    logger.info("Visualization complete. Results saved in: %s", results_dir.absolute())
    logger.info("=========================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Visualizations for Adversarial Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset's original data.yaml")
    args = parser.parse_args()
    
    generate_visualizations(args.model, args.data)
