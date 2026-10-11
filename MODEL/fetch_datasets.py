"""
fetch_datasets.py — Download the public datasets used to train the KAAL detector.

Sources (all public, no account needed):

  * ``uav_tank``  Simuletic/UAV-Aerial-View-Battle-Tank-Detection-Dataset
                  (Hugging Face, ~170 MB, synthetic drone-view tanks, YOLO format)
  * ``aircraft``  a2015003713/military-aircraft-detection-dataset
                  (Hugging Face, 25k images / 8.6 GB in full). Only a
                  rarity-weighted subset is fetched by default — the COCO
                  annotation files are downloaded first and used to pick images.
  * VisDrone2019-DET is a GitHub release asset; fetch it with the gh CLI:

        gh release download v0.0.0 --repo ultralytics/assets \\
            --pattern 'VisDrone2019-DET-*.zip' --dir MODEL/Datasets/visdrone

Usage:
    python -m MODEL.fetch_datasets uav_tank
    python -m MODEL.fetch_datasets aircraft --train 3500 --val 400 --test 400

Requires ``huggingface_hub`` (``pip install huggingface_hub``). Anonymous Hub
requests are rate-limited; the downloader backs off and resumes, and files that
already exist are skipped, so it is safe to re-run.
"""

import argparse
import json
import logging
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent / "Datasets"
UAV_TANK_REPO = "Simuletic/UAV-Aerial-View-Battle-Tank-Detection-Dataset"
AIRCRAFT_REPO = "a2015003713/military-aircraft-detection-dataset"


def fetch_uav_tank(out_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(UAV_TANK_REPO, repo_type="dataset", local_dir=str(out_dir))
    logger.info("UAV tank dataset ready at %s", out_dir)


def select_aircraft_images(coco: Dict, n: int, seed: int) -> List[str]:
    """
    Pick ``n`` image file names, favouring rare aircraft types.

    Weighted sampling without replacement (Efraimidis–Spirakis): each image's
    weight is 1 / frequency of its rarest category, so rotorcraft and rare
    types are not drowned out by the most common fighters.
    """
    freq = Counter(a["category_id"] for a in coco["annotations"])
    rarest: Dict[int, float] = {}
    for a in coco["annotations"]:
        w = 1.0 / freq[a["category_id"]]
        rarest[a["image_id"]] = max(rarest.get(a["image_id"], 0.0), w)

    rng = random.Random(seed)
    keyed = [
        (rng.random() ** (1.0 / rarest[img["id"]]), img["file_name"])
        for img in coco["images"]
        if img["id"] in rarest
    ]
    keyed.sort(reverse=True)
    return [name for _, name in keyed[:n]]


def _download_one(repo: str, filename: str, out_dir: Path, retries: int = 8) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    if (out_dir / filename).exists():
        return True
    delay = 5.0
    for _ in range(retries):
        try:
            hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(out_dir))
            return True
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                logger.warning("Giving up on %s (%s)", filename, exc)
                return False
            time.sleep(delay)
            delay = min(delay * 2, 300.0)
        except OSError as exc:  # transient network / disk errors
            logger.warning("Retrying %s after %s", filename, exc)
            time.sleep(delay)
    return False


def fetch_aircraft(out_dir: Path, counts: Dict[str, int], seed: int, workers: int) -> None:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    wanted: List[str] = []
    for split, n in counts.items():
        ann = f"annotations/coco/{split}.json"
        hf_hub_download(AIRCRAFT_REPO, ann, repo_type="dataset", local_dir=str(out_dir))
        coco = json.loads((out_dir / ann).read_text())
        names = select_aircraft_images(coco, n, seed)
        logger.info("%s: selected %d / %d images", split, len(names), len(coco["images"]))
        wanted += [f"images/{name[0]}/{name}" for name in names]

    (out_dir / "subset.json").write_text(json.dumps(sorted(wanted), indent=0))
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, AIRCRAFT_REPO, f, out_dir) for f in wanted]
        for fut in as_completed(futures):
            if fut.result():
                done += 1
            else:
                failed += 1
            if (done + failed) % 200 == 0:
                logger.info("aircraft images: %d done, %d failed / %d", done, failed, len(wanted))
    logger.info("aircraft subset ready: %d images (%d failed) at %s", done, failed, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["uav_tank", "aircraft"])
    parser.add_argument("--out", default=None, help="Output dir (default: MODEL/Datasets/<name>)")
    parser.add_argument("--train", type=int, default=3500)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset == "uav_tank":
        fetch_uav_tank(Path(args.out or DATASETS_DIR / "simuletic_uav_tank"))
    else:
        counts = {"train": args.train, "val": args.val, "test": args.test}
        fetch_aircraft(
            Path(args.out or DATASETS_DIR / "military_aircraft"), counts, args.seed, args.workers
        )


if __name__ == "__main__":
    main()
