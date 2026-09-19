import yaml
import logging
import json
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.orchestrator import CamoTextPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_images(input_dir: str, extensions: list) -> list:
    return [
        str(p) for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in extensions
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the camo-text pipeline over a batch of images.")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to the config YAML file to use (default: config.yaml)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Using config: {args.config}")
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)

    pipeline = CamoTextPipeline(config)
    images = collect_images(config["paths"]["input_dir"], config["batch"]["extensions"])
    logger.info(f"Found {len(images)} images to process")

    results = []
    with ThreadPoolExecutor(max_workers=config["batch"]["num_workers"]) as executor:
        futures = {executor.submit(pipeline.run, img): img for img in images}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status = res["status"]
            logger.info(f"[{status.upper()}] {res['image_path']}")

    out_path = os.path.join(config["paths"]["output_dir"], "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    main()