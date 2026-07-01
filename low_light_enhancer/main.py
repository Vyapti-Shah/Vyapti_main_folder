"""
Forensic Video Enhancement Pipeline — entry point.

Implements:
1. Luminance-Aware AGC (Low Light Enhancement)

Run from the parent directory of video_enhancer/, e.g.:
    python main.py -i input.mp4 -o output.mp4
"""

import argparse
import logging

from video_enhancer.pipeline.controller import PipelineController

logging.basicConfig(

    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low Light Video Enhancement")
    parser.add_argument("-i", "--input", required=True, help="Path to input video")
    parser.add_argument("-o", "--output", required=True, help="Path to save video")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    controller = PipelineController()
    controller.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
