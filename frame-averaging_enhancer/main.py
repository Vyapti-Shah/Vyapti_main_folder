"""Entry point for the video thumbnail enhancement pipeline."""

import argparse
import logging

from controller.video_controller import VideoController


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Full Video Thumbnail Enhancement Pipeline")
    parser.add_argument("-i", "--i", required=True, help="Input video file path")
    parser.add_argument(
        "-o", "--output", default="pipeline_output", help="Output directory for graphs and thumbnails"
    )
    parser.add_argument(
        "-w", "--window", type=int, default=5, help="Number of frames to average (default: 5)"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    controller = VideoController()
    controller.run(args.i, args.output, args.window)


if __name__ == "__main__":
    main()
