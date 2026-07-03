"""
CLI entrypoint for RAFT-based video stabilization.

Usage:
    python main.py -i input.mp4 -o output.mp4 --radius 60
"""

import argparse

from service.settings import settings
from controller.stabilization_controller import StabilizationController


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAFT dense optical flow video stabilization")
    parser.add_argument("-i", "--input", required=True, help="Path to the shaky input video")
    parser.add_argument("-o", "--output", required=True, help="Path to save the stabilized output video")
    parser.add_argument(
        "--radius",
        type=int,
        default=settings.DEFAULT_RADIUS,
        help="Smoothing radius (higher = smoother/more drone-like, default: %(default)s)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    controller = StabilizationController(radius=args.radius)
    controller.stabilize(args.input, args.output)


if __name__ == "__main__":
    main()
