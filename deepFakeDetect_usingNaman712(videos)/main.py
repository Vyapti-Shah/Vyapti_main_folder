import argparse

from controller.detection_controller import DetectionController


def main() -> None:
    """Parses CLI arguments and runs the detection pipeline."""
    parser = argparse.ArgumentParser(
        description="AI/Deepfake Media Detection Pass"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input image or video"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save watermarked image or video"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnext-lstm-untrained", 
        help="Hugging Face model ID or custom untrained architecture to use."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum confidence (0.0 to 1.0) required to flag a frame as AI GENERATED. Default is 0.70."
    )

    args = parser.parse_args()

    controller = DetectionController(model_name=args.model)
    # The detector is inside the controller, so we must set the threshold.
    # Note: DetectionController's init doesn't take threshold yet, but we can set it directly:
    controller.detector.threshold = args.threshold
    
    controller.process(args.input, args.output)


if __name__ == "__main__":
    main()
