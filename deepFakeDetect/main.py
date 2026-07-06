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
        default="prithivMLmods/Deep-Fake-Detector-Model", 
        help="Hugging Face model ID to use for image classification."
    )

    args = parser.parse_args()

    controller = DetectionController(model_name=args.model)
    controller.process(args.input, args.output)


if __name__ == "__main__":
    main()
