import argparse
from service.image_service import DeepFakeService


class ImageController:
    """
    Controller class to handle input arguments and coordinate
    the image processing through the service layer.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="AI Image Detector using EfficientNet-B0"
        )
        self._setup_arguments()

    def _setup_arguments(self):
        self.parser.add_argument(
            "-i", "--input", required=True, help="Path to input image"
        )
        self.parser.add_argument(
            "-o", "--output", required=True, help="Path to save watermarked image"
        )
        self.parser.add_argument(
            "-m", "--model", required=True, help="Path to model weights (.pt/.pth)"
        )
        self.parser.add_argument(
            "--fake_class_index",
            type=int,
            default=1,
            help="Index of the 'Fake/AI' class in model output (default: 1)"
        )

    def run(self):
        """
        Parses arguments and runs the deepfake detection service.
        """
        args = self.parser.parse_args()

        service = DeepFakeService(
            model_path=args.model,
            fake_class_index=args.fake_class_index
        )

        is_fake = service.predict(args.input)

        if is_fake:
            print("Prediction: AI Generated / Fake")
        else:
            print("Prediction: Real")

        service.add_watermark(
            image_path=args.input,
            output_path=args.output,
            is_real=not is_fake
        )
