import os
import tempfile
from pathlib import Path

import cv2

from service.ai_detector import AIDetectorService
from service.media_io import MediaIOService
from service.watermarker import WatermarkerService


class DetectionController:
    """Coordinates the deepfake detection pipeline."""

    def __init__(self, model_name: str):
        """
        Initializes the Detection Controller and its dependent services.

        Args:
            model_name (str): The Hugging Face model ID.
        """
        self.detector = AIDetectorService(model_name=model_name)
        self.io = MediaIOService()
        self.watermarker = WatermarkerService()

    def is_video(self, file_path: str) -> bool:
        """
        Determines if the given file path is a video based on its extension.

        Args:
            file_path (str): The path to the file.

        Returns:
            bool: True if the file is a video, False otherwise.
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        return Path(file_path).suffix.lower() in video_extensions

    def process(self, input_path: str, output_path: str) -> None:
        """
        Processes an image or video file.

        Args:
            input_path (str): The path to the input file.
            output_path (str): The path to save the output file.
        """
        if not os.path.exists(input_path):
            print(f"[-] ERROR: Input file not found: {input_path}")
            return

        print(f"\n--- Starting Processing for: {input_path} ---")

        if self.is_video(input_path):
            self._process_video(input_path, output_path)
        else:
            self._process_image(input_path, output_path)

        print(f"\n[+] Processing complete! Output saved to: {output_path}")

    def _process_image(self, input_path: str, output_path: str) -> None:
        """Process a single image."""
        print("[Stage 1] Loading image...")
        img = cv2.imread(input_path)
        if img is None:
            print("[-] ERROR: Could not read image.")
            return

        print("[Stage 2] Running AI Detection...")
        label = self.detector.analyze_frame(img)
        print(f"  -> Prediction: {label}")

        print("[Stage 3] Adding Watermark and Saving...")
        img = self.watermarker.add_watermark(img, label)
        cv2.imwrite(output_path, img)

    def _process_video(self, input_path: str, output_path: str) -> None:
        """Process a video frame-by-frame."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            frames_dir = tmp_dir_path / "frames"
            frames_dir.mkdir()

            print("\n[Stage 1] Extracting frames from video...")
            fps = self.io.extract_frames(input_path, str(frames_dir))
            
            processed_dir = tmp_dir_path / "processed"
            processed_dir.mkdir()

            frame_files = sorted(list(frames_dir.glob("*.png")))
            total_frames = len(frame_files)

            print(
                f"\n[Stage 2] Analyzing {total_frames} frames "
                "(Frame-by-Frame mode)..."
            )

            for i, frame_path in enumerate(frame_files):
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                
                # Analyze frame
                label = self.detector.analyze_frame(img)
                
                # Watermark frame
                img = self.watermarker.add_watermark(img, label)
                
                # Save processed frame
                cv2.imwrite(str(processed_dir / f"{i:06d}.png"), img)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{total_frames} frames")

            print("\n[Stage 3] Recompiling video...")
            self.io.compile_video(str(processed_dir), output_path, fps)
