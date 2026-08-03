import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    print("[-] ERROR: 'tqdm' is not installed. Run: pip install tqdm")
    import sys
    sys.exit(1)

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
        label, conf, matrix = self.detector.analyze_frame(img)
        print(f"  -> Prediction: {label} (Accuracy/Confidence: {conf*100:.2f}%)")
        print(f"  -> Confidence Matrix: {matrix}")

        print("[Stage 3] Adding Watermark and Saving...")
        img = self.watermarker.add_watermark(img, label)
        cv2.imwrite(output_path, img)

    def _process_video(self, input_path: str, output_path: str) -> None:
        """Process a video by analyzing a sequence of frames, then watermarking."""
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
            
            if total_frames == 0:
                print("[-] ERROR: No frames could be extracted from the video.")
                return

            print(f"\n[Stage 2] Analyzing video sequence ({total_frames} total frames)...")
            
            # Deepfake LSTMs rely on smooth motion/micro-flickering. 
            # We must sample 20 CONSECUTIVE frames, not spread them out like a slideshow.
            num_samples = min(20, total_frames)
            start_idx = max(0, (total_frames - num_samples) // 2)  # Take from the middle of the video
            indices = range(start_idx, start_idx + num_samples)
            sampled_files = [frame_files[i] for i in indices]
            
            sequence_frames = []
            for f_path in sampled_files:
                img = cv2.imread(str(f_path))
                if img is not None:
                    sequence_frames.append(img)
                    
            print(f"  -> Sending {len(sequence_frames)} frames to the AI for sequence analysis...")
            
            # Analyze the entire sequence at once to get a single label for the video
            final_label, conf, matrix = self.detector.analyze_sequence(sequence_frames)
            
            print(f"\n[+] Video Analysis Complete!")
            print(f"  -> Final Prediction: {final_label} (Confidence: {conf*100:.2f}%)")
            
            print(f"\n[Stage 3] Applying '{final_label}' watermark to all {total_frames} frames...")
            pbar = tqdm(enumerate(frame_files), total=total_frames, desc="Watermarking", unit="frame")
            for i, frame_path in pbar:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                
                # Watermark frame with the single final label
                img = self.watermarker.add_watermark(img, final_label)
                
                # Save processed frame
                cv2.imwrite(str(processed_dir / f"{i:06d}.png"), img)
                
            print("\n[Stage 4] Recompiling video...")
            self.io.compile_video(str(processed_dir), output_path, fps)
