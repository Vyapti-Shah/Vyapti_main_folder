import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def calculate_brightness(frame: np.ndarray) -> float:
    """Calculates the mean brightness of a BGR image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def calculate_contrast(frame: np.ndarray) -> float:
    """Calculates the contrast (standard deviation of pixel intensities) of a BGR image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def get_frame_metrics(frame: np.ndarray) -> dict:
    """Returns a dictionary of metrics for the given frame."""
    return {
        "brightness": calculate_brightness(frame),
        "contrast": calculate_contrast(frame)
    }


class VideoIO:
    """Handles video frame extraction and compilation."""

    @staticmethod
    def extract_frames(video_path: Path, output_dir: Path) -> int:
        """Extracts all frames from a video to a directory. Returns the number of frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(output_dir / f"{idx:06d}.png"), frame)
            idx += 1
        cap.release()
        return idx

    @staticmethod
    def compile_video(frames_dir: Path, output_path: Path, original_video_path: Path) -> None:
        """Compiles a sequence of frames back into a video."""
        cap = cv2.VideoCapture(str(original_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 30.0
        cap.release()

        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            logger.warning("No frames found to compile.")
            return

        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_files[0]}")
            
        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame_file in tqdm(frame_files, desc="Compiling video", unit="frame"):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
        out.release()


class FrameEnhancer:
    """Handles algorithmic low-light frame enhancement using OpenCV."""

    def __init__(self, watermark: str = "ENHANCED (LOW LIGHT)"):
        self.watermark = watermark

    def enhance_frame(self, img: np.ndarray) -> np.ndarray:
        """Applies CLAHE and global contrast boost."""
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 2. Global Contrast Boost
        enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=1.15, beta=15)

        return enhanced_img

    def add_watermark(self, img: np.ndarray) -> np.ndarray:
        """Adds a watermark to the image to indicate it was enhanced."""
        if self.watermark:
            cv2.putText(
                img, self.watermark, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
            )
        return img


class VideoPipelineCoordinator:
    """Coordinates the video enhancement pipeline."""

    def __init__(self):
        self.frame_enhancer = FrameEnhancer()
        self.video_io = VideoIO()

    def process_video(self, input_video_path: str, output_video_path: str) -> None:
        """Runs the enhancement pipeline on a video."""
        input_path = Path(input_video_path)
        output_path = Path(output_video_path)

        if not input_path.exists():
            logger.error(f"Input video not found: {input_path}")
            sys.exit(1)

        logger.info(f"--- Starting Processing for: {input_path.name} ---")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            extracted_frames_dir = tmp_dir_path / "extracted"
            extracted_frames_dir.mkdir()

            # STAGE 1: Extract frames
            logger.info("[Stage 1] Extracting frames...")
            frames_extracted = self.video_io.extract_frames(input_path, extracted_frames_dir)
            logger.info(f"Extracted {frames_extracted} frames.")

            # STAGE 2: Frame-by-frame Enhancement
            final_frames_dir = tmp_dir_path / "final"
            final_frames_dir.mkdir()

            frame_files = sorted(list(extracted_frames_dir.glob("*.png")))
            if not frame_files:
                logger.error("No frames were extracted. Aborting.")
                sys.exit(1)

            logger.info(f"[Stage 2] Applying Low Light Enhancements on {len(frame_files)} frames...")

            total_orig_metrics = {"brightness": 0.0, "contrast": 0.0}
            total_enh_metrics = {"brightness": 0.0, "contrast": 0.0}
            num_frames = len(frame_files)

            for frame_path in tqdm(frame_files, desc="Processing frames", unit="frame"):
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                orig_metrics = get_frame_metrics(img)
                total_orig_metrics["brightness"] += orig_metrics["brightness"]
                total_orig_metrics["contrast"] += orig_metrics["contrast"]

                # Apply low-light enhancement
                img = self.frame_enhancer.enhance_frame(img)

                enh_metrics = get_frame_metrics(img)
                total_enh_metrics["brightness"] += enh_metrics["brightness"]
                total_enh_metrics["contrast"] += enh_metrics["contrast"]

                # Add watermark
                img = self.frame_enhancer.add_watermark(img)

                # Save final frame
                out_file = final_frames_dir / frame_path.name
                cv2.imwrite(str(out_file), img)

            if num_frames > 0:
                avg_orig_brightness = total_orig_metrics["brightness"] / num_frames
                avg_orig_contrast = total_orig_metrics["contrast"] / num_frames
                avg_enh_brightness = total_enh_metrics["brightness"] / num_frames
                avg_enh_contrast = total_enh_metrics["contrast"] / num_frames
                
                logger.info("--- Enhancement Metrics Summary ---")
                logger.info(f"Average Brightness: {avg_orig_brightness:.2f} -> {avg_enh_brightness:.2f} (Gain: {avg_enh_brightness - avg_orig_brightness:+.2f})")
                logger.info(f"Average Contrast:   {avg_orig_contrast:.2f} -> {avg_enh_contrast:.2f} (Gain: {avg_enh_contrast - avg_orig_contrast:+.2f})")
                logger.info("-----------------------------------")

            # STAGE 3: Video Reassembly
            logger.info("[Stage 3] Recompiling enhanced video...")
            self.video_io.compile_video(final_frames_dir, output_path, input_path)

        logger.info(f"[+] Enhancement complete! Output saved to: {output_path}")


def main() -> None:
    """Parses arguments and runs the pipeline."""
    parser = argparse.ArgumentParser(description="Low Light Video Enhancement")
    parser.add_argument("-i", "--input", required=True, help="Path to input video")
    parser.add_argument("-o", "--output", required=True, help="Path to save video")

    args = parser.parse_args()

    pipeline = VideoPipelineCoordinator()
    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
