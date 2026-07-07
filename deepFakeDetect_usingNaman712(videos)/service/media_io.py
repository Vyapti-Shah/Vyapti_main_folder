import os
from pathlib import Path

import cv2


class MediaIOService:
    """Handles image and video reading, writing, and frame extraction."""

    @staticmethod
    def extract_frames(video_path: str, output_dir: str) -> float:
        """
        Extracts all frames from a video and returns the FPS.

        Args:
            video_path (str): The path to the input video.
            output_dir (str): The directory to save the extracted frames.

        Returns:
            float: The frames per second of the video.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(output_dir, f"{idx:06d}.png"), frame)
            idx += 1
        cap.release()
        return fps

    @staticmethod
    def compile_video(frames_dir: str, output_path: str, fps: float) -> None:
        """
        Compiles a sequence of frames back into a video.

        Args:
            frames_dir (str): The directory containing the frames.
            output_path (str): The path to save the output video.
            fps (float): The frames per second for the output video.
        """
        frame_files = sorted(Path(frames_dir).glob("*.png"))
        if not frame_files:
            return

        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            return

        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
        out.release()
