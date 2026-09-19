import cv2
import subprocess
import tempfile
import os
import threading
from pathlib import Path
from .detection_service import DeepFakeService


class VideoProcessor:
    """Service for processing videos frame-by-frame and adding watermarks."""

    def __init__(
        self,
        detection_service: DeepFakeService,
        frame_sample_fps: float = 10.0,
        fake_frame_threshold: float = 0.6,
    ):
        self.detection_service = detection_service
        self.frame_sample_fps = frame_sample_fps
        self.fake_frame_threshold = fake_frame_threshold

    def process_video(
        self, video_path: str, output_path: str, sample_fps: float = None
    ) -> tuple[bool, float, int]:
        """
        Sample frames at ~sample_fps, detect+classify faces per sampled frame
        (via DeepFakeService.predict, which does MTCNN cropping internally),
        and aggregate: the video is flagged fake if the fraction of fake
        frames reaches fake_frame_threshold.
        Returns (is_fake_detected, avg_confidence, frames_analyzed)
        """
        if sample_fps is None:
            sample_fps = self.frame_sample_fps

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Convert the target sampling rate (frames per second) into a frame
        # interval relative to the source video's actual fps.
        frame_interval = max(1, round(fps / sample_fps))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Encode via ffmpeg (libx264/yuv420p) so the result plays in browsers -
        # cv2.VideoWriter's mp4v (MPEG-4 Part 2) output is not decodable by
        # Chrome/Firefox/Safari's <video> element.
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # ffmpeg writes continuous progress output to stderr. If nobody reads
        # it, the OS pipe buffer (~64KB) fills up, ffmpeg blocks trying to
        # write to it, stops draining stdin, and our frame-writing loop below
        # deadlocks once the stdin pipe also fills - which only shows up on
        # longer/higher-resolution videos once enough stderr output accumulates.
        # Draining it continuously in a background thread avoids that.
        stderr_lines: list[bytes] = []

        def _drain_stderr():
            for line in iter(ffmpeg_proc.stderr.readline, b""):
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        frame_count = 0
        analyzed_frames = 0
        fake_frames = 0
        confidence_scores = []
        last_is_fake = False

        with tempfile.TemporaryDirectory() as tmpdir:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Classify every nth frame; carry the verdict forward so every
                # frame in between is watermarked consistently instead of flickering.
                if frame_count % frame_interval == 0:
                    temp_frame_path = os.path.join(tmpdir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(temp_frame_path, frame)

                    is_fake, confidence = self.detection_service.predict(
                        temp_frame_path
                    )
                    analyzed_frames += 1
                    if is_fake:
                        fake_frames += 1
                    confidence_scores.append(confidence)
                    last_is_fake = is_fake

                    os.remove(temp_frame_path)

                frame = self._add_watermark_to_frame(frame, not last_is_fake)
                ffmpeg_proc.stdin.write(frame.tobytes())
                frame_count += 1

                # Print progress
                if frame_count % (frame_interval * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing video: {progress:.1f}% ({frame_count}/{total_frames})")

        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        stderr_thread.join(timeout=5)
        if ffmpeg_proc.returncode != 0:
            stderr = b"".join(stderr_lines).decode(errors="ignore")
            raise RuntimeError(f"ffmpeg encoding failed: {stderr}")

        # Temporal aggregation: flag the video if the fraction of fake frames
        # reaches fake_frame_threshold (e.g. 0.6 = 60% of analyzed frames).
        is_fake_detected = (
            analyzed_frames > 0
            and (fake_frames / analyzed_frames) >= self.fake_frame_threshold
        )
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        print(f"Video processing complete: {analyzed_frames} frames analyzed, {fake_frames} fake frames detected")
        return is_fake_detected, avg_confidence, analyzed_frames

    def _add_watermark_to_frame(self, frame, is_real: bool) -> cv2.Mat:
        """Add watermark text to a video frame."""
        text = "REAL" if is_real else "AI Generated"
        color = (0, 255, 0) if is_real else (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = 10, 30

        cv2.rectangle(
            frame,
            (x, y - text_height - 5),
            (x + text_width, y + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )

        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return frame
