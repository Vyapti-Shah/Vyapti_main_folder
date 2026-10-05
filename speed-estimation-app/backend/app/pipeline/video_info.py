"""Robust fps/frame-count detection.

Container-reported fps/frame-count (cv2.CAP_PROP_FPS / CAP_PROP_FRAME_COUNT)
is unreliable for variable-frame-rate recordings — e.g. browser
MediaRecorder-produced .webm files, which often have no real average frame
rate and cause ffmpeg/OpenCV to report the stream's millisecond time-base
(1000) as if it were the fps, with frame count derived from that (duration
* 1000), wildly overcounting actual frames. Every per-frame timestamp used
for speed calculation depends on fps, so a wrong fps silently produces
wrong speeds (or, combined with other filters, no usable tracks at all).

We detect that mismatch by actually decoding the video once and comparing
against what the container claims.
"""
import cv2


def probe_video(video_path: str) -> tuple[float, int]:
    """Returns (fps, n_frames), falling back to a real full decode when the
    container's reported values don't match reality."""
    cap = cv2.VideoCapture(video_path)
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

    n = 0
    last_msec = 0.0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        last_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        n += 1
    cap.release()

    duration_s = last_msec / 1000.0 if last_msec > 0 else None
    real_fps = (n - 1) / duration_s if duration_s and n > 1 else (reported_fps or 30.0)

    mismatched = (
        reported_fps <= 0 or reported_fps > 240
        or reported_count <= 0
        or abs(reported_count - n) > max(5, 0.05 * n)
    )
    if mismatched:
        return real_fps, n
    return reported_fps, int(reported_count)
