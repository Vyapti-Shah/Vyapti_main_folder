import threading
import traceback
import uuid
from pathlib import Path
from typing import Dict

import numpy as np

from app.config import (
    OUTPUT_DIR,
    DEFAULT_BBOX_JITTER_PX,
    DEFAULT_CALIB_JITTER_PX,
    DEFAULT_TIMESTAMP_JITTER_S,
    STATIONARY_SPEED_THRESHOLD_KMH,
    MIN_RELATIVE_MOTION,
    MOTION_WARMUP_FRAMES,
    MOTION_MIN_AREA_PX,
)
from app.pipeline.calibration import compute_homography
from app.pipeline.sam2_video_tracker import run_sam2_tracking
from app.pipeline.tracker import TrackData
from app.pipeline.speed import compute_speed_series, compute_speed_series_auto, reject_outliers_mad
from app.pipeline.uncertainty import estimate_uncertainty, estimate_uncertainty_auto
from app.pipeline.annotate import render_annotated_video
from app.pipeline.report import (
    save_speed_vs_time_chart,
    save_actual_vs_estimated_chart,
    compute_evaluation,
)
from app.schemas import CalibrationRequest, JobStatus, Report, TrackSpeedResult

JOBS: Dict[str, JobStatus] = {}
REPORTS: Dict[str, Report] = {}
_lock = threading.Lock()

MIN_OBS = 6


def _set_status(job_id: str, status: str, progress: float = None, message: str = ""):
    with _lock:
        js = JOBS[job_id]
        js.status = status
        if progress is not None:
            js.progress = progress
        js.message = message


def start_job(video_path: Path, req: CalibrationRequest) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(job_id=job_id, status="queued", progress=0.0)
    t = threading.Thread(target=_run_pipeline, args=(job_id, video_path, req), daemon=True)
    t.start()
    return job_id


def _make_lookup(times, speeds, v_unc):
    times_arr = np.array(times) if times else np.array([0.0])
    speeds_arr = np.array(speeds) if speeds else np.array([v_unc])

    def f(t):
        idx = int(np.argmin(np.abs(times_arr - t)))
        return float(speeds_arr[idx]), v_unc
    return f


def _pixel_motion_ratio(tdata: TrackData) -> float:
    obs = tdata.observations
    if len(obs) < 2:
        return 0.0
    disp = float(np.linalg.norm(obs[-1].ground_px - obs[0].ground_px))
    diagonals = [float(np.hypot(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1])) for o in obs]
    avg_diag = float(np.mean(diagonals)) or 1.0
    return disp / avg_diag


def _run_pipeline(job_id: str, video_path: Path, req: CalibrationRequest):
    try:
        manual = req.calibration_mode == "manual" and req.image_points and req.world_points
        calib = None

        if manual:
            _set_status(job_id, "running", 0.02, "Calibrating ground plane (manual)")
            calib = compute_homography(
                [(p.x, p.y) for p in req.image_points],
                [(p.x, p.y) for p in req.world_points],
            )
        else:
            _set_status(job_id, "running", 0.02,
                        "No calibration points given -- using automatic class-size scale estimation")

        _set_status(job_id, "running", 0.05, "Finding moving objects and tracking with SAM2")

        def progress_cb(frac):
            _set_status(job_id, "running", 0.05 + 0.6 * frac, "Tracking with SAM2 (this is the slow step, especially on CPU)")

        tracks = run_sam2_tracking(
            str(video_path),
            fps_override=req.fps_override,
            progress_cb=progress_cb,
            warmup_frames=MOTION_WARMUP_FRAMES,
            min_area_px=MOTION_MIN_AREA_PX,
        )

        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = req.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if not tracks:
            report = Report(
                job_id=job_id, video_id=req.video_id, fps=fps, n_frames=n_frames,
                calibration_mode="manual" if manual else "auto",
                calibration_rmse_px=calib.rmse_px if manual else None,
                tracks=[], evaluation=None,
            )
            REPORTS[job_id] = report
            out_video_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
            render_annotated_video(str(video_path), str(out_video_path), {}, {}, fps_override=fps)
            _set_status(job_id, "done", 1.0, "Complete — no moving objects detected in this video")
            return

        bbox_jitter = req.bbox_jitter_px or DEFAULT_BBOX_JITTER_PX
        ts_jitter = req.timestamp_jitter_s or DEFAULT_TIMESTAMP_JITTER_S
        if manual:
            calib_jitter = req.calib_jitter_px or DEFAULT_CALIB_JITTER_PX
            calib_jitter = float(np.sqrt(calib_jitter**2 + calib.rmse_px**2))

        _set_status(job_id, "running", 0.68, "Computing speed and uncertainty")

        track_results: list[TrackSpeedResult] = []
        speed_lookup = {}
        n_dropped_stationary = 0

        for tid, tdata in tracks.items():
            if len(tdata.observations) < MIN_OBS:
                continue

            if manual:
                ss = compute_speed_series(tdata, calib)
                known_class = None
            else:
                ss, mean_rel_std, known_class = compute_speed_series_auto(tdata)

            inliers = reject_outliers_mad(ss.speed_kmh)
            time_series = ss.time_s[inliers].tolist()
            speed_series = ss.speed_kmh[inliers].tolist()
            if not speed_series:
                time_series = ss.time_s.tolist()
                speed_series = ss.speed_kmh.tolist()

            point_estimate = float(np.mean(speed_series)) if speed_series else 0.0

            if manual:
                px = np.array([o.ground_px for o in tdata.observations])
                t_arr = np.array([o.time_s for o in tdata.observations])
                unc = estimate_uncertainty(
                    px, t_arr, calib,
                    bbox_jitter_px=bbox_jitter,
                    calib_jitter_px=calib_jitter,
                    timestamp_jitter_s=ts_jitter,
                )
                total_d = float(np.linalg.norm(ss.world_xy[-1] - ss.world_xy[0]))
            else:
                bboxes = np.array([o.bbox for o in tdata.observations])
                typical_pixel_dim = float(np.median(bboxes[:, 2] - bboxes[:, 0]))
                t_arr = np.array([o.time_s for o in tdata.observations])
                typical_dt = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else 1.0 / fps
                unc = estimate_uncertainty_auto(
                    point_estimate_kmh=point_estimate,
                    mean_relative_size_std=mean_rel_std,
                    bbox_jitter_px=bbox_jitter,
                    typical_pixel_dim=typical_pixel_dim,
                    timestamp_jitter_s=ts_jitter,
                    typical_dt_s=typical_dt,
                )
                total_d = float(ss.cum_distance_m[-1] - ss.cum_distance_m[0])

            motion_ratio = _pixel_motion_ratio(tdata)
            speed_says_moving = unc.ci95_high_kmh >= STATIONARY_SPEED_THRESHOLD_KMH
            pixels_say_moving = motion_ratio >= MIN_RELATIVE_MOTION
            if not (speed_says_moving and pixels_say_moving):
                n_dropped_stationary += 1
                continue

            mean_conf = float(np.mean([o.conf for o in tdata.observations]))
            start_t = tdata.observations[0].time_s
            end_t = tdata.observations[-1].time_s

            tr = TrackSpeedResult(
                track_id=tid,
                class_name=tdata.class_name,
                start_time_s=start_t,
                end_time_s=end_t,
                distance_m=total_d,
                elapsed_s=end_t - start_t,
                mean_speed_kmh=unc.mean_speed_kmh,
                speed_std_kmh=unc.std_kmh,
                ci95_low_kmh=unc.ci95_low_kmh,
                ci95_high_kmh=unc.ci95_high_kmh,
                n_samples=unc.n_samples,
                tracking_confidence=mean_conf,
                speed_series_kmh=speed_series,
                time_series_s=time_series,
                known_class_size=known_class,
            )
            track_results.append(tr)
            speed_lookup[tid] = _make_lookup(time_series, speed_series, unc.std_kmh)

        _set_status(job_id, "running", 0.85, "Rendering annotated video")
        out_video_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
        moving_track_ids = {tr.track_id for tr in track_results}
        moving_tracks = {tid: tdata for tid, tdata in tracks.items() if tid in moving_track_ids}
        render_annotated_video(str(video_path), str(out_video_path), moving_tracks, speed_lookup, fps_override=fps)

        _set_status(job_id, "running", 0.93, "Building report and charts")
        charts_dir = OUTPUT_DIR / job_id
        charts_dir.mkdir(parents=True, exist_ok=True)
        for tr in track_results:
            if len(tr.time_series_s) >= 2:
                save_speed_vs_time_chart(tr, charts_dir / f"track_{tr.track_id}_speed_vs_time.png")

        evaluation = compute_evaluation(track_results, req.known_speeds_kmh)
        if evaluation:
            save_actual_vs_estimated_chart(
                evaluation["actual_kmh"], evaluation["estimated_kmh"],
                charts_dir / "actual_vs_estimated.png"
            )

        report = Report(
            job_id=job_id,
            video_id=req.video_id,
            fps=fps,
            n_frames=n_frames,
            calibration_mode="manual" if manual else "auto",
            calibration_rmse_px=calib.rmse_px if manual else None,
            tracks=track_results,
            evaluation=evaluation,
        )
        REPORTS[job_id] = report

        _set_status(
            job_id, "done", 1.0,
            f"Complete — {len(track_results)} moving object(s) reported, "
            f"{n_dropped_stationary} stationary candidate(s) excluded"
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _set_status(job_id, "error", None, f"{type(exc).__name__}: {exc}")