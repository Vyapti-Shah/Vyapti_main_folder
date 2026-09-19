from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.schemas import TrackSpeedResult


def save_speed_vs_time_chart(track_result: TrackSpeedResult, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(track_result.time_series_s, track_result.speed_series_kmh, marker="o", ms=3)
    ax.axhline(track_result.mean_speed_kmh, color="gray", linestyle="--", linewidth=1,
               label=f"mean = {track_result.mean_speed_kmh:.1f} km/h")
    ax.fill_between(
        track_result.time_series_s,
        track_result.ci95_low_kmh,
        track_result.ci95_high_kmh,
        alpha=0.15, color="gray", label="95% CI"
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (km/h)")
    ax.set_title(f"Track #{track_result.track_id} ({track_result.class_name}) — speed vs time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_actual_vs_estimated_chart(
    actual: List[float], estimated: List[float], out_path: Path
):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, estimated, color="tab:blue")
    lims = [0, max(actual + estimated) * 1.1 + 1]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x (ideal)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("actual speed (km/h)")
    ax.set_ylabel("estimated speed (km/h)")
    ax.set_title("Actual vs Estimated Speed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_evaluation(
    tracks: List[TrackSpeedResult], known_speeds_kmh: Optional[Dict[str, float]]
) -> Optional[dict]:
    if not known_speeds_kmh:
        return None

    actual, estimated, ids = [], [], []
    for t in tracks:
        key = str(t.track_id)
        if key in known_speeds_kmh:
            actual.append(float(known_speeds_kmh[key]))
            estimated.append(t.mean_speed_kmh)
            ids.append(t.track_id)

    if not actual:
        return None

    actual_arr = np.array(actual)
    est_arr = np.array(estimated)
    errors = est_arr - actual_arr
    abs_errors = np.abs(errors)
    pct_errors = abs_errors / actual_arr * 100

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(pct_errors))
    bias = float(np.mean(errors))

    coverage = 0.0
    covered = 0
    for t in tracks:
        key = str(t.track_id)
        if key in known_speeds_kmh:
            a = known_speeds_kmh[key]
            if t.ci95_low_kmh <= a <= t.ci95_high_kmh:
                covered += 1
    coverage = covered / len(actual) * 100 if actual else 0.0

    return {
        "track_ids": ids,
        "actual_kmh": actual,
        "estimated_kmh": estimated,
        "MAE_kmh": mae,
        "RMSE_kmh": rmse,
        "MAPE_pct": mape,
        "bias_kmh": bias,
        "ci95_coverage_pct": coverage,
    }
