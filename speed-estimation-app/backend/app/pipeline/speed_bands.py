"""Shared speed -> color-band classification, used both for the on-video
badge (annotate.py, OpenCV/BGR) and the API report (frontend renders the
matching CSS color from the band name)."""
from app.config import SPEED_BAND_SLOW_KMH, SPEED_BAND_FAST_KMH

# OpenCV draws in BGR, not RGB.
BAND_COLOR_BGR = {
    "slow": (90, 180, 60),       # green
    "moderate": (0, 165, 255),   # amber
    "fast": (50, 50, 220),       # red
}


def speed_band(speed_kmh: float) -> str:
    if speed_kmh < SPEED_BAND_SLOW_KMH:
        return "slow"
    if speed_kmh < SPEED_BAND_FAST_KMH:
        return "moderate"
    return "fast"
