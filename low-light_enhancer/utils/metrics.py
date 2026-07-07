"""Utility functions for calculating image quality metrics."""

import cv2
import numpy as np


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
