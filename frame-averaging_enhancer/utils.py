"""Utility functions and constants for the application."""

import os
import cv2
import numpy as np


GRAPH_FILENAME_TEMPLATE: str = "scene_{index}_graph.png"
CSV_FILENAME_TEMPLATE: str = "scene_{index}_data.csv"
THUMBNAIL_FILENAME_TEMPLATE: str = "scene_{index}_thumbnail.jpg"


def create_directory(path: str) -> None:
    """Creates a directory (including parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def build_path(*parts: str) -> str:
    """Joins path components into a single path string."""
    return os.path.join(*parts)


def convert_gray(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def clip_to_uint8(image: np.ndarray) -> np.ndarray:
    """Clips a float image to the valid [0, 255] range and casts to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)
