import cv2
import numpy as np
from PIL import Image, ExifTags

CAMERA_EXIF_TAGS = {"Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"}


def has_camera_exif(image_path: str) -> bool:
    """
    True if the image carries camera-style EXIF metadata. Real camera photos
    usually have this; AI-generated images usually don't. Weak signal only:
    social apps (WhatsApp, Instagram, etc.) commonly strip EXIF from real
    photos too, so absence alone doesn't mean "fake".
    """
    try:
        exif = Image.open(image_path).getexif()
        if not exif:
            return False
        tags = {ExifTags.TAGS.get(k, k) for k in exif.keys()}
        return bool(tags & CAMERA_EXIF_TAGS)
    except Exception:
        return False


def is_unnaturally_smooth(img: Image.Image, threshold: float = 40.0) -> bool:
    """
    Flags unusually low high-frequency detail (the "overly smooth skin" tell
    of many diffusion-generated images), via Laplacian variance on a
    normalized grayscale crop. Weak/heuristic signal: resolution, JPEG
    compression, and genuine soft lighting can also produce low variance.
    """
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold
