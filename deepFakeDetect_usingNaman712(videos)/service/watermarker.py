import cv2
import numpy as np


class WatermarkerService:
    """Handles adding watermarks to frames."""

    @staticmethod
    def add_watermark(img: np.ndarray, label: str) -> np.ndarray:
        """
        Overlays a watermark indicating if the media is REAL or AI GENERATED.

        Args:
            img (np.ndarray): The image to watermark.
            label (str): The label to display ("AI GENERATED" or "REAL").

        Returns:
            np.ndarray: The watermarked image.
        """
        # Red for AI, Green for Real
        color = (0, 0, 255) if label == "AI GENERATED" else (0, 255, 0)
        text = "REAL DETECTED" if label == "REAL" else label

        # Setup text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size to draw a background rectangle for better visibility
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Position: top-left corner
        x, y = 10, 30
        
        # Draw background rectangle
        cv2.rectangle(
            img,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
        
        return img
