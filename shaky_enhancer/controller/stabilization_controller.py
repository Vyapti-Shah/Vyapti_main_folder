"""
Controller layer: coordinates the flow between the CLI and the service
layer. Intentionally contains no OpenCV, Torch, or numpy — only routing.
"""

from config.settings import settings
from service.stabilization_service import StabilizationService


class StabilizationController:
    def __init__(self, radius: int = settings.DEFAULT_RADIUS):
        self.service = StabilizationService(radius=radius)

    def stabilize(self, input_path: str, output_path: str) -> None:
        self.service.process_video(input_path, output_path)
