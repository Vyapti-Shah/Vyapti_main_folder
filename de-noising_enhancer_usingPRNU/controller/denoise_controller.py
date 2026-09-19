# controller/denoise_controller.py
from service.pipeline_service import PipelineService

class DenoiseController:
    def __init__(self):
        self.pipeline = PipelineService()

    def run(self, input_path, output_path):
        self.pipeline.process(input_path, output_path)
