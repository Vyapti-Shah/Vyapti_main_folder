# service/video_service.py
import cv2
import sys

class VideoService:
    def __init__(self):
        self.setup_opencl()

    def setup_opencl(self):
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            device_name = cv2.ocl.Device.getDefault().name()
            print(f"[INFO] Using device: {device_name} (GPU/OpenCL)")
        else:
            print(f"[INFO] Using device: CPU (Hardware Acceleration not found)")

    def open_video(self, input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {input_path}")
            sys.exit(1)
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return cap, width, height, fps, total_frames

    def create_writer(self, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def read_frame(self, cap):
        ret, frame = cap.read()
        return ret, frame

    def write_frame(self, writer, frame):
        # Always write standard CPU numpy arrays
        if isinstance(frame, cv2.UMat):
            writer.write(frame.get())
        else:
            writer.write(frame)
