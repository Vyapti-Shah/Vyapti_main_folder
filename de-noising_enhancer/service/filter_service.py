# service/filter_service.py
import cv2
import config

class FilterService:
    def __init__(self):
        self.prev_frame = None

    def calculate_noise(self, frame):
        """
        Calculates Laplacian variance as a proxy for noise/texture.
        Lower variance means smoother/denoised frame.
        """
        # Ensure we are operating on CPU for cvtColor/Laplacian if frame is UMat
        if isinstance(frame, cv2.UMat):
            gray = cv2.cvtColor(frame.get(), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def denoise(self, frame):
        # 1. Extreme Median Filter (on CPU to avoid driver bug)
        if isinstance(frame, cv2.UMat):
            frame = frame.get()
        denoised = cv2.medianBlur(frame, config.MEDIAN_KERNEL)
        
        # 2. Convert to UMat for hardware acceleration of Bilateral
        u_denoised = cv2.UMat(denoised)
        denoised_bilat = cv2.bilateralFilter(u_denoised, config.BILATERAL_DIAMETER, config.SIGMA_COLOR, config.SIGMA_SPACE)
        
        # 3. Temporal Blending
        if self.prev_frame is not None and config.TEMPORAL_WEIGHT > 0.0:
            final_frame = cv2.addWeighted(denoised_bilat, 1.0 - config.TEMPORAL_WEIGHT, self.prev_frame, config.TEMPORAL_WEIGHT, 0)
        else:
            final_frame = denoised_bilat
            
        self.prev_frame = final_frame
        return final_frame

    def sharpen(self, frame):
        # Apply Unsharp Mask
        blurred_final = cv2.GaussianBlur(frame, (5, 5), 0)
        sharpened_frame = cv2.addWeighted(frame, config.SHARPEN_ALPHA, blurred_final, config.SHARPEN_BETA, 0)
        return sharpened_frame
