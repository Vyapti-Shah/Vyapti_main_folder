# service/filter_service.py
import cv2
import config
import numpy as np
import sys
import os

# sys.path append removed since drunet is a standard package now
from drunet.functions import noise_extract_drunet

class FilterService:
    def __init__(self):
        pass

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
        if isinstance(frame, cv2.UMat):
            frame = frame.get()
            
        # 1. Use PRNU noise extraction to estimate the scene via DRUNet
        noise = noise_extract_drunet(frame, levels=int(config.PRNU_LEVELS), sigma=int(config.PRNU_SIGMA))
        
        # 2. Compute denoised image by subtracting the noise residual
        denoised = frame.astype(np.float32) - noise
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        
        return denoised

    def sharpen(self, frame):
        # Disabled legacy sharpening step, returning frame as-is
        return frame
