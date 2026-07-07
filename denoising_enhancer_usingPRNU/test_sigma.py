import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'prnu_repo'))
from prnu.functions import noise_extract

cap = cv2.VideoCapture("inputs/noisy_clip.mp4")
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
orig_var = cv2.Laplacian(gray, cv2.CV_64F).var()
print(f"Original Variance: {orig_var:.2f}")

for sigma in [5.0, 15.0, 30.0, 50.0, 100.0]:
    noise = noise_extract(frame, levels=4, sigma=sigma)
    denoised = frame.astype(np.float32) - noise
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    denoised_var = cv2.Laplacian(gray_denoised, cv2.CV_64F).var()
    print(f"Sigma: {sigma}, Variance: {denoised_var:.2f}, Noise Reduction: {(orig_var - denoised_var) / orig_var * 100:.2f}%")
