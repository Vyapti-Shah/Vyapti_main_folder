"""Generic conversion utilities between numpy images and PyTorch tensors."""

import cv2
import numpy as np
import torch


def img2tensor(img: np.ndarray, bgr2rgb: bool = True, float32: bool = True) -> torch.Tensor:
    """Converts a numpy image array to a normalized PyTorch tensor."""
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if float32:
        img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img)
    tensor = (tensor - 0.5) / 0.5
    return tensor


def tensor2img(tensor: torch.Tensor, rgb2bgr: bool = True) -> np.ndarray:
    """Converts a PyTorch tensor back to a numpy image array."""
    tensor = tensor.squeeze().float().cpu().clamp_(-1, 1)
    tensor = (tensor + 1) / 2
    img = tensor.numpy().transpose(1, 2, 0)
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return (img * 255.0).round().astype(np.uint8)
