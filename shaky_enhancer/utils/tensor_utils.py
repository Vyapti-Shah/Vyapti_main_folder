"""Conversions between OpenCV frames (BGR numpy) and RAFT input tensors."""

import numpy as np
import torch


def frames_to_tensors(
    frame1: np.ndarray,
    frame2: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a pair of BGR frames into normalized, batched RAFT input tensors."""
    return _frame_to_tensor(frame1, device), _frame_to_tensor(frame2, device)


def _frame_to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    rgb = frame[:, :, ::-1].copy()  # BGR -> RGB
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = (tensor * 2) - 1.0  # scale to [-1, 1]
    return tensor.unsqueeze(0).to(device)


def pad_to_multiple(
    img1: torch.Tensor,
    img2: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RAFT requires H and W divisible by `multiple`; pad with edge-replication."""
    h, w = img1.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h or pad_w:
        img1 = torch.nn.functional.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
        img2 = torch.nn.functional.pad(img2, (0, pad_w, 0, pad_h), mode="replicate")

    return img1, img2


def unpad_flow(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Strip the padding added by pad_to_multiple() back off the flow output."""
    return flow[:, :height, :width]
