"""
Contract that every optical-flow backend must satisfy.

Keeping this abstract means the pipeline (StabilizationService /
RaftService) never needs to change if RAFT is swapped for GMFlow,
FlowFormer, or any future model — only a new class implementing this
interface needs to be written.
"""

from abc import ABC, abstractmethod

import torch


class OpticalFlowEstimator(ABC):
    """Base class for dense optical-flow inference backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Must be called before estimate()."""

    @abstractmethod
    def estimate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Run inference on two preprocessed, batched image tensors.

        Args:
            img1: Tensor of shape [1, 3, H, W], normalized to [-1, 1].
            img2: Tensor of shape [1, 3, H, W], normalized to [-1, 1].

        Returns:
            Dense flow tensor of shape [2, H, W] (u, v channels).
        """
