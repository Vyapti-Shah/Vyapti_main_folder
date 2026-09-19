"""
ResNeXt50 + LSTM Architecture for Deepfake Video Detection
Extracted and cleaned from https://github.com/namandhakad712/Deepfake-detector
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNeXtLSTMDetector(nn.Module):
    """
    Spatio-Temporal Deepfake Detector.
    Uses ResNeXt50 for spatial feature extraction per frame,
    and an LSTM for temporal consistency analysis across frames.
    """
    def __init__(
        self,
        num_classes=2,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False
    ):
        super(ResNeXtLSTMDetector, self).__init__()
        
        # Load the base ResNeXt50 model (without pretrained weights)
        # We strip off the final AveragePool and Classifier layers
        base_model = models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        
        # Temporal analysis block
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass for a sequence of frames.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_length, channels, height, width)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_length, c, h, w = x.shape
        
        # Flatten batch and sequence to pass through CNN
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Extract spatial features
        fmap = self.model(x)
        x = self.avgpool(fmap)
        
        # Reshape for LSTM: (seq_length, batch_size, features)
        x = x.view(batch_size, seq_length, 2048)
        
        # We need to swap dimensions for standard PyTorch LSTM (seq_len, batch, input_size)
        # But wait, original code used batch_first=False default, which expects (seq, batch, feature)
        # Original code did: x_lstm, _ = self.lstm(x, None) and then x_lstm[:, -1, :] 
        # In PyTorch, if batch_first=False, x should be (seq_len, batch, hidden).
        # Let's adjust it to be safe or keep exact original logic.
        
        # The original code provided (x_lstm[:, -1, :]) implies it treated dimension 1 as sequence.
        # Let's explicitly set batch_first=True to avoid bugs.
        # However, to avoid altering their class signature, we'll manually transpose.
        x = x.transpose(0, 1)  # now (seq_length, batch_size, 2048)
        x_lstm, _ = self.lstm(x, None)
        
        # x_lstm is (seq_length, batch_size, hidden_dim)
        # We take the output of the last timestep
        last_timestep = x_lstm[-1, :, :]  # (batch_size, hidden_dim)
        
        # Final classification
        out = self.dp(self.linear1(last_timestep))
        return out
