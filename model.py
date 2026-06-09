"""Skeleton-to-semaphore recognition model.

Architecture:
  - 1D-CNN temporal compressor (learns local motion, projects to d_model)
  - Transformer encoder with positional encoding (dependencies across time)
  - Classification head for whole-clip labels (matches current data)

Built to swap the head later to CTC or autoregressive decoder for continuous
transcription once we're signing cursively.
"""
import math
import torch
import torch.nn as nn
from features import FEATURE_DIM


class TemporalCompressor(nn.Module):
    """1D conv stack: downsamples T -> T' and projects FEATURE_DIM -> hidden."""
    def __init__(self, in_dim=FEATURE_DIM, hidden=128, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, stride=stride, padding=2)
        # GroupNorm is batch-size independent - safer than BatchNorm for the
        # tiny, variable-size batches we train on locally.
        self.norm1 = nn.GroupNorm(8, hidden)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, D] -> [B, D, T] for conv1d
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x.transpose(1, 2)  # [B, T', hidden]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, so the transformer can see frame order.
    Without this the encoder is permutation-invariant over time."""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Small transformer encoder. Input is already at d_model (from the
    compressor), so there is no feature embedding here - only positional encoding."""
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T', d_model]
        return self.encoder(self.pos(x))


class ClipClassifier(nn.Module):
    """Whole-clip label classifier: pool over time, predict one label.
    Swap to CTC or an autoregressive decoder for continuous transcription later."""
    def __init__(self, d_model=128, num_classes=26):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, T', d_model] -> pool over time -> [B, d_model]
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x)  # [B, num_classes]


class SemaphoreRecognizer(nn.Module):
    """Full pipeline: compress -> encode -> classify. Input is normalized features."""
    def __init__(self, num_classes=26, d_model=128):
        super().__init__()
        self.compress = TemporalCompressor(in_dim=FEATURE_DIM, hidden=d_model)
        self.encode = TransformerEncoder(d_model=d_model, nhead=4, num_layers=2)
        self.classify = ClipClassifier(d_model=d_model, num_classes=num_classes)

    def forward(self, x):
        # x: [B, T, FEATURE_DIM] (already normalized)
        x = self.compress(x)
        x = self.encode(x)
        return self.classify(x)
