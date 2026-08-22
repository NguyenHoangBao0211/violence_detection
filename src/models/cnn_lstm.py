from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SpatialEncoder(nn.Module):
    """2D CNN extracts appearance (spatial) features from each frame."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 2048
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 3, H, W) -> (B*T, 2048)
        return self.features(x).flatten(1)


class TemporalAttention(nn.Module):
    """Learns which frames in the clip contribute most to the decision."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(h).squeeze(-1)
        alpha = torch.softmax(energy, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class TemporalHead(nn.Module):
    """BiLSTM + attention over a sequence of spatial feature vectors."""

    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        m = cfg["model"]
        hidden = m["lstm_hidden"]
        bidirectional = m.get("bidirectional", True)
        n_layers = m.get("lstm_layers", 2)
        dropout = m.get("dropout", 0.5)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out = hidden * (2 if bidirectional else 1)
        self.use_attention = m.get("use_attention", True)
        self.attn = TemporalAttention(lstm_out) if self.use_attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, m.get("num_classes", 2)),
        )

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # seq: (B, T, D)
        lstm_out, _ = self.lstm(seq)
        if self.attn is not None:
            pooled, alpha = self.attn(lstm_out)
        else:
            pooled = lstm_out[:, -1]
            alpha = None
        return self.classifier(pooled), alpha


class CNNLSTM(nn.Module):
    """
    Spatial-temporal violence detector.

    Spatial: shared ResNet50 on each frame.
    Temporal: BiLSTM over the frame feature sequence, then attention pooling.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.spatial = SpatialEncoder(
            pretrained=m.get("pretrained", True),
            freeze=m.get("freeze_backbone", False),
        )
        self.temporal = TemporalHead(self.spatial.out_dim, cfg)

    def encode_spatial(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        feat = self.spatial(frames.reshape(b * t, c, h, w))
        return feat.view(b, t, -1)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.temporal(self.encode_spatial(frames))
