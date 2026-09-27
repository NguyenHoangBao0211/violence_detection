"""Second opinion for the webcam: MoViNet-A0 violence detector (VD-MIL).

best2.pt (ResNet50-BiLSTM) catches fights well but scores some ordinary
webcam scenes as violence, e.g. a person standing still in a bright room
(0.88 on average over 398 such windows). MoViNet-A0 trained on RWF-2000 +
SCVD scores those same windows 0.01 while still reacting to real fights, so
the Camera page uses it to veto best2: see MOVINET_VETO in web/templates/live.html.

Weights: MoViNet-A0 stream backbone (MoViNet-pytorch, MIT) and the VD-MIL head
by Roger Figueroa Quintero, CC BY-NC 4.0 (non-commercial use, credit the author):
https://github.com/roggerfq/violence-detection-mil
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.utils.config import project_root

FRAMES = 8     # VD-MIL windows: 8 frames sampled at ~8 fps (about 1 s)
SIZE = 172

_model = None
_error: str | None = None
_load_lock = threading.Lock()


class _Head(nn.Module):
    """VD-MIL classifier on L2-normalised MoViNet features."""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class _MoViNetViolence(nn.Module):
    def __init__(self, backbone_path, head_path):
        super().__init__()
        from movinets import MoViNet
        from movinets.config import _C

        self.backbone = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=False, conv_type="2plus1d")
        self.backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=False))
        self.backbone.classifier[3] = nn.Identity()
        self.head = _Head()
        self.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        self.eval()

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> float:
        # Causal (streaming) MoViNet keeps activations between calls; each
        # window is scored independently.
        self.backbone.clean_activation_buffers()
        feats = F.normalize(self.backbone(clip), dim=1, p=2)
        return float(self.head(feats)[0, 0])


def get_model():
    """Load once; None (with the reason in error()) when unavailable."""
    global _model, _error
    if _model is not None or _error is not None:
        return _model
    with _load_lock:
        if _model is None and _error is None:
            ckpt = project_root() / "checkpoints"
            backbone, head = ckpt / "movinet_a0_stream.pth", ckpt / "movinet_vdmil_head.pt"
            if not backbone.exists() or not head.exists():
                _error = f"Missing {backbone.name} / {head.name}"
            else:
                try:
                    _model = _MoViNetViolence(backbone, head)
                except ImportError as exc:
                    _error = f"movinets not installed ({exc}); see requirements.txt"
                except Exception as exc:  # noqa: BLE001
                    _error = f"MoViNet load failed: {exc}"
    return _model


def error() -> str | None:
    return _error


def score(frames_rgb: list[np.ndarray]) -> float | None:
    """Violence probability for a ~1 s RGB clip; None if MoViNet is unavailable."""
    model = get_model()
    if model is None or not frames_rgb:
        return None
    idx = np.linspace(0, len(frames_rgb) - 1, FRAMES).round().astype(int)
    clip = np.stack([cv2.resize(frames_rgb[i], (SIZE, SIZE), interpolation=cv2.INTER_AREA) for i in idx])
    x = torch.from_numpy(clip).permute(3, 0, 1, 2)[None].float() / 255.0  # (1, C, T, H, W)
    return model(x)
