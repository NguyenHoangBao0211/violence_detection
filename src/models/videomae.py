"""Local VideoMAE adapter for the upload/video-link inference pipeline."""
from copy import deepcopy
import os
from pathlib import Path

import torch


class LinearBiasAttention(torch.nn.Module):
    """Attention layout used by the supplied Transformers 5 checkpoint."""
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        for name in ("query", "key", "value"):
            setattr(self, name, torch.nn.Linear(config.hidden_size, config.hidden_size))

    def forward(self, hidden_states, head_mask=None):
        b, t, width = hidden_states.shape
        q, k, v = [getattr(self, name)(hidden_states).view(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for name in ("query", "key", "value")]
        context = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if head_mask is not None:
            context = context * head_mask
        return context.transpose(1, 2).reshape(b, t, width), None


class VideoMAEDetector(torch.nn.Module):
    def __init__(self, backbone, processor):
        super().__init__()
        self.backbone = backbone
        self.processor = processor

    def clip_transform(self, frames):
        # Processor owns resize/crop/rescale/normalization from the model card.
        return self.processor(list(frames), return_tensors="pt")["pixel_values"][0]

    def forward(self, frames):
        return self.backbone(pixel_values=frames).logits, None


def load_videomae(checkpoint, cfg, device):
    os.environ.setdefault("USE_TF", "0")
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor
    from safetensors.torch import load_file

    path = Path(checkpoint)
    folder = path if path.is_dir() else path.parent
    weights = folder / "model.safetensors" if path.is_dir() else path
    config = VideoMAEConfig.from_pretrained(folder, local_files_only=True)
    # Public card defines 0=non-violent and 1=violent; require binary weights.
    state = load_file(str(weights), device="cpu")
    if state["classifier.weight"].shape[0] != 2:
        raise ValueError("VideoMAE upload requires two classes.")
    config.num_labels = 2
    backbone = VideoMAEForVideoClassification(config)
    # Transformers 5 stores Q/K/V biases on Linear layers; version 4 uses
    # separate Q/V parameters. Preserve all supplied biases, including K.
    if any(k.endswith("attention.query.bias") for k in state):
        for layer in backbone.videomae.encoder.layer:
            layer.attention.attention = LinearBiasAttention(config)
    backbone.load_state_dict(state, strict=True)
    processor = VideoMAEImageProcessor.from_pretrained(folder, local_files_only=True)
    model = VideoMAEDetector(backbone, processor).to(device).eval()
    out_cfg = deepcopy(cfg or {})
    out_cfg["model_type"] = "videomae"
    out_cfg.setdefault("data", {}).update(
        classes=["non_violence", "violence"], image_size=config.image_size,
        num_frames=config.num_frames, frame_diff=False,
    )
    out_cfg.setdefault("infer", {}).update(
        window_frames=config.num_frames, stride=config.num_frames // 2,
        threshold=0.5, batch_size=1,
    )
    return model, out_cfg, device
