"""Forward-pass smoke test that does not require a video dataset."""

from __future__ import annotations

import torch

from src.models.cnn_lstm import CNNLSTM
from src.utils.config import load_config, project_root


def main() -> None:
    cfg = load_config(project_root() / "configs/default.yaml")
    cfg["model"]["pretrained"] = False
    model = CNNLSTM(cfg)
    model.eval()
    b, t = 2, cfg["data"]["num_frames"]
    x = torch.randn(b, t, 3, 224, 224)
    logits, attn = model(x)
    assert logits.shape == (b, 2)
    assert attn is not None and attn.shape == (b, t)
    print("smoke ok", {"logits": tuple(logits.shape), "attn": tuple(attn.shape)})


if __name__ == "__main__":
    main()
