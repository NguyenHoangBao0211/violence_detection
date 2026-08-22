"""Bước 4 — CNN không gian: ResNet50 trích đặc trưng từng frame."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.models.cnn_lstm import SpatialEncoder
from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.utils.config import project_root


def run(cfg: dict) -> dict:
    banner(4)
    root = project_root()
    clips_dir = root / cfg["data"].get("clips_dir", "data/clips")
    feat_dir = root / cfg["data"].get("features_dir", "data/features")
    ckpt_dir = root / cfg["train"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  backbone={cfg['model']['backbone']}  freeze={cfg['model']['freeze_backbone']}")

    encoder = SpatialEncoder(
        pretrained=cfg["model"].get("pretrained", True),
        freeze=True,
    ).to(device)
    encoder.eval()

    n_clips = 0
    feat_shape = None
    with torch.no_grad():
        for split in ("train", "val", "test"):
            split_in = clips_dir / split
            split_out = feat_dir / split
            split_out.mkdir(parents=True, exist_ok=True)
            files = sorted(split_in.glob("*.pt"))
            for path in tqdm(files, desc=f"  spatial {split}", leave=False):
                item = torch.load(path, map_location="cpu", weights_only=False)
                frames = item["frames"].unsqueeze(0).to(device)
                b, t, c, h, w = frames.shape
                feat = encoder(frames.reshape(b * t, c, h, w)).view(t, -1).cpu().numpy()
                np.save(split_out / f"{item['clip_id']}.npy", feat)
                feat_shape = list(feat.shape)
                n_clips += 1
            print(f"  {split}: {len(files)} chuỗi đặc trưng")

    torch.save({"spatial": encoder.state_dict(), "cfg": cfg}, ckpt_dir / "spatial.pt")
    report = {
        "n_clips": n_clips,
        "feature_shape": feat_shape,
        "meaning": "T x 2048 — mỗi hàng là vector không gian của 1 frame",
        "spatial_checkpoint": str(ckpt_dir / "spatial.pt"),
    }
    save_json(out_dir(cfg) / "step04_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(load_cfg(args.config))


if __name__ == "__main__":
    main()
