"""Bước 3 — Lấy mẫu T frame, resize, chuẩn hóa, lưu clip tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.feature_dataset import clip_id_of
from src.data.preprocess import load_frame_dir
from src.data.transforms import ClipTransform
from src.pipeline.common import banner, ensure_split_clip_id, load_cfg, out_dir, save_json
from src.utils.config import project_root
from src.utils.viz import plot_attention


def _save_clip_preview(frames_rgb: list[np.ndarray], weights: np.ndarray, path: Path) -> None:
    plot_attention(frames_rgb, weights, path)


def run(cfg: dict) -> dict:
    banner(3)
    root = project_root()
    splits_dir = root / cfg["data"]["splits_dir"]
    clips_dir = root / cfg["data"].get("clips_dir", "data/clips")
    t = cfg["data"]["num_frames"]
    size = cfg["data"]["image_size"]
    transform_eval = ClipTransform(image_size=size, train=False)
    transform_train = ClipTransform(image_size=size, train=True)

    counts = {}
    preview_done = set()
    for split in ("train", "val", "test"):
        df = ensure_split_clip_id(pd.read_csv(splits_dir / f"{split}.csv"))
        dest = clips_dir / split
        dest.mkdir(parents=True, exist_ok=True)
        tfm = transform_train if split == "train" else transform_eval
        for _, row in df.iterrows():
            cid = clip_id_of(row)
            rgb = load_frame_dir(row["path"], t)
            tensor = tfm(rgb)
            torch.save(
                {
                    "frames": tensor,
                    "label": int(row["label"]),
                    "clip_id": cid,
                    "split": split,
                    "class_name": row["class_name"],
                },
                dest / f"{cid}.pt",
            )
            if row["class_name"] not in preview_done and split == "train":
                weights = np.ones(len(rgb[: min(8, len(rgb))]), dtype=np.float32)
                weights /= weights.sum()
                _save_clip_preview(
                    rgb[: min(8, len(rgb))],
                    weights,
                    out_dir(cfg) / f"step03_preview_{row['class_name']}.png",
                )
                preview_done.add(row["class_name"])
        counts[split] = len(df)
        print(f"  {split}: {len(df)} clip  shape=(T={t}, C=3, H={size}, W={size})")

    sample = torch.load(next((clips_dir / "train").glob("*.pt")), map_location="cpu")
    report = {
        "num_frames": t,
        "image_size": size,
        "clip_shape": list(sample["frames"].shape),
        "counts": counts,
        "normalize": "ImageNet mean/std",
        "clips_dir": str(clips_dir),
    }
    save_json(out_dir(cfg) / "step03_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(load_cfg(args.config))


if __name__ == "__main__":
    main()
