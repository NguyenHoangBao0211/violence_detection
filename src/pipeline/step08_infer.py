"""Bước 8 — Suy luận cửa sổ trượt trên một video."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.infer import infer_video
from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.utils.config import project_root


def _pick_video(cfg: dict, video: str | None) -> Path:
    if video:
        return Path(video)
    splits = project_root() / cfg["data"]["splits_dir"] / "test.csv"
    if splits.exists():
        df = pd.read_csv(splits)
        if "src" in df.columns and len(df):
            return Path(df.iloc[0]["src"])
    raw = project_root() / cfg["data"]["raw_dir"] / "violence"
    files = sorted(raw.glob("*.avi")) + sorted(raw.glob("*.mp4"))
    if not files:
        raise FileNotFoundError("Không tìm thấy video để suy luận. Truyền --video.")
    return files[0]


def run(cfg: dict, video: str | None = None) -> dict:
    banner(8)
    root = project_root()
    video_path = _pick_video(cfg, video)
    ckpt = root / cfg["train"]["checkpoint_dir"] / "best.pt"
    dest = out_dir(cfg) / "step08_infer"
    print(f"  video: {video_path}")
    print(f"  ckpt : {ckpt}")
    result = infer_video(cfg, ckpt, video_path, dest)
    save_json(out_dir(cfg) / "step08_report.json", {k: v for k, v in result.items() if k != "clip_probs"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--video", default=None)
    args = parser.parse_args()
    run(load_cfg(args.config), video=args.video)


if __name__ == "__main__":
    main()
