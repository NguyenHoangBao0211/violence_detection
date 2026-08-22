"""Bước 1 — Thu thập / kiểm kê dữ liệu video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.preprocess import VIDEO_EXTS, list_videos
from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.utils.config import project_root
from src.utils.seed import set_seed


def _writer(path: Path, size: int, fps: int = 12) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not vw.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    return vw


def _make_non_violence(path: Path, n_frames: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vw = _writer(path, size)
    cx, cy = size // 3, size // 2
    vx, vy = 0.8, 0.2
    color_bg = (int(rng.integers(50, 90)), int(rng.integers(90, 140)), int(rng.integers(50, 90)))
    for _ in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = color_bg
        cx = np.clip(cx + vx, 30, size - 30)
        cy = np.clip(cy + vy, 30, size - 30)
        if cx in (30, size - 30):
            vx *= -1
        if cy in (30, size - 30):
            vy *= -1
        cv2.circle(frame, (int(cx), int(cy)), 16, (220, 220, 220), -1)
        vw.write(frame)
    vw.release()


def _make_violence(path: Path, n_frames: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vw = _writer(path, size)
    x1, x2 = 20.0, float(size - 20)
    y = size / 2
    speed = 4.0 + float(rng.random())
    for t in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = (25, 25, 40)
        x1 = 20 + (t * speed) % (size - 40)
        x2 = size - 20 - (t * speed) % (size - 40)
        if abs(x1 - x2) < 36:
            frame[:] = (20, 20, 200)
            cv2.line(frame, (int(x1), int(y - 20)), (int(x2), int(y + 20)), (0, 0, 255), 4)
        cv2.circle(frame, (int(x1), int(y)), 14, (40, 40, 255), -1)
        cv2.circle(frame, (int(x2), int(y)), 14, (40, 80, 255), -1)
        vw.write(frame)
    vw.release()


def generate_demo_videos(cfg: dict) -> int:
    root = project_root()
    raw = root / cfg["data"]["raw_dir"]
    n = int(cfg["data"].get("demo_videos_per_class", 10))
    n_frames = int(cfg["data"].get("demo_source_frames", 32))
    size = int(cfg["data"]["image_size"])
    created = 0
    for i in range(n):
        nv = raw / "non_violence" / f"demo_nv_{i:03d}.avi"
        vv = raw / "violence" / f"demo_v_{i:03d}.avi"
        if not nv.exists():
            _make_non_violence(nv, n_frames, size, seed=cfg["seed"] + i)
            created += 1
        if not vv.exists():
            _make_violence(vv, n_frames, size, seed=cfg["seed"] + 100 + i)
            created += 1
    return created


def run(cfg: dict, make_demo: bool = True) -> dict:
    banner(1)
    set_seed(cfg["seed"])
    root = project_root()
    raw = root / cfg["data"]["raw_dir"]
    for cls in cfg["data"]["classes"]:
        (raw / cls).mkdir(parents=True, exist_ok=True)

    existing = list_videos(raw, cfg["data"]["classes"])
    generated = 0
    if make_demo and not existing:
        print("  Chưa có video thật → tạo bộ demo tổng hợp để chạy đủ 8 bước.")
        generated = generate_demo_videos(cfg)
        existing = list_videos(raw, cfg["data"]["classes"])
    elif make_demo and existing:
        print(f"  Đã có {len(existing)} video, giữ nguyên (không ghi đè).")

    if not existing:
        raise FileNotFoundError(
            f"Không tìm thấy video trong {raw}. Đặt file vào violence/ và non_violence/."
        )

    df = pd.DataFrame(existing)
    counts = df.groupby("class_name").size().to_dict()
    print(f"  Tổng video: {len(df)}")
    for k, v in counts.items():
        print(f"    {k}: {v}")

    report = {
        "n_videos": len(df),
        "per_class": {str(k): int(v) for k, v in counts.items()},
        "generated_demo": generated,
        "raw_dir": str(raw),
        "extensions": sorted({Path(p).suffix.lower() for p in df["src"]}),
    }
    df.to_csv(out_dir(cfg) / "step01_inventory.csv", index=False)
    save_json(out_dir(cfg) / "step01_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--no-demo", action="store_true")
    args = parser.parse_args()
    run(load_cfg(args.config), make_demo=not args.no_demo)


if __name__ == "__main__":
    main()
