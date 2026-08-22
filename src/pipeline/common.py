from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.utils.config import load_config, project_root

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


STEPS = {
    1: "Thu thập dữ liệu video (violence / non_violence)",
    2: "Tiền xử lý: trích frame và chia train/val/test",
    3: "Tạo clip: lấy mẫu T frame, resize, chuẩn hóa",
    4: "CNN không gian: ResNet50 → đặc trưng từng frame",
    5: "Khối thời gian: BiLSTM + attention trên chuỗi đặc trưng",
    6: "Huấn luyện bộ phân loại không gian–thời gian",
    7: "Đánh giá trên tập test",
    8: "Suy luận cửa sổ trượt trên video mới",
}


def load_cfg(config_path: str | Path) -> dict:
    return load_config(project_root() / config_path)


def banner(step: int) -> None:
    title = STEPS[step]
    line = "=" * 72
    print(f"\n{line}\nBƯỚC {step}/8  {title}\n{line}", flush=True)


def out_dir(cfg: dict) -> Path:
    path = project_root() / cfg["train"]["log_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {path}", flush=True)


def ensure_split_clip_id(df):
    if "clip_id" not in df.columns:
        df = df.copy()
        df["clip_id"] = df["class_name"].astype(str) + "__" + df["video_id"].astype(str)
    return df
