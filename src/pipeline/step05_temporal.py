"""Bước 5 — Khối thời gian BiLSTM + attention (kiểm tra forward trên chuỗi đặc trưng)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.data.preprocess import load_frame_dir
from src.models.cnn_lstm import TemporalHead
from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.utils.config import project_root
from src.utils.viz import plot_attention


def run(cfg: dict) -> dict:
    banner(5)
    root = project_root()
    feat_dir = root / cfg["data"].get("features_dir", "data/features") / "train"
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError("Chưa có file đặc trưng. Hãy chạy bước 4 trước.")

    sample = np.load(files[0])
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    head = TemporalHead(input_dim=sample.shape[1], cfg=cfg)
    head.eval()
    with torch.no_grad():
        logits, alpha = head(x)

    print(f"  input  : {tuple(x.shape)}  (B, T, D=2048)")
    print(f"  logits : {tuple(logits.shape)}  (B, 2)")
    print(f"  attn   : {tuple(alpha.shape) if alpha is not None else None}  (B, T)")
    print(f"  LSTM hidden={cfg['model']['lstm_hidden']}  bidirectional={cfg['model']['bidirectional']}")

    splits = root / cfg["data"]["splits_dir"] / "train.csv"
    df = pd.read_csv(splits)
    row = df.iloc[0]
    frames = load_frame_dir(row["path"], cfg["data"]["num_frames"])
    weights = alpha[0].numpy() if alpha is not None else np.ones(len(frames)) / len(frames)
    plot_attention(frames, weights, out_dir(cfg) / "step05_attention_untrained.png")

    ckpt_dir = root / cfg["train"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"temporal": head.state_dict(), "cfg": cfg}, ckpt_dir / "temporal_init.pt")

    report = {
        "input_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "attention_shape": list(alpha.shape) if alpha is not None else None,
        "lstm_hidden": cfg["model"]["lstm_hidden"],
        "bidirectional": cfg["model"]["bidirectional"],
        "use_attention": cfg["model"]["use_attention"],
        "note": "Weights chưa train — bước 6 mới tối ưu. Ảnh attention lúc này gần như đều.",
    }
    save_json(out_dir(cfg) / "step05_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(load_cfg(args.config))


if __name__ == "__main__":
    main()
