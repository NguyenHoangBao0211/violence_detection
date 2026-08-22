from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.preprocess import sliding_windows
from src.data.transforms import eval_transform
from src.models.cnn_lstm import CNNLSTM
from src.utils.config import load_config, project_root
from src.utils.viz import plot_attention


def load_all_frames(video_path: str | Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return frames


@torch.no_grad()
def infer_video(cfg: dict, checkpoint: str | Path, video_path: str | Path, out_dir: str | Path) -> dict:
    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("cfg", cfg)
    model = CNNLSTM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frames = load_all_frames(video_path)
    window = cfg["infer"].get("window_frames", cfg["data"]["num_frames"])
    stride = cfg["infer"].get("stride", window // 2)
    threshold = cfg["infer"].get("threshold", 0.5)
    transform = eval_transform(cfg["data"]["image_size"])
    windows = sliding_windows(len(frames), window, stride)

    clip_probs = []
    clip_attn = []
    for idxs in windows:
        clip_np = [frames[int(i)] for i in idxs]
        tensor = transform(clip_np).unsqueeze(0).to(device)
        logits, attn = model(tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        clip_probs.append(prob)
        clip_attn.append(None if attn is None else attn[0].cpu().numpy())

    mean_prob = float(np.mean(clip_probs))
    max_prob = float(np.max(clip_probs))
    label = int(mean_prob >= threshold)
    class_name = cfg["data"]["classes"][label]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "video": str(video_path),
        "n_frames": len(frames),
        "n_clips": len(windows),
        "violence_prob_mean": mean_prob,
        "violence_prob_max": max_prob,
        "threshold": threshold,
        "pred_label": label,
        "pred_class": class_name,
        "clip_probs": clip_probs,
    }
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    best_i = int(np.argmax(clip_probs))
    if clip_attn[best_i] is not None:
        vis_frames = [frames[int(i)] for i in windows[best_i]]
        plot_attention(vis_frames, clip_attn[best_i], out_dir / "attention_best_clip.png")

    print(json.dumps({k: v for k, v in result.items() if k != "clip_probs"}, indent=2))
    print(f"saved: {out_dir / 'prediction.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run violence detection on a video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="reports/outputs/infer")
    args = parser.parse_args()
    root = project_root()
    cfg = load_config(root / args.config)
    infer_video(cfg, root / args.checkpoint, args.video, root / args.out)


if __name__ == "__main__":
    main()
