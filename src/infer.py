from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.preprocess import sliding_windows
from src.data.transforms import eval_transform
from src.models.cnn_lstm import CNNLSTM, MobileNetBiLSTM, ResNet50BiLSTMAttention
from src.utils.config import load_config, project_root
from src.utils.viz import plot_attention

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(project_root() / "storage" / "ultralytics"),
)

_yolo_person_model = None


def _get_yolo_person_model():
    """Load the lightweight person detector once, only when webcam boxes are requested."""
    global _yolo_person_model
    if _yolo_person_model is None:
        weights = project_root() / "checkpoints" / "yolo11n.pt"
        if not weights.exists():
            return None
        try:
            from ultralytics import YOLO
        except ImportError:
            return None
        _yolo_person_model = YOLO(str(weights))
    return _yolo_person_model


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


def load_detector(checkpoint: str | Path, cfg: dict | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(checkpoint).lower().endswith((".h5", ".keras")):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("Model .h5 cần TensorFlow. Cài bằng: pip install tensorflow") from exc

        keras_model = tf.keras.models.load_model(checkpoint, compile=False)

        class KerasDetector(torch.nn.Module):
            def forward(self, frames):
                # Notebook model expects (B,T,64,64,3), values in [0,1].
                x = torch.nn.functional.interpolate(
                    frames.flatten(0, 1), size=(64, 64), mode="bilinear", align_corners=False
                ).view(frames.shape[0], frames.shape[1], 3, 64, 64)
                x = x.permute(0, 1, 3, 4, 2).clamp(0, 1).cpu().numpy()
                probs = keras_model.predict(x, verbose=0)
                probs = torch.as_tensor(probs, dtype=torch.float32, device=frames.device)
                return torch.log(probs.clamp_min(1e-7)), None

        out_cfg = dict(cfg or {})
        out_cfg["model_type"] = "keras_mobilenet_bilstm"
        out_cfg["data"] = dict(out_cfg.get("data", {}), image_size=224, num_frames=16,
                                classes=["non_violence", "violence"])
        out_cfg["infer"] = dict(out_cfg.get("infer", {}), window_frames=16, threshold=0.5)
        return KerasDetector().to(device).eval(), out_cfg, device

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # X3D checkpoints produced by the realtime ablation notebook carry a
    # `chosen_config` record rather than the CNN-LSTM config.
    chosen = ckpt.get("chosen_config")
    if chosen and chosen.get("model") == "x3d_m":
        try:
            from pytorchvideo.models.hub import x3d_m
        except ImportError as exc:
            raise RuntimeError(
                "X3D model cần pytorchvideo. Cài bằng: pip install pytorchvideo"
            ) from exc

        class X3DDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = x3d_m(pretrained=False)
                # The supplied checkpoint was trained with Dropout + Linear
                # at the final projection, whereas the stock X3D head is a
                # single Linear layer.
                self.backbone.blocks[5].proj = torch.nn.Sequential(
                    torch.nn.Dropout(0.5), torch.nn.Linear(2048, 2)
                )

            def forward(self, frames):
                # Project tensors are (B,T,C,H,W); X3D expects (B,C,T,H,W).
                logits = self.backbone(frames.permute(0, 2, 1, 3, 4).contiguous())
                return logits, None

        model = X3DDetector().to(device)
        model.load_state_dict(ckpt["model"])
        out_cfg = dict(cfg or {})
        out_cfg["data"] = dict(out_cfg.get("data", {}))
        out_cfg["data"].update(classes=["non_violence", "violence"], image_size=224, num_frames=chosen.get("frames", 16))
        out_cfg["model_type"] = "x3d_m"
        out_cfg["infer"] = dict(out_cfg.get("infer", {}))
        out_cfg["infer"].update(window_frames=chosen.get("frames", 16), threshold=chosen.get("threshold", 0.4))
        model.eval()
        return model, out_cfg, device

    cfg = ckpt.get("cfg", cfg)
    if cfg is None:
        raise ValueError("Checkpoint missing cfg; pass a config dict.")
    # Older training notebooks saved only data/model. Keep inference defaults
    # so those checkpoints remain usable by the upload and webcam APIs.
    cfg = dict(cfg)
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["infer"] = dict(cfg.get("infer", {}))
    cfg["infer"].setdefault("window_frames", cfg["data"].get("num_frames", 16))
    cfg["infer"].setdefault("stride", max(1, cfg["infer"]["window_frames"] // 2))
    cfg["infer"].setdefault("threshold", 0.6)
    state_keys = ckpt["model"].keys()
    if any(k.startswith("cnn.") for k in state_keys) and any(k.startswith("attention.") for k in state_keys):
        model = ResNet50BiLSTMAttention(cfg).to(device)
    elif cfg.get("model", {}).get("backbone") == "mobilenet_v2":
        model = MobileNetBiLSTM(cfg).to(device)
    else:
        model = CNNLSTM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, device


def _sample_clip(frames: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
    """Uniformly sample (or pad) frames to a fixed clip length."""
    if not frames:
        raise RuntimeError("Empty frame list")
    n = len(frames)
    if n == num_frames:
        return list(frames)
    if n < num_frames:
        idxs = np.linspace(0, n - 1, num_frames).astype(int)
        return [frames[int(i)] for i in idxs]
    idxs = np.linspace(0, n - 1, num_frames).astype(int)
    return [frames[int(i)] for i in idxs]


def detect_persons(frame_rgb: np.ndarray) -> list[dict]:
    """Detect people with YOLO11n, with HOG as a local fallback."""
    h, w = frame_rgb.shape[:2]
    yolo = _get_yolo_person_model()
    if yolo is not None:
        result = yolo.predict(
            source=frame_rgb,
            classes=[0],
            conf=0.25,
            imgsz=640,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu",
        )[0]
        boxes = []
        if result.boxes is not None:
            for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = xyxy.tolist()
                boxes.append({
                    "x": max(0.0, x1 / w), "y": max(0.0, y1 / h),
                    "w": max(0.0, (x2 - x1) / w), "h": max(0.0, (y2 - y1) / h),
                    "score": float(conf),
                })
        return boxes

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # Downscale for speed; map boxes back to original size.
    scale = 1.0
    max_side = 640
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        bgr, winStride=(8, 8), padding=(16, 16), scale=1.05, hitThreshold=0.0
    )

    boxes: list[dict] = []
    for (x, y, bw, bh), wt in zip(rects, weights):
        if scale != 1.0:
            x, y, bw, bh = x / scale, y / scale, bw / scale, bh / scale
        conf = float(wt[0]) if hasattr(wt, "__len__") else float(wt)
        boxes.append(
            {
                "x": float(x) / w,
                "y": float(y) / h,
                "w": float(bw) / w,
                "h": float(bh) / h,
                "score": conf,
            }
        )

    # If both detectors miss, keep the UI response valid without inventing a
    # person crop; classification remains on the original frame.
    if not boxes:
        return []
    return boxes


def crop_person_clip(frames: list[np.ndarray], boxes: list[dict]) -> list[np.ndarray]:
    """Crop the union of detected people with a small context margin."""
    if not boxes:
        return frames
    h, w = frames[0].shape[:2]
    x1 = min(b["x"] for b in boxes)
    y1 = min(b["y"] for b in boxes)
    x2 = max(b["x"] + b["w"] for b in boxes)
    y2 = max(b["y"] + b["h"] for b in boxes)
    pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    left = max(0, int((x1 - pad_x) * w))
    top = max(0, int((y1 - pad_y) * h))
    right = min(w, int((x2 + pad_x) * w))
    bottom = min(h, int((y2 + pad_y) * h))
    if right - left < 32 or bottom - top < 32:
        return frames
    return [frame[top:bottom, left:right] for frame in frames]


@torch.no_grad()
def run_inference_frames(
    model,
    cfg: dict,
    device: torch.device,
    frames: list[np.ndarray],
    *,
    with_boxes: bool = True,
) -> dict:
    """Classify a short RGB frame clip and optionally detect person boxes."""
    window = cfg["infer"].get("window_frames", cfg["data"]["num_frames"])
    threshold = cfg["infer"].get("threshold", 0.5)
    transform = eval_transform(cfg["data"]["image_size"], x3d=cfg.get("model_type") == "x3d_m")
    clip = _sample_clip(frames, window)
    tensor = transform(clip).unsqueeze(0).to(device)
    logits, _attn = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    violence_prob = float(probs[1].item())
    non_prob = float(probs[0].item())
    label = int(violence_prob >= threshold)
    class_name = cfg["data"]["classes"][label]

    last = frames[-1]
    boxes = detect_persons(last) if with_boxes else []
    # For webcam mode, use the YOLO person union crop when available. This
    # gives distant subjects more pixels and removes most background noise.
    if with_boxes and boxes:
        cropped = crop_person_clip(clip, boxes)
        tensor = transform(cropped).unsqueeze(0).to(device)
        logits, _attn = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        violence_prob = float(probs[1].item())
        non_prob = float(probs[0].item())
        label = int(violence_prob >= threshold)
        class_name = cfg["data"]["classes"][label]
    return {
        "n_frames": len(frames),
        "n_clips": 1,
        "violence_prob_mean": violence_prob,
        "non_violence_prob": non_prob,
        "violence_prob_max": violence_prob,
        "threshold": threshold,
        "pred_label": label,
        "pred_class": class_name,
        "boxes": boxes,
        "frame_w": int(last.shape[1]),
        "frame_h": int(last.shape[0]),
    }


@torch.no_grad()
def run_inference(model, cfg: dict, device: torch.device, video_path: str | Path, out_dir: str | Path) -> dict:
    frames = load_all_frames(video_path)
    window = cfg["infer"].get("window_frames", cfg["data"]["num_frames"])
    stride = cfg["infer"].get("stride", window // 2)
    threshold = cfg["infer"].get("threshold", 0.5)
    transform = eval_transform(cfg["data"]["image_size"], x3d=cfg.get("model_type") == "x3d_m")
    windows = sliding_windows(len(frames), window, stride)

    clip_probs = []
    clip_attn = []
    for idxs in windows:
        clip_np = [frames[int(i)] for i in idxs]
        tensor = transform(clip_np).unsqueeze(0).to(device)
        logits, attn = model(tensor)
        full_prob = torch.softmax(logits, dim=1)[0, 1].item()
        prob = full_prob

        # Upload ensemble: combine the original scene with a YOLO person crop.
        # The full-frame branch keeps compatibility with the training data;
        # the crop branch gives distant people more useful pixels.
        boxes = detect_persons(clip_np[-1])
        if boxes:
            cropped = crop_person_clip(clip_np, boxes)
            crop_tensor = transform(cropped).unsqueeze(0).to(device)
            crop_logits, _ = model(crop_tensor)
            crop_prob = torch.softmax(crop_logits, dim=1)[0, 1].item()
            prob = 0.4 * full_prob + 0.6 * crop_prob

        clip_probs.append(prob)
        clip_attn.append(None if attn is None else attn[0].cpu().numpy())

    mean_prob = float(np.mean(clip_probs))
    max_prob = float(np.max(clip_probs))
    # A violent event may occupy only part of a long video. Do not dilute it
    # with intro/outro frames: require three adjacent windows above threshold.
    streak = 0
    max_streak = 0
    violent_windows = 0
    for prob in clip_probs:
        if prob >= threshold:
            streak += 1
            violent_windows += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    # Do not let a short false-positive burst label the whole video.
    # Require both temporal persistence and sufficient probability across
    # the complete video. This is more conservative for normal motion,
    # waving, dancing, and exercise clips.
    event_detected = max_streak >= 3 and mean_prob >= threshold
    label = int(event_detected)
    class_name = cfg["data"]["classes"][label]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "video": str(video_path),
        "n_frames": len(frames),
        "n_clips": len(windows),
        "violence_prob_mean": mean_prob,
        "violence_prob_max": max_prob,
        "violent_windows": violent_windows,
        "max_violence_streak": max_streak,
        "event_detected": event_detected,
        "threshold": threshold,
        "pred_label": label,
        "pred_class": class_name,
        "clip_probs": clip_probs,
        "boxes": detect_persons(frames[-1]),
    }
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    best_i = int(np.argmax(clip_probs))
    if clip_attn[best_i] is not None:
        vis_frames = [frames[int(i)] for i in windows[best_i]]
        plot_attention(vis_frames, clip_attn[best_i], out_dir / "attention_best_clip.png")

    print(json.dumps({k: v for k, v in result.items() if k != "clip_probs"}, indent=2))
    print(f"saved: {out_dir / 'prediction.json'}")
    return result


@torch.no_grad()
def infer_video(cfg: dict, checkpoint: str | Path, video_path: str | Path, out_dir: str | Path) -> dict:
    model, cfg, device = load_detector(checkpoint, cfg)
    return run_inference(model, cfg, device, video_path, out_dir)


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
