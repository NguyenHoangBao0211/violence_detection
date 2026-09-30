from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src import movinet_veto
from src.data.preprocess import sliding_windows
from src.data.transforms import eval_transform
from src.models.cnn_lstm import CNNLSTM, MobileNetBiLSTM, ResNet50BiLSTMAttention, TwoStreamBiLSTMAttention
from src.utils.config import load_config, project_root
from src.utils.viz import plot_attention

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(project_root() / "storage" / "ultralytics"),
)

_yolo_person_model = None

# One model call at a time. On CPU, a long upload and realtime webcam requests
# running in parallel oversubscribe the cores and every request gets several
# times slower. Uploads take the lock per window, so a webcam request waits
# at most for one window instead of the whole video.
_infer_lock = threading.Lock()


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


def load_all_frames(video_path: str | Path, max_side: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if max_side:
            height, width = frame.shape[:2]
            longest = max(height, width)
            if longest > max_side:
                scale = max_side / longest
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return frames


def _video_fps(video_path: str | Path, default: float = 25.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # Some containers (webm from MediaRecorder) report 0 or a bogus 1000 fps.
    return float(fps) if 1.0 <= fps <= 240.0 else default


def _window_frame_diffs(clip: list[np.ndarray]) -> list[np.ndarray]:
    """Same input the Kaggle notebook trains on: frame 0 is black, later frames are absdiff."""
    if not clip:
        return clip
    diffs = [np.zeros_like(clip[0])]
    for prev, cur in zip(clip, clip[1:]):
        diffs.append(cv2.absdiff(cur, prev))
    return diffs


def _violent_runs(probs: list[float], threshold: float) -> list[tuple[int, int]]:
    """Inclusive (first, last) index of each run of windows at or above threshold."""
    runs = []
    start = None
    for i, prob in enumerate(probs):
        if prob >= threshold and start is None:
            start = i
        elif prob < threshold and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(probs) - 1))
    return runs


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
    model_cfg = cfg.get("model", {})
    has_cnn = any(k.startswith("cnn.") for k in state_keys)
    if model_cfg.get("arch") == "two_stream":
        model = TwoStreamBiLSTMAttention(cfg).to(device)
    elif has_cnn and any(k.startswith("attention.") for k in state_keys):
        model = ResNet50BiLSTMAttention(cfg).to(device)
    elif model_cfg.get("backbone") == "mobilenet_v2":
        model = MobileNetBiLSTM(cfg).to(device)
    elif has_cnn and model_cfg.get("use_attention") is False:
        # Ablation checkpoint from the notebook: same keys, mean pooling.
        model = ResNet50BiLSTMAttention(cfg).to(device)
    else:
        model = CNNLSTM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, device


def _as_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class probabilities for one clip.

    X3D's head already applies softmax, so its two outputs sum to 1. Softmaxing
    those again squashes a confident non-violence call into about 73% and the
    number barely moves. Real logits (ResNet) still go through softmax.
    """
    row = logits.reshape(-1)
    if (
        row.numel() == 2
        and torch.all(row >= -1e-4)
        and torch.isclose(row.sum(), row.new_tensor(1.0), atol=1e-3)
    ):
        return row.clamp(0, 1)
    return torch.softmax(logits.flatten(1), dim=1)[0]


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


def detect_persons(frame_rgb: np.ndarray, imgsz: int = 640) -> list[dict]:
    """Detect people with YOLO11n, with HOG as a local fallback."""
    h, w = frame_rgb.shape[:2]
    yolo = _get_yolo_person_model()
    if yolo is not None:
        result = yolo.predict(
            source=frame_rgb,
            classes=[0],
            conf=0.25,
            imgsz=imgsz,
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


def _clip_motion(frames: list[np.ndarray]) -> float:
    """Mean pixel change from the first to the last frame, 0–255.

    Adjacent frames at 30 fps differ by only 1–3 even in a fight (small
    people on CCTV), so compare across the whole window (~0.5 s) instead.
    Fights in the saved clips score about 2.5–40; a still scene is near 0.
    """
    if len(frames) < 2:
        return 0.0

    def _gray(frame):
        small = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    return float(np.mean(cv2.absdiff(_gray(frames[0]), _gray(frames[-1]))))


@torch.no_grad()
def run_inference_frames(
    model,
    cfg: dict,
    device: torch.device,
    frames: list[np.ndarray],
    *,
    with_boxes: bool = True,
    with_movinet: bool = False,
) -> dict:
    """Classify a short RGB frame clip and optionally detect person boxes.

    With more frames than the model window (the Camera page sends ~1 s), the
    classifier sees the last `window` consecutive frames, as in training, and
    MoViNet (with_movinet) scores the whole second as a second opinion.
    """
    started = time.perf_counter()
    window = cfg["infer"].get("window_frames", cfg["data"]["num_frames"])
    threshold = cfg["infer"].get("threshold", 0.5)
    transform = eval_transform(cfg["data"]["image_size"], x3d=cfg.get("model_type") == "x3d_m")
    clip = list(frames[-window:]) if len(frames) >= window else _sample_clip(frames, window)
    last = frames[-1]

    use_diff = bool(cfg.get("data", {}).get("frame_diff"))

    def _score(clip_frames):
        # A frame-difference checkpoint must see differences here too, as in upload.
        if use_diff:
            clip_frames = _window_frame_diffs(clip_frames)
        tensor = transform(clip_frames).unsqueeze(0).to(device)
        logits, attn = model(tensor)
        probs = _as_probs(logits)
        return float(probs[1].item()), float(probs[0].item()), attn

    movinet_prob = None
    with _infer_lock:
        # Webcam frames arrive at <=640px; YOLO at the same size keeps small,
        # distant people detectable. The checkpoint was trained on full frames,
        # but a laptop webcam leaves people small, so also score the person
        # crop and keep the higher violence probability.
        boxes = detect_persons(last, imgsz=640) if with_boxes else []
        # One forward only. A second pass on the person crop doubled the wait
        # on CPU, so the label fell behind the live camera. Use the crop when
        # a person is in frame (more pixels on the action) and the full frame
        # otherwise.
        scored = clip
        if boxes and cfg.get("model_type") != "x3d_m":
            cropped = crop_person_clip(clip, boxes)
            if cropped is not clip:
                scored = cropped
        violence_prob, non_prob, attn = _score(scored)
        # Second opinion only while a ResNet score is uncertain. X3D already
        # has its own threshold and should not be pulled down by MoViNet.
        if with_movinet and cfg.get("model_type") != "x3d_m" and threshold <= violence_prob < 0.8:
            movinet_prob = movinet_veto.score(frames)
    # A close-up of a person looks like the fight clips this model was trained
    # on, so a seated webcam user can score near 100% violence. No real strike
    # leaves the picture this still.
    motion = _clip_motion(clip)
    if motion < 2.5 and violence_prob > 0.35:
        violence_prob = min(violence_prob, 0.2)
        non_prob = 1.0 - violence_prob
    # Attention marks the frame that drove the decision (usually the strike),
    # not the first frame of the window and not the time the response returns.
    if attn is not None:
        peak_frame = int(torch.argmax(attn[0]).item())
    else:
        peak_frame = max(0, len(clip) - 1)
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
        "movinet_prob": movinet_prob,
        "peak_frame": peak_frame,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


@torch.no_grad()
def run_inference(model, cfg: dict, device: torch.device, video_path: str | Path, out_dir: str | Path) -> dict:
    # The model only sees 224px. Shrink once here so decode and every later
    # resize are not done on a full-HD frame.
    frames = load_all_frames(video_path, max_side=480)
    fps = _video_fps(video_path)
    window = cfg["infer"].get("window_frames", cfg["data"]["num_frames"])
    base_stride = max(1, int(cfg["infer"].get("stride", max(1, window // 2))))
    stride = base_stride
    # A long video at stride 8 is hundreds of ResNet50 passes on CPU. Widen
    # the stride so an upload stays around a few dozen windows.
    if len(frames) > window:
        rough = (len(frames) - window) // base_stride + 1
        if rough > 32:
            stride = max(base_stride, int(np.ceil((len(frames) - window) / 31)))
    threshold = cfg["infer"].get("threshold", 0.5)
    transform = eval_transform(cfg["data"]["image_size"], x3d=cfg.get("model_type") == "x3d_m")
    windows = sliding_windows(len(frames), window, stride)
    # Three overlapping windows at the base stride. A wider stride covers the
    # same amount of time with fewer windows, so the streak shrinks with it.
    # Never fewer than two, or one lone window of a long video decides the label.
    need_streak = 3 if stride == base_stride else max(2, int(round(3 * base_stride / stride)))
    # A clip shorter than that many windows can never reach the streak.
    need_streak = min(need_streak, len(windows))

    use_diff = bool(cfg.get("data", {}).get("frame_diff"))
    needed = sorted({int(i) for w in windows for i in w})
    prepared = {} if use_diff else {i: transform([frames[i]])[0] for i in needed}

    def _window_tensor(idxs):
        if not use_diff:
            return torch.stack([prepared[int(i)] for i in idxs], dim=0)
        clip = _window_frame_diffs([frames[int(i)] for i in idxs])
        return transform(clip)

    clip_probs = []
    clip_attn = []
    batch_size = 4
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        batch = torch.stack([_window_tensor(idxs) for idxs in batch_windows]).to(device)
        with _infer_lock:
            logits, attn = model(batch)
        for b in range(batch.shape[0]):
            clip_probs.append(float(_as_probs(logits[b : b + 1])[1].item()))
            clip_attn.append(None if attn is None else attn[b].detach().cpu().numpy())

    mean_prob = float(np.mean(clip_probs))
    max_prob = float(np.max(clip_probs))
    violent_windows = sum(1 for prob in clip_probs if prob >= threshold)
    # A violent event may occupy only part of a long video, so the label comes
    # from runs of adjacent windows above threshold, not from the video mean.
    # A run shorter than `need_streak` is a false-positive burst and is dropped.
    runs = _violent_runs(clip_probs, threshold)
    max_streak = max((end - start + 1 for start, end in runs), default=0)
    segments = [
        {
            "start_s": round(int(windows[start][0]) / fps, 2),
            "end_s": round((int(windows[end][-1]) + 1) / fps, 2),
            "prob_max": round(max(clip_probs[start : end + 1]), 4),
            "prob_mean": round(float(np.mean(clip_probs[start : end + 1])), 4),
            "n_windows": end - start + 1,
        }
        for start, end in runs
        if end - start + 1 >= need_streak
    ]
    event_detected = bool(segments)
    label = int(event_detected)
    class_name = cfg["data"]["classes"][label]

    with _infer_lock:
        last_boxes = detect_persons(frames[-1])

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
        "min_violence_streak": need_streak,
        "event_detected": event_detected,
        "threshold": threshold,
        "pred_label": label,
        "pred_class": class_name,
        "fps": fps,
        "duration_s": round(len(frames) / fps, 2),
        "segments": segments,
        "clip_probs": clip_probs,
        # Centre of each window in seconds, for the probability timeline.
        "clip_times": [round((int(w[0]) + int(w[-1]) + 1) / 2 / fps, 2) for w in windows],
        "boxes": last_boxes,
    }
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    best_i = int(np.argmax(clip_probs))
    if clip_attn[best_i] is not None:
        vis_frames = [frames[int(i)] for i in windows[best_i]]
        plot_attention(vis_frames, clip_attn[best_i], out_dir / "attention_best_clip.png")

    print(json.dumps({k: v for k, v in result.items() if k not in {"clip_probs", "clip_times"}}, indent=2))
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
