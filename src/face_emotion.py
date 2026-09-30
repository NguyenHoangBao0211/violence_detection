"""Webcam decision from the face, not the whole scene.

Quy tắc:
- mặt cộc bực (anger, disgust, contempt) -> violence
- cười vui / đùa giỡn (happiness, surprise) -> non_violence
- cảm xúc khác hoặc không thấy mặt -> non_violence
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

EMOTIONS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)
PLAYFUL = {"happiness", "surprise"}
GRUMPY = {"anger", "disgust", "contempt"}
LABELS = {
    "happiness": "Vui",
    "surprise": "Đùa giỡn",
    "anger": "Cộc bực",
    "disgust": "Cộc bực",
    "contempt": "Cộc bực",
    "neutral": "Bình thường",
    "sadness": "Buồn",
    "fear": "Sợ",
}

_session = None
_cascade = None
_error: str | None = None


def _model_path() -> Path:
    from src.utils.config import project_root

    return project_root() / "checkpoints" / "emotion-ferplus-8.onnx"


def get_session():
    global _session, _error
    if _session is not None or _error:
        return _session
    path = _model_path()
    if not path.exists():
        _error = f"Thiếu model cảm xúc: {path}"
        return None
    try:
        import onnxruntime as ort

        _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        _error = str(exc)
        _session = None
    return _session


def error() -> str | None:
    get_session()
    return _error


def _detector() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


def _largest_face(frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    min_side = max(48, int(min(h, w) * 0.18))
    faces = _detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
    )
    if len(faces) == 0:
        return None
    x, y, bw, bh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(bw), int(bh)


def _emotion_probs(face_rgb: np.ndarray) -> np.ndarray | None:
    session = get_session()
    if session is None or face_rgb.size == 0:
        return None
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    blob = resized.astype(np.float32)[None, None, :, :]
    name = session.get_inputs()[0].name
    raw = np.asarray(session.run(None, {name: blob})[0], dtype=np.float32).reshape(-1)
    if raw.size < 8:
        return None
    raw = raw[:8]
    # FER+ có bản đã softmax và bản logits. Chuẩn hóa lại cho chắc.
    if raw.min() < 0 or abs(float(raw.sum()) - 1.0) > 0.05:
        raw = raw - raw.max()
        exp = np.exp(raw)
        raw = exp / exp.sum()
    return raw


def score_frames(frames_rgb: list[np.ndarray]) -> dict:
    """Score the latest frames by facial expression."""
    if not frames_rgb:
        raise RuntimeError("Empty frames")
    last = frames_rgb[-1]
    h, w = last.shape[:2]
    samples = frames_rgb[-6:]
    probs_list: list[np.ndarray] = []
    box = None
    for frame in samples:
        found = _largest_face(frame)
        if found is None:
            continue
        x, y, bw, bh = found
        pad = int(0.12 * max(bw, bh))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + bw + pad), min(frame.shape[0], y + bh + pad)
        probs = _emotion_probs(frame[y1:y2, x1:x2])
        if probs is not None:
            probs_list.append(probs)
            box = (x1, y1, x2 - x1, y2 - y1, frame.shape[1], frame.shape[0])

    if not probs_list or box is None:
        return {
            "n_frames": len(frames_rgb),
            "n_clips": 1,
            "violence_prob_mean": 0.0,
            "non_violence_prob": 1.0,
            "violence_prob_max": 0.0,
            "threshold": 0.45,
            "pred_label": 0,
            "pred_class": "non_violence",
            "boxes": [],
            "frame_w": int(w),
            "frame_h": int(h),
            "face_label": "Không thấy mặt",
            "emotion": None,
            "mode": "face",
        }

    mean = np.mean(np.stack(probs_list, axis=0), axis=0)
    playful = float(mean[1] + mean[2])  # happiness + surprise
    grumpy = float(mean[4] + mean[5] + mean[7])  # anger + disgust + contempt
    top = int(np.argmax(mean))
    emotion = EMOTIONS[top]
    is_violence = emotion in GRUMPY and grumpy >= 0.45
    violence_prob = grumpy
    if is_violence:
        pred_class = "violence"
        face_label = "Cộc bực"
    elif emotion in PLAYFUL or playful >= grumpy:
        pred_class = "non_violence"
        face_label = "Cười vui"
    else:
        pred_class = "non_violence"
        face_label = LABELS.get(emotion, emotion)

    x, y, bw, bh, fw, fh = box
    return {
        "n_frames": len(frames_rgb),
        "n_clips": 1,
        "violence_prob_mean": violence_prob,
        "non_violence_prob": float(1.0 - violence_prob),
        "violence_prob_max": violence_prob,
        "threshold": 0.45,
        "pred_label": int(is_violence),
        "pred_class": pred_class,
        "boxes": [{"x": x / fw, "y": y / fh, "w": bw / fw, "h": bh / fh, "score": float(mean[top])}],
        "frame_w": int(w),
        "frame_h": int(h),
        "face_label": face_label,
        "emotion": emotion,
        "mode": "face",
    }
