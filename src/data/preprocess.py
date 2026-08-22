from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}


def list_videos(raw_dir: str | Path, classes: list[str]) -> list[dict]:
    records = []
    raw_dir = Path(raw_dir)
    for label, cls in enumerate(classes):
        folder = raw_dir / cls
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTS:
                records.append(
                    {
                        "video_id": path.stem,
                        "src": str(path),
                        "label": label,
                        "class_name": cls,
                    }
                )
    return records


def extract_frames(video_path: str | Path, out_dir: str | Path, max_frames: int | None = None) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{count:06d}.jpg"), frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    return count


def sample_indices(n_frames: int, num_frames: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("Video has no frames")
    if n_frames >= num_frames:
        return np.linspace(0, n_frames - 1, num_frames, dtype=int)
    idxs = np.arange(n_frames)
    pad = np.full(num_frames - n_frames, n_frames - 1, dtype=int)
    return np.concatenate([idxs, pad])


def read_video_sampled(video_path: str | Path, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Empty video: {video_path}")
        idxs = sample_indices(len(frames), num_frames)
        return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idxs]

    idxs = sample_indices(total, num_frames)
    sampled = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not sampled:
        raise RuntimeError(f"Failed to sample frames: {video_path}")
    while len(sampled) < num_frames:
        sampled.append(sampled[-1])
    return sampled[:num_frames]


def load_frame_dir(frame_dir: str | Path, num_frames: int) -> list[np.ndarray]:
    paths = sorted(Path(frame_dir).glob("*.jpg")) + sorted(Path(frame_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")
    idxs = sample_indices(len(paths), num_frames)
    frames = []
    for i in idxs:
        img = cv2.imread(str(paths[i]))
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError(f"Cannot decode frames in {frame_dir}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def sliding_windows(n_frames: int, window: int, stride: int) -> list[np.ndarray]:
    if n_frames <= window:
        return [sample_indices(n_frames, window)]
    starts = list(range(0, n_frames - window + 1, stride))
    if starts[-1] != n_frames - window:
        starts.append(n_frames - window)
    return [np.arange(s, s + window) for s in starts]
