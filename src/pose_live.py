"""Webcam violence score from YOLO pose, not from the frame classifier.

A strike is a wrist that moves fast relative to the torso while the arm
straightens. Sitting still, and a hand that only moves near the face, stay
below the live threshold.
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from src.utils.config import project_root

os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root() / "storage" / "ultralytics"))

# COCO: shoulder, elbow, wrist for each arm.
_ARMS = ((5, 7, 9), (6, 8, 10))
_KP_MIN = 0.35
_FRAME_DT = 0.030
_THRESHOLD = 0.6

_pose_model = None
_pose_error: str | None = None
_pose_lock = threading.Lock()


def pose_error() -> str | None:
    return _pose_error


def warm_pose():
    """Load YOLO pose once so the first camera frame is not the download."""
    return _load_pose()


def _load_pose():
    global _pose_model, _pose_error
    if _pose_model is not None or _pose_error is not None:
        return _pose_model
    weights = project_root() / "checkpoints" / "yolo11n-pose.pt"
    try:
        from ultralytics import YOLO
    except ImportError:
        _pose_error = "Chua cai ultralytics, khong chay duoc YOLO pose."
        return None
    try:
        if weights.exists():
            _pose_model = YOLO(str(weights))
        else:
            downloaded = YOLO("yolo11n-pose.pt")
            source = getattr(downloaded, "ckpt_path", None)
            if source:
                weights.parent.mkdir(parents=True, exist_ok=True)
                if os.path.abspath(source) != os.path.abspath(weights):
                    import shutil
                    shutil.copy2(source, weights)
                    _pose_model = YOLO(str(weights))
                else:
                    _pose_model = downloaded
            else:
                _pose_model = downloaded
    except Exception as exc:  # noqa: BLE001
        _pose_error = f"Khong tai duoc yolo11n-pose.pt: {exc}"
        return None
    return _pose_model


def _point(kpts: np.ndarray, index: int):
    x, y, conf = kpts[index]
    if conf < _KP_MIN:
        return None
    return np.array([float(x), float(y)])


def _torso(kpts: np.ndarray, box_h: float) -> float:
    left, right = _point(kpts, 5), _point(kpts, 6)
    if left is not None and right is not None:
        span = float(np.linalg.norm(left - right))
        if span > 0.02:
            return span
    return max(0.05, float(box_h) * 0.35)


def score_track(observations: list[dict]) -> float:
    """0–1 violence score for one person tracked across a short clip."""
    speeds: list[float] = []
    extensions: list[float] = []
    near_face: list[bool] = []
    for prev, cur in zip(observations, observations[1:]):
        dt = max(_FRAME_DT, float(cur["t"]) - float(prev["t"]))
        torso = _torso(cur["kpts"], cur["box"]["h"])
        nose = _point(cur["kpts"], 0)
        for shoulder_i, elbow_i, wrist_i in _ARMS:
            start = _point(prev["kpts"], wrist_i)
            end = _point(cur["kpts"], wrist_i)
            shoulder = _point(cur["kpts"], shoulder_i)
            elbow = _point(cur["kpts"], elbow_i)
            if start is None or end is None:
                continue
            speeds.append(float(np.linalg.norm(end - start)) / torso / dt)
            if shoulder is not None and elbow is not None:
                reach = float(np.linalg.norm(end - shoulder))
                chain = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(end - elbow))
                extensions.append(reach / chain if chain > 1e-4 else 0.0)
            else:
                extensions.append(0.0)
            if nose is None:
                near_face.append(False)
            else:
                near_face.append(float(np.linalg.norm(end - nose)) < 0.55 * torso)
    if not speeds:
        return 0.08
    peak = int(np.argmax(speeds))
    peak_speed = speeds[peak]
    extension = extensions[peak]
    if peak_speed < 1.3:
        return float(min(0.2, 0.05 + 0.08 * peak_speed))
    punch = min(1.0, (peak_speed - 1.3) / 3.2) * min(1.0, max(0.0, (extension - 0.72) / 0.22))
    if near_face[peak]:
        punch *= 0.35
    fast_hits = sum(1 for speed, ext in zip(speeds, extensions) if speed > 2.2 and ext > 0.78)
    clear_strike = peak_speed > 3.5 and extension > 0.85 and not near_face[peak]
    if fast_hits < 2 and not clear_strike:
        punch *= 0.55
    return float(min(0.97, 0.12 + 0.85 * punch))


def _people(result, when: float) -> list[dict]:
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return []
    xy = result.keypoints.xyn.cpu().numpy()
    conf = result.keypoints.conf
    conf = np.ones((len(xy), xy.shape[1])) if conf is None else conf.cpu().numpy()
    boxes = result.boxes.xyxyn.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    people = []
    for index, box in enumerate(boxes):
        kpts = np.concatenate([xy[index], conf[index][:, None]], axis=1)
        x1, y1, x2, y2 = [float(v) for v in box]
        people.append({
            "t": when,
            "kpts": kpts,
            "box": {
                "x": max(0.0, x1),
                "y": max(0.0, y1),
                "w": max(0.0, x2 - x1),
                "h": max(0.0, y2 - y1),
                "score": float(scores[index]),
            },
        })
    return people


def _assign(tracks: list[list[dict]], people: list[dict]) -> None:
    used: set[int] = set()
    for track in tracks:
        last = track[-1]["box"]
        origin = np.array([last["x"] + last["w"] / 2, last["y"] + last["h"] / 2])
        best = None
        best_distance = 0.25
        for index, person in enumerate(people):
            if index in used:
                continue
            box = person["box"]
            center = np.array([box["x"] + box["w"] / 2, box["y"] + box["h"] / 2])
            distance = float(np.linalg.norm(center - origin))
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            used.add(best)
            track.append(people[best])
    for index, person in enumerate(people):
        if index not in used:
            tracks.append([person])


def run_pose_frames(frames: list[np.ndarray]) -> dict:
    """Score the newest webcam clip and return boxes plus the last skeleton."""
    import time

    started = time.perf_counter()
    model = _load_pose()
    if model is None:
        raise RuntimeError(_pose_error or "YOLO pose chua san sang")
    count = len(frames)
    sample_n = min(6, count)
    indices = np.linspace(0, count - 1, sample_n).astype(int)
    indices[-1] = count - 1
    indices = sorted({int(i) for i in indices})
    bgrs = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR) for i in indices]
    with _pose_lock:
        results = model.predict(
            source=bgrs,
            conf=0.35,
            imgsz=480,
            verbose=False,
            device=0 if _cuda() else "cpu",
        )
    tracks: list[list[dict]] = []
    for index, result in zip(indices, results):
        _assign(tracks, _people(result, index * _FRAME_DT))
    violence = max((score_track(track) for track in tracks), default=0.08)
    when = indices[-1] * _FRAME_DT
    visible = [track[-1] for track in tracks if abs(track[-1]["t"] - when) < 1e-6]
    poses = []
    for person in visible:
        points = []
        for x, y, conf in person["kpts"]:
            if conf < _KP_MIN:
                points.append(None)
            else:
                points.append([round(float(x), 4), round(float(y), 4), round(float(conf), 3)])
        poses.append({"keypoints": points})
    label = int(violence >= _THRESHOLD)
    return {
        "n_frames": count,
        "n_clips": 1,
        "violence_prob_mean": violence,
        "non_violence_prob": 1.0 - violence,
        "violence_prob_max": violence,
        "threshold": _THRESHOLD,
        "pred_label": label,
        "pred_class": "violence" if label else "non_violence",
        "boxes": [person["box"] for person in visible],
        "poses": poses,
        "frame_w": int(frames[-1].shape[1]),
        "frame_h": int(frames[-1].shape[0]),
        "movinet_prob": None,
        "peak_frame": count - 1,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "mode": "pose",
    }


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False
