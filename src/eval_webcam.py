"""Evaluate checkpoints on labelled webcam clips the way the Camera page runs.

Each clip is downscaled like the browser frames, split into 16-frame windows
(stride 8) and scored with run_inference_frames (YOLO person crop + one model
pass). The clip probability is the mean over windows.

Event level (how the camera behaves): an alarm is a run of at least
--min-streak windows at or above the threshold, as the upload scorer counts
violent streaks. Reported: share of violent clips with an alarm, alarms per
hour of non-violent video, and seconds from violence onset to the alarm.
Onsets come from <data>/onsets.csv (columns file,onset_s); without it a
violent clip counts as violent from its first frame.

  python -m src.eval_webcam --checkpoint checkpoints/best2.pt checkpoints/best_webcam.pt
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.data.webcam_dataset import list_clips
from src.infer import load_detector, run_inference_frames
from src.utils.config import load_config, project_root


def read_frames(path, max_side: int = 640, jpeg_quality: int = 90) -> list[np.ndarray]:
    """Frames as the Camera page sends them: <=640px and JPEG quality 0.9.
    The JPEG step matters: the model is sensitive to JPEG artefacts."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        if jpeg_quality:
            bgr = cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1], cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def video_fps(path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def window_probs(model, cfg, device, frames, crop: bool, window: int = 16, stride: int = 8) -> np.ndarray:
    starts = list(range(0, max(1, len(frames) - window + 1), stride))
    return np.array([
        run_inference_frames(model, cfg, device, frames[s:s + window], with_boxes=crop)["violence_prob_mean"]
        for s in starts
    ])


def alarm_times(probs: np.ndarray, thr: float, min_streak: int, fps: float,
                window: int = 16, stride: int = 8) -> list[float]:
    """Second at which each alarm fires: the end of the window that completes the streak."""
    times, run = [], 0
    for i, hit in enumerate(probs >= thr):
        run = run + 1 if hit else 0
        if run == min_streak:
            times.append((i * stride + window) / fps)
    return times


def event_report(clips, probs, fps, n_frames, thr: float, min_streak: int, onsets: dict) -> dict:
    violent_alarms, delays, false_alarms, calm_seconds = 0, [], 0, 0.0
    for (path, label), pr, f, n in zip(clips, probs, fps, n_frames):
        times = alarm_times(pr, thr, min_streak, f)
        if label == 1:
            onset = onsets.get(path.name, 0.0)
            after = [t for t in times if t >= onset]
            if after:
                violent_alarms += 1
                delays.append(after[0] - onset)
        else:
            false_alarms += len(times)
            calm_seconds += n / f
    n_violent = sum(lbl for _, lbl in clips)
    return {
        "min_streak": min_streak,
        "event_recall": round(violent_alarms / n_violent, 4) if n_violent else None,
        "false_alarms_per_hour": round(false_alarms / (calm_seconds / 3600), 2) if calm_seconds else None,
        "false_alarms": false_alarms,
        "non_violent_minutes": round(calm_seconds / 60, 1),
        "median_delay_s": round(float(np.median(delays)), 2) if delays else None,
    }


def report(y: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_violence": round(precision_score(y, pred, zero_division=0), 4),
        "recall_violence": round(recall_score(y, pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def main() -> None:
    # Windows consoles default to cp1252, which cannot print Vietnamese.
    sys.stdout.reconfigure(encoding="utf-8")
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate webcam violence detection")
    parser.add_argument("--checkpoint", nargs="+", default=["checkpoints/best2.pt"])
    parser.add_argument("--data", default="data/webcam_dataset")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--threshold", type=float, default=None, help="default: the checkpoint's threshold")
    parser.add_argument("--no-crop", action="store_true", help="classify full frames instead of YOLO crops")
    parser.add_argument("--jpeg", type=int, default=90, help="JPEG quality like the Camera page; 0 = off")
    parser.add_argument("--min-streak", type=int, default=3, help="windows in a row that raise an alarm")
    parser.add_argument("--out", default="reports/outputs/eval_webcam.json")
    args = parser.parse_args()

    clips = list_clips(root / args.data, args.split)
    if not clips:
        raise SystemExit(f"Không có clip nào trong {root / args.data} (split={args.split})")
    y = np.array([label for _, label in clips])
    print(f"{len(clips)} clip ({int(y.sum())} bạo lực, {int(len(y) - y.sum())} không bạo lực), split={args.split}")
    frames = [read_frames(p, jpeg_quality=args.jpeg) for p, _ in clips]
    fps = [video_fps(p) for p, _ in clips]
    onsets_csv = root / args.data / "onsets.csv"
    onsets = {}
    if onsets_csv.is_file():
        import pandas as pd
        onsets = dict(pd.read_csv(onsets_csv).set_index("file")["onset_s"].astype(float))

    results = {}
    for ckpt in args.checkpoint:
        model, cfg, device = load_detector(root / ckpt, load_config(root / "configs/default.yaml"))
        windows = [window_probs(model, cfg, device, f, crop=not args.no_crop) for f in frames]
        probs = np.array([w.mean() for w in windows])
        thr = args.threshold if args.threshold is not None else float(cfg["infer"].get("threshold", 0.6))
        sweep = {f"{t:.2f}": report(y, probs, float(t))["f1_macro"] for t in np.arange(0.3, 0.91, 0.05)}
        wrong = [f"{p.parent.name}/{p.name} p={pr:.2f}" for (p, lbl), pr in zip(clips, probs) if int(pr >= thr) != lbl]
        events = event_report(clips, windows, fps, [len(f) for f in frames], thr, args.min_streak, onsets)
        results[ckpt] = {"at_threshold": report(y, probs, thr), "events": events,
                         "f1_by_threshold": sweep, "misclassified": wrong}
        print(f"\n== {ckpt}")
        print(json.dumps(results[ckpt]["at_threshold"], indent=2))
        print("Su kien:", json.dumps(events, ensure_ascii=False))
        best = max(sweep, key=sweep.get)
        print(f"Ngưỡng tốt nhất trên tập này: {best} (F1={sweep[best]})")
        for w in wrong:
            print("  sai:", w)

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
