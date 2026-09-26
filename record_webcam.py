"""Quay video tu webcam va luu ra file .mp4 de test model violence detection.

Cach dung (chay tren may that, sau khi da kich hoat .venv):

    python record_webcam.py --output data\\webcam_tests\\test01.mp4 --seconds 10

Nhan phim 'q' de dung quay som. Sau khi co file video, chay:

    python -m src.infer --video data\\webcam_tests\\test01.mp4 --checkpoint checkpoints\\best.pt

Ket qua se in ra man hinh (violence_prob_mean, pred_class...) va luu vao
reports/outputs/infer/prediction.json + attention_best_clip.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def record(output: str, seconds: float, fps: float, cam_index: int, width: int, height: int) -> Path:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc webcam index={cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (actual_w, actual_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Khong tao duoc VideoWriter (kiem tra codec mp4v)")

    print(f"Dang quay -> {out_path} | {seconds}s @ {fps}fps | {actual_w}x{actual_h}")
    print("Nhan 'q' trong cua so video de dung som.")

    start = time.time()
    n_frames = 0
    try:
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                print("Mat frame, dung lai.")
                break
            writer.write(frame)
            n_frames += 1
            cv2.imshow("Recording - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"Da luu {n_frames} frame trong {elapsed:.1f}s -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quay video tu webcam de test model")
    parser.add_argument("--output", default="data/webcam_tests/test01.mp4", help="Duong dan file mp4 dau ra")
    parser.add_argument("--seconds", type=float, default=10.0, help="So giay quay (mac dinh 10s)")
    parser.add_argument("--fps", type=float, default=20.0, help="FPS muc tieu")
    parser.add_argument("--cam-index", type=int, default=0, help="Chi so webcam (0 = mac dinh)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    record(args.output, args.seconds, args.fps, args.cam_index, args.width, args.height)


if __name__ == "__main__":
    main()
