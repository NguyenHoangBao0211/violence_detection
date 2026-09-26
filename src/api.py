"""Web app FastAPI: upload, camera, history for CNN-LSTM violence detection."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from src.history import add_detection, delete_all_detections, delete_detection, get_detection, init_db, list_detections, save_video_copy
from src.infer import load_detector, run_inference, run_inference_frames
from src.utils.config import load_config, project_root

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".webm"}

root = project_root()
web_dir = root / "web"
templates = Jinja2Templates(directory=str(web_dir / "templates"))

cfg = load_config(root / "configs/default.yaml")
ckpt = root / "checkpoints" / "best2.pt"
webcam_ckpt = root / "checkpoints" / "best2.pt"
if not ckpt.exists():
    ckpt = root / "checkpoints" / "MoBiLSTM_model.h5"
if not ckpt.exists():
    for alt in (
        Path.home() / "Downloads" / "best (1).pt",
        Path.home() / "Downloads" / "best.pt",
    ):
        if alt.exists():
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt, ckpt)
            break

_model = _cfg = _device = None
_model_error: str | None = None
_webcam_model = _webcam_cfg = _webcam_device = None
_webcam_model_error: str | None = None
_backend = "cnn_lstm"
if ckpt.exists():
    try:
        _model, _cfg, _device = load_detector(ckpt, cfg)
    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
else:
    _model_error = f"Missing checkpoint: {ckpt}"

if webcam_ckpt.exists():
    try:
        _webcam_model, _webcam_cfg, _webcam_device = load_detector(webcam_ckpt, cfg)
        # The checkpoint's saved threshold (0.6) was tuned on RWF-2000/AIRTLab,
        # where "non_violence" clips are mostly low-motion. Ordinary webcam
        # movement (walking, waving) sits well above that on this model, so
        # raise the bar for a live "violence" call. This is a stopgap, not a
        # real fix — the model itself needs more non-violent-but-moving
        # webcam footage to stop conflating motion with violence.
        _webcam_cfg["infer"]["threshold"] = 0.75
    except Exception as exc:  # noqa: BLE001
        _webcam_model_error = str(exc)
else:
    _webcam_model_error = f"Missing checkpoint: {webcam_ckpt}"

init_db()
storage = root / "storage"
storage.mkdir(parents=True, exist_ok=True)
(storage / "alerts").mkdir(parents=True, exist_ok=True)
(storage / "attention").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Violence Detection Web (CNN-LSTM)", version="2.1")
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")
app.mount("/media", StaticFiles(directory=str(storage)), name="media")


def _ensure_model(kind: str = "upload") -> None:
    model = _model if kind == "upload" else _webcam_model
    error = _model_error if kind == "upload" else _webcam_model_error
    if model is None:
        raise HTTPException(503, error or "Model chưa sẵn sàng")


def _rel_media(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    try:
        return "/media/" + path.relative_to(storage).as_posix()
    except ValueError:
        dest = storage / "attention" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return "/media/attention/" + dest.name


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active": "upload"})


@app.get("/live", response_class=HTMLResponse)
def live(request: Request):
    return templates.TemplateResponse(request, "live.html", {"active": "live"})


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    rows = list_detections(200)
    for r in rows:
        vp = r.get("video_path")
        r["video_url"] = _rel_media(Path(vp) if vp else None)
        r["prob_pct"] = f"{float(r['prob']) * 100:.1f}"
        try:
            dt = datetime.fromisoformat(r["created_at"])
            r["time_display"] = dt.strftime("%H:%M:%S")
            r["date_display"] = dt.strftime("%d/%m/%Y")
        except (TypeError, ValueError):
            r["time_display"] = r.get("created_at") or "--"
            r["date_display"] = ""
    return templates.TemplateResponse(
        request,
        "history.html",
        {"active": "history", "rows": rows, "total": len(rows)},
    )


@app.get("/api/history")
def api_history(limit: int = 100) -> dict:
    return {"items": list_detections(limit)}


def _remove_record_files(row: dict) -> None:
    for key in ("video_path", "attention_path"):
        value = row.get(key)
        if value:
            path = Path(value)
            if path.exists() and path.is_file():
                path.unlink()


@app.post("/history/{det_id}/delete")
def delete_history_item(det_id: int):
    row = delete_detection(det_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy bản ghi")
    _remove_record_files(row)
    return {"ok": True, "id": det_id}


@app.post("/history/delete-all")
def delete_history_all():
    rows = delete_all_detections()
    for row in rows:
        _remove_record_files(row)
    return {"ok": True, "deleted": len(rows)}


@app.get("/history/{det_id}/video")
def history_video(det_id: int):
    row = get_detection(det_id)
    if not row or not row.get("video_path"):
        raise HTTPException(404, "Không có video")
    path = Path(row["video_path"])
    if not path.exists():
        raise HTTPException(404, "File video không còn")
    return FileResponse(path)


@app.get("/json/version")
def json_version() -> dict:
    """Trả lời probe của Chrome DevTools / extension — tránh 404 nhiễu log."""
    return {"ok": True, "app": "violence-detection", "backend": _backend}


@app.get("/health")
def health() -> dict:
    return {
        "ok": _model is not None,
        "backend": _backend,
        "device": str(_device) if _device is not None else None,
        "checkpoint": str(ckpt),
        "error": _model_error,
        "db": str(root / "data" / "history.db"),
    }


def _prediction_payload(result: dict, *, history_id: int | None = None, extra: dict | None = None) -> dict:
    pred_class = result["pred_class"]
    prob = float(result["violence_prob_mean"])
    # A violent event can occupy only a short part of the video.  The
    # full-video mean is then diluted by normal frames and may contradict
    # the event-based label shown to the user.  Show the strongest detected
    # window as confidence when the final label is violence.
    if pred_class == "violence" and result.get("violence_prob_max") is not None:
        prob = float(result["violence_prob_max"])
    non_prob = float(result.get("non_violence_prob", 1.0 - prob))
    non_prob = 1.0 - prob if pred_class == "violence" else non_prob
    pred = "Violence" if pred_class == "violence" else "NonViolence"
    conf = float(prob if pred_class == "violence" else non_prob)
    alert = pred_class == "violence"
    payload = {
        "prediction": pred,
        "confidence": conf,
        "pred_class": pred_class,
        "violence_prob_mean": prob,
        "violence_prob_max": result.get("violence_prob_max"),
        "non_violence_prob": non_prob,
        "violence_pct": round(prob * 100, 1),
        "non_violence_pct": round(non_prob * 100, 1),
        "alert": alert,
        "alert_message": "CẢNH BÁO BẠO LỰC" if alert else None,
        "n_frames": result.get("n_frames"),
        "n_clips": result.get("n_clips"),
        "threshold": result.get("threshold", 0.5),
        "boxes": result.get("boxes") or [],
        "history_id": history_id,
        "backend": _backend,
    }
    if extra:
        payload.update(extra)
    return payload


def _decode_frames(raw_frames: list[bytes]) -> list[np.ndarray]:
    rgb_frames: list[np.ndarray] = []
    for raw in raw_frames:
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return rgb_frames


def _save_live_video(rgb_frames: list[np.ndarray], result: dict) -> Path:
    day_dir = storage / "alerts"
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:10]
    snap = day_dir / f"live_{stamp}_{result['pred_class']}.mp4"
    first = cv2.cvtColor(rgb_frames[0], cv2.COLOR_RGB2BGR)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(snap), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h))
    try:
        for rgb in rgb_frames:
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for box in result.get("boxes") or []:
                x1, y1 = int(box["x"] * w), int(box["y"] * h)
                x2, y2 = int((box["x"] + box["w"]) * w), int((box["y"] + box["h"]) * h)
                color = (0, 0, 220) if result["pred_class"] == "violence" else (40, 180, 60)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            writer.write(frame)
    finally:
        writer.release()
    return snap


@app.post("/predict/live")
async def predict_live(
    frames: list[UploadFile] = File(...),
    save_history: str = Form("0"),
) -> dict:
    """Live webcam → CNN–LSTM + bounding box."""
    _ensure_model("webcam")
    if not frames:
        raise HTTPException(400, "Thiếu frame camera")

    raw_frames = [await up.read() for up in frames]
    rgb_frames = await run_in_threadpool(_decode_frames, raw_frames)
    if len(rgb_frames) < 2:
        raise HTTPException(400, "Không đọc được frame (cần ≥2 ảnh JPEG)")

    try:
        result = await run_in_threadpool(
            partial(run_inference_frames, _webcam_model, _webcam_cfg, _webcam_device, rgb_frames, with_boxes=True)
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc

    hist_id = None
    snap_url = None
    # Client quyết định thời điểm kết thúc một sự kiện và gửi save_history=1.
    # Không lưu từng lần dự đoán realtime để tránh tạo nhiều bản ghi trùng nhau.
    if save_history in {"1", "true", "yes"}:
        snap = await run_in_threadpool(_save_live_video, rgb_frames, result)
        _prob_v = float(result["violence_prob_mean"])
        _conf = _prob_v if result["pred_class"] == "violence" else (1.0 - _prob_v)
        hist_id = add_detection(
            source="webcam",
            pred_class=result["pred_class"],
            prob=_conf,
            video_path=snap,
            note="live_frames",
        )
        snap_url = _rel_media(snap)

    return _prediction_payload(result, history_id=hist_id, extra={"snapshot_url": snap_url})


@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    video: UploadFile = File(None),
    source: str = Form("upload"),
) -> dict:
    model_kind = "webcam" if (source or "upload").strip().lower() in {"webcam", "rtsp"} else "upload"
    _ensure_model(model_kind)
    upload = file or video
    if upload is None:
        raise HTTPException(400, "No video file found in the request")
    suffix = Path(upload.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXTS:
        raise HTTPException(400, f"Định dạng không hỗ trợ: {suffix}")
    source = (source or "upload").strip().lower()
    if source not in {"upload", "webcam", "rtsp"}:
        source = "upload"

    tmp = Path(tempfile.gettempdir()) / f"vd-{uuid.uuid4().hex}{suffix}"
    try:
        tmp.write_bytes(await upload.read())
        out = root / "reports" / "outputs" / "api" / tmp.stem
        selected_model = _webcam_model if model_kind == "webcam" else _model
        selected_cfg = _webcam_cfg if model_kind == "webcam" else _cfg
        selected_device = _webcam_device if model_kind == "webcam" else _device
        result = await run_in_threadpool(run_inference, selected_model, selected_cfg, selected_device, tmp, out)

        pred_class = result["pred_class"]
        prob_v = float(result["violence_prob_mean"])
        prob = prob_v if pred_class == "violence" else (1.0 - prob_v)
        saved = save_video_copy(tmp, pred_class, suffix)
        attn = out / "attention_best_clip.png"
        attn_path = None
        if attn.exists():
            attn_dest = storage / "attention" / f"{saved.stem}_attn.png"
            shutil.copy2(attn, attn_dest)
            attn_path = attn_dest

        hist_id = add_detection(
            source=source,
            pred_class=pred_class,
            prob=prob,
            video_path=saved,
            attention_path=attn_path,
            note=upload.filename,
        )
        row = get_detection(hist_id)
        return _prediction_payload(
            result,
            history_id=hist_id,
            extra={
                "video": upload.filename,
                "video_url": _rel_media(saved),
                "attention_url": _rel_media(attn_path),
                "created_at": row["created_at"] if row else None,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)
