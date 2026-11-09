"""Web app FastAPI: upload, camera, history for CNN-LSTM violence detection."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import threading
import time
from collections import defaultdict, deque
import tempfile
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from src.history import add_detection, alerts_dir, delete_all_detections, delete_detection, get_detection, init_db, list_detections, save_video_copy
from src.mail_alert import send_violence_alert
from src import movinet_veto
from src.infer import load_detector, render_annotated_video, run_inference, run_inference_frames
from src.utils.config import load_config, project_root

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".webm"}
# Longest upload (file or link) the server will analyse.
MAX_UPLOAD_SECONDS = 20 * 60
# Largest uploaded file accepted (bytes); streaming stops and returns 413 above it.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
# Browsers send these for video files; octet-stream covers clients that omit the type.
VIDEO_MIME_PREFIXES = ("video/",)
VIDEO_MIME_EXTRA = {"application/octet-stream", "application/x-matroska", ""}
# Requests per client IP in a 60 s window (429 above it). /predict/live is sent
# every ~220 ms by the camera page, so it gets a higher limit.
RATE_LIMITS = {"/predict": 10, "/predict/url": 10, "/predict/live": 600, "/events/webcam": 60,
               "/reports/pdf": 20, "/dataset/clip": 120}

root = project_root()
web_dir = root / "web"
templates = Jinja2Templates(directory=str(web_dir / "templates"))
# Cache-busting query from the file's mtime, so an edited ui.js or app.css is
# never served stale (a stale ui.js shows raw i18n keys like "segments_label").
templates.env.globals["asset_v"] = lambda rel: int((web_dir / "static" / rel).stat().st_mtime)

cfg = load_config(root / "configs/default.yaml")
ckpt = root / "checkpoints" / "resnet50_bilstm_attention.pt"
if not ckpt.exists():
    ckpt = root / "checkpoints" / "best2.pt"
# The camera prefers the MobileNetV2–BiLSTM–Attention model (~13x fewer FLOPs
# per frame, keeps up with the live feed on CPU); upload keeps ResNet50. Without
# that checkpoint both pages share the ResNet50 model.
webcam_trial_path = root / "configs" / "webcam_trial.json"
webcam_trial = json.loads(webcam_trial_path.read_text(encoding="utf-8")) if webcam_trial_path.exists() else {}
webcam_ckpt = root / "checkpoints" / webcam_trial.get("checkpoint", "mobilenet_bilstm_attention.pt")
if not webcam_ckpt.exists():
    webcam_ckpt = ckpt
# Violence threshold for the Camera page only (checkpoint default is 0.6).
WEBCAM_THRESHOLD = 0.5

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
        if _model is not None and webcam_ckpt.resolve() == Path(ckpt).resolve():
            _webcam_model, _webcam_cfg, _webcam_device = _model, _cfg, _device
        else:
            _webcam_model, _webcam_cfg, _webcam_device = load_detector(webcam_ckpt, cfg)
        # The checkpoint's saved threshold (0.6) was tuned on RWF-2000/AIRTLab,
        # where "non_violence" clips are mostly low-motion. Ordinary webcam
        # movement (walking, waving) sits well above that on this model, so
        # raise the bar for a live "violence" call. This is a stopgap, not a
        # real fix — the model itself needs more non-violent-but-moving
        # webcam footage to stop conflating motion with violence.
        # Only when the camera borrows the upload model: a dedicated webcam
        # checkpoint carries the threshold chosen on its own validation set.
        # Upload keeps its threshold because the config is copied first.
        if _webcam_cfg is _cfg:
            _webcam_cfg = dict(_cfg)
            _webcam_cfg["infer"] = dict(_cfg["infer"])
            _webcam_cfg["infer"]["threshold"] = WEBCAM_THRESHOLD
    except Exception as exc:  # noqa: BLE001
        _webcam_model_error = str(exc)
else:
    _webcam_model_error = f"Missing checkpoint: {webcam_ckpt}"

webcam_ensemble = webcam_ckpt != ckpt and webcam_trial.get("ensemble", True)
webcam_mode = (_webcam_cfg or {}).get("model_type") or f"{(_webcam_cfg or {}).get("model", {}).get("backbone", "?")}_bilstm_attention"

# Optional upload checkpoint; keep the existing live ensemble model.
upload_ckpt = root / "checkpoints" / webcam_trial.get("upload_checkpoint", "mobilenet_upload_base.pt")
_upload_model, _upload_cfg, _upload_device = _model, _cfg, _device
_upload_error = _model_error
if upload_ckpt.exists():
    try:
        if upload_ckpt == webcam_ckpt and _webcam_model is not None:
            _upload_model, _upload_cfg, _upload_device = _webcam_model, _webcam_cfg, _webcam_device
        else:
            _upload_model, _upload_cfg, _upload_device = load_detector(upload_ckpt, cfg)
        _upload_error = None
    except Exception as exc:
        _upload_model = None
        _upload_error = str(exc)
else:
    upload_ckpt = ckpt

# Load the webcam second-opinion model now rather than on the first request.
movinet_veto.get_model()

init_db()
storage = root / "storage"
storage.mkdir(parents=True, exist_ok=True)
(storage / "alerts").mkdir(parents=True, exist_ok=True)
(storage / "attention").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Violence Detection Web (CNN-LSTM)", version="2.1")
templates.env.globals["app_version"] = app.version
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")
app.mount("/media", StaticFiles(directory=str(storage)), name="media")

try:  # same .env as the e-mail settings
    from dotenv import load_dotenv
    load_dotenv(root / ".env")
except ImportError:
    pass
# Login for every page, API route and /media file. Set APP_USER and APP_PASSWORD
# in .env; /health stays open for the deployment health check.
APP_USER = os.environ.get("APP_USER", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
AUTH_OPEN_PATHS = {"/health"}
if not (APP_USER and APP_PASSWORD):
    print("WARNING: APP_USER/APP_PASSWORD not set in .env - login is disabled.")

_rate_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _rate_limited(ip: str, path: str) -> bool:
    limit = RATE_LIMITS.get(path)
    if limit is None:
        return False
    now = time.monotonic()
    with _rate_lock:
        hits = _rate_hits[(ip, path)]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            return True
        hits.append(now)
    return False


@app.middleware("http")
async def security_guard(request: Request, call_next):
    path = request.url.path
    if APP_USER and APP_PASSWORD and path not in AUTH_OPEN_PATHS:
        ok = False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(user, APP_USER) and secrets.compare_digest(pwd, APP_PASSWORD)
            except Exception:  # noqa: BLE001 - malformed header means not logged in
                ok = False
        if not ok:
            return Response("Authentication required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Violence Detection"'})
    if request.method == "POST" and _rate_limited(request.client.host if request.client else "?", path):
        return JSONResponse({"detail": "Quá nhiều yêu cầu, vui lòng thử lại sau."}, status_code=429)
    return await call_next(request)


def _copy_limited(src, dst, limit: int = MAX_UPLOAD_BYTES) -> int:
    total = 0
    while True:
        chunk = src.read(1024 * 1024)
        if not chunk:
            return total
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"Tệp vượt quá {limit // (1024 * 1024)} MB")
        dst.write(chunk)


def _check_video_content(path: Path) -> None:
    """Reject files whose content is not a decodable video, whatever the extension says."""
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not cap.isOpened() or not ok or frame is None:
            raise HTTPException(400, "Nội dung tệp không phải video hợp lệ")
    finally:
        cap.release()


def _ensure_model(kind: str = "upload") -> None:
    if kind == "webcam" and webcam_ensemble and _model is None:
        raise HTTPException(503, _model_error or "Secondary webcam model unavailable")
    model = _upload_model if kind == "upload" else _webcam_model
    error = _upload_error if kind == "upload" else _webcam_model_error
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


@app.get("/video-link", response_class=HTMLResponse)
def video_link_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active": "video_link", "link_mode": True})


@app.get("/live", response_class=HTMLResponse)
def live(request: Request):
    return templates.TemplateResponse(request, "live.html", {"active": "live", "window_frames": (_webcam_cfg or {}).get("infer", {}).get("window_frames", 16)})


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


@app.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request):
    rows = list_detections(100000, include_hidden=True)
    violent = [r for r in rows if r["pred_class"] == "violence"]
    today = datetime.now().date()
    days = []
    for back in range(6, -1, -1):
        day = today.fromordinal(today.toordinal() - back).isoformat()
        day_rows = [r for r in rows if str(r["created_at"]).startswith(day)]
        n_violent = sum(r["pred_class"] == "violence" for r in day_rows)
        days.append({"date": datetime.fromisoformat(day).strftime("%d/%m"), "total": len(day_rows), "violent": n_violent})
    peak = max((d["violent"] for d in days), default=0) or 1
    for d in days:
        d["pct"] = round(100 * d["violent"] / peak)
    sources = {}
    for r in violent:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    summary = {
        "total": len(rows),
        "violent": len(violent),
        "non_violent": len(rows) - len(violent),
        "today": days[-1]["violent"],
        "avg_conf": f"{100 * sum(float(r['prob']) for r in violent) / len(violent):.1f}" if violent else "—",
        "webcam": sources.get("webcam", 0),
        "upload": sources.get("upload", 0),
    }
    return templates.TemplateResponse(request, "stats.html", {"active": "stats", "s": summary, "days": days})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    info = {
        "checkpoint": upload_ckpt.name,
        "webcam_checkpoint": webcam_ckpt.name,
        "webcam_mode": webcam_mode,
        "device": str(_device) if _device is not None else "—",
        "threshold": (_webcam_cfg or {}).get("infer", {}).get("threshold", WEBCAM_THRESHOLD),
        "movinet": movinet_veto.get_model() is not None,
        "movinet_error": movinet_veto.error(),
        "model_ok": _upload_model is not None,
        "model_error": _upload_error,
    }
    return templates.TemplateResponse(request, "settings.html", {"active": "settings", "info": info})


@app.get("/api/history")
def api_history(limit: int = 100) -> dict:
    items = list_detections(limit)
    for r in items:
        vp = r.get("video_path")
        r["video_url"] = _rel_media(Path(vp) if vp else None)
    return {"items": items}


@app.post("/history/{det_id}/delete")
def delete_history_item(det_id: int):
    row = delete_detection(det_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy bản ghi")
    return {"ok": True, "id": det_id}


@app.post("/history/delete-all")
def delete_history_all():
    rows = delete_all_detections()
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
        "ok": _upload_model is not None,
        "backend": _backend,
        "device": str(_upload_device) if _upload_device is not None else None,
        "upload_model_type": (_upload_cfg or {}).get("model_type"),
        "checkpoint": str(upload_ckpt),
        "error": _upload_error,
        "webcam_checkpoint": str(webcam_ckpt),
        "webcam_secondary_checkpoint": str(ckpt) if webcam_ensemble else None,
        "webcam_ensemble": "mean_50_50" if webcam_ensemble else "single",
        "webcam_error": _webcam_model_error,
        "webcam_mode": webcam_mode,
        "webcam_movinet_veto": movinet_veto.get_model() is not None,
        "webcam_movinet_error": movinet_veto.error(),
        "db": str(root / "data" / "history.db"),
    }


def _prediction_payload(result: dict, *, history_id: int | None = None, extra: dict | None = None) -> dict:
    pred_class = result["pred_class"]
    prob = float(result["violence_prob_mean"])
    # Upload confidence is the share of scenes in the winning class.
    ratio = result.get("scene_violence_ratio")
    if ratio is not None:
        ratio = float(ratio)
        prob = ratio
        non_prob = 1.0 - ratio
    elif pred_class == "violence" and result.get("violence_prob_max") is not None:
        prob = float(result["violence_prob_max"])
        non_prob = 1.0 - prob
    else:
        non_prob = float(result.get("non_violence_prob", 1.0 - prob))
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
        "poses": result.get("poses") or [],
        "latency_ms": result.get("latency_ms"),
        "movinet_prob": result.get("movinet_prob"),
        "model_probs": result.get("model_probs"),
        "peak_frame": result.get("peak_frame"),
        "history_id": history_id,
        "backend": _backend,
        "face_label": result.get("face_label"),
        "emotion": result.get("emotion"),
        "mode": result.get("mode"),
        "duration_s": result.get("duration_s"),
        "segments": result.get("segments"),
        "clip_probs": result.get("clip_probs"),
        "clip_times": result.get("clip_times"),
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


def _save_event_video(rgb_frames: list[np.ndarray], fps: float) -> Path:
    """Write a webcam event as WebM/VP8, which browsers play in the history page
    (OpenCV's mp4v is MPEG-4 Part 2, which browsers do not decode)."""
    day_dir = storage / "alerts"
    day_dir.mkdir(parents=True, exist_ok=True)
    dest = day_dir / f"event_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}_violence.webm"
    h, w = rgb_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
    try:
        for rgb in rgb_frames:
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return dest


@app.post("/events/webcam")
async def save_webcam_event(
    frames: list[UploadFile] = File(...),
    fps: float = Form(15.0),
    start_s: float = Form(...),
    end_s: float = Form(...),
    start_clock: str = Form(""),
    end_clock: str = Form(""),
    peak_prob: float = Form(1.0),
) -> dict:
    """Save one webcam violence event (start → end) sent by the Camera page.

    The page already decided the event from its realtime predictions and
    timestamps each frame, so nothing is re-classified here.
    """
    raw_frames = [await up.read() for up in frames]
    rgb_frames = await run_in_threadpool(_decode_frames, raw_frames)
    if len(rgb_frames) < 2:
        raise HTTPException(400, "Không đọc được frame sự kiện")
    fps = float(min(60.0, max(1.0, fps)))
    video = await run_in_threadpool(_save_event_video, rgb_frames, fps)
    duration = max(0.0, end_s - start_s)
    note = f"⏱ {start_clock[:12]} → {end_clock[:12]} ({duration:.1f}s) · giây {start_s:.1f}–{end_s:.1f}"
    hist_id = add_detection(
        source="webcam",
        pred_class="violence",
        prob=float(min(1.0, max(0.0, peak_prob))),
        video_path=video,
        note=note,
    )
    mail_alert = await run_in_threadpool(
        partial(send_violence_alert, note=note, peak_prob=float(min(1.0, max(0.0, peak_prob))),
                history_id=hist_id, source="webcam")
    )
    return {
        "ok": True,
        "history_id": hist_id,
        "video_url": _rel_media(video),
        "duration_s": duration,
        "note": note,
        "mail_alert": mail_alert,
    }


@app.post("/predict/live")
async def predict_live(
    frames: list[UploadFile] = File(...),
    save_history: str = Form("0"),
) -> dict:
    """Live webcam: ResNet50–BiLSTM–Attention trên toàn cảnh và vùng người."""
    _ensure_model("webcam")
    if not frames:
        raise HTTPException(400, "Thiếu frame camera")

    raw_frames = [await up.read() for up in frames]
    rgb_frames = await run_in_threadpool(_decode_frames, raw_frames)
    if len(rgb_frames) < 2:
        raise HTTPException(400, "Không đọc được frame (cần ≥2 ảnh JPEG)")

    try:
        result = await run_in_threadpool(
            partial(run_inference_frames, _webcam_model, _webcam_cfg, _webcam_device, rgb_frames,
                    with_boxes=True, with_movinet=webcam_ensemble,
                    secondary_detector=(_model, _cfg, _device) if webcam_ensemble else None)
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


DATASET_DIR = root / "data" / "webcam_dataset"
DATASET_LABELS = ("violence", "non_violence")


def _save_dataset_clip(src: Path, label: str, duration_ms: float) -> Path:
    """Re-encode a browser WebM clip to MP4 with a real frame count and fps.

    MediaRecorder WebM files often report no frame count, which breaks the
    training loader (it seeks by frame index), so decode and write again.
    """
    frames = []
    cap = cv2.VideoCapture(str(src))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 16:
        raise HTTPException(400, f"Clip quá ngắn ({len(frames)} khung), cần ≥16 khung")
    fps = len(frames) / (duration_ms / 1000.0) if duration_ms > 0 else 30.0
    fps = float(min(60.0, max(10.0, round(fps))))
    h, w = frames[0].shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
    out_dir = DATASET_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.mp4"
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        for frame in frames:
            writer.write(cv2.resize(frame, size) if scale < 1.0 else frame[: size[1], : size[0]])
    finally:
        writer.release()
    return dest


def _dataset_stats() -> dict:
    return {label: len(list((DATASET_DIR / label).glob("*.mp4"))) for label in DATASET_LABELS}


@app.get("/dataset/stats")
def dataset_stats() -> dict:
    return {"counts": _dataset_stats(), "dir": str(DATASET_DIR)}


@app.post("/dataset/clip")
async def dataset_add_clip(
    file: UploadFile = File(...),
    label: str = Form(...),
    duration_ms: float = Form(0),
) -> dict:
    """Save a labelled webcam clip for fine-tuning."""
    if label not in DATASET_LABELS:
        raise HTTPException(400, f"Nhãn không hợp lệ: {label}")
    tmp = Path(tempfile.gettempdir()) / f"ds-{uuid.uuid4().hex}.webm"
    try:
        tmp.write_bytes(await file.read())
        dest = await run_in_threadpool(_save_dataset_clip, tmp, label, duration_ms)
    finally:
        tmp.unlink(missing_ok=True)
    return {"ok": True, "label": label, "name": dest.name, "counts": _dataset_stats()}


@app.delete("/dataset/clip/{label}/{name}")
def dataset_delete_clip(label: str, name: str) -> dict:
    path = DATASET_DIR / label / name
    if label not in DATASET_LABELS or path.parent != DATASET_DIR / label or path.suffix != ".mp4":
        raise HTTPException(400, "Clip không hợp lệ")
    if not path.exists():
        raise HTTPException(404, "Không tìm thấy clip")
    path.unlink()
    return {"ok": True, "counts": _dataset_stats()}


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
    mime = (upload.content_type or "").lower()
    if not (mime.startswith(VIDEO_MIME_PREFIXES) or mime in VIDEO_MIME_EXTRA):
        raise HTTPException(400, f"Kiểu MIME không phải video: {mime}")
    source = (source or "upload").strip().lower()
    if source not in {"upload", "webcam", "rtsp"}:
        source = "upload"

    tmp = Path(tempfile.gettempdir()) / f"vd-{uuid.uuid4().hex}{suffix}"
    try:
        # Stream to disk: a 20-minute file can be hundreds of MB.
        with tmp.open("wb") as dst:
            await run_in_threadpool(_copy_limited, upload.file, dst)
        await run_in_threadpool(_check_video_content, tmp)
        out = root / "reports" / "outputs" / "api" / tmp.stem
        selected_model = _webcam_model if model_kind == "webcam" else _upload_model
        selected_cfg = _webcam_cfg if model_kind == "webcam" else _upload_cfg
        selected_device = _webcam_device if model_kind == "webcam" else _upload_device
        result = await run_in_threadpool(
            partial(run_inference, selected_model, selected_cfg, selected_device, tmp, out,
                    max_duration_s=MAX_UPLOAD_SECONDS)
        )

        pred_class = result["pred_class"]
        # Confidence stored in history matches the page: share of the winning class.
        ratio = result.get("scene_violence_ratio")
        if ratio is not None:
            ratio = float(ratio)
            prob = ratio if pred_class == "violence" else 1.0 - ratio
        elif pred_class == "violence":
            prob = float(result["violence_prob_max"])
        else:
            prob = 1.0 - float(result["violence_prob_mean"])
        # The video the user gets back: every person boxed and labelled
        # Violence / Non-violence per segment. History keeps this one; the
        # plain copy is only the fallback when rendering fails.
        annotated = alerts_dir() / f"{datetime.now():%H%M%S}_{uuid.uuid4().hex[:6]}_{pred_class}_annotated.webm"
        try:
            saved = await run_in_threadpool(render_annotated_video, tmp, result, annotated)
        except Exception as exc:  # noqa: BLE001 - the label is still valid without the overlay
            print(f"annotated video failed: {exc}")
            annotated.unlink(missing_ok=True)
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
        mail_alert = {"sent": False, "skipped": True}
        segments = result.get("segments") or []
        if pred_class == "violence" or segments:
            segment_note = "; ".join(
                f"{s['start_s']:.1f}s–{s['end_s']:.1f}s" for s in segments
            )
            mail_alert = await run_in_threadpool(
                partial(send_violence_alert,
                        note=f"Video: {upload.filename}\nCác đoạn bạo lực: {segment_note or 'Theo kết quả phân loại'}",
                        peak_prob=float(result.get("violence_prob_max", result["violence_prob_mean"])),
                        history_id=hist_id, source=source)
            )
        row = get_detection(hist_id)
        return _prediction_payload(
            result,
            history_id=hist_id,
            extra={
                "video": upload.filename,
                "video_url": _rel_media(saved),
                "annotated": saved.suffix == ".webm",
                "attention_url": _rel_media(attn_path),
                "created_at": row["created_at"] if row else None,
                "mail_alert": mail_alert,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/predict/url")
async def predict_url(url: str = Form(...)) -> dict:
    from src.video_url import download_video
    _ensure_model("upload")
    temp = Path(tempfile.gettempdir()) / f"url-{uuid.uuid4().hex}.mp4"
    try:
        name, size = await run_in_threadpool(download_video, url.strip(), temp)
        # Reuse upload processing, history, annotation, and notification handling.
        if Path(name).suffix.lower() not in VIDEO_EXTS:
            name = "video.mp4"
        with temp.open("rb") as stream:
            uploaded = UploadFile(filename=name, file=stream)
            result = await predict(file=uploaded, video=None, source="upload")
        result.update({"original_name": name, "size_bytes": size})
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, "Không tải được video. Kiểm tra link và thử lại.") from exc
    finally:
        temp.unlink(missing_ok=True)


@app.post("/reports/pdf")
async def export_pdf(report: str = Form(...)):
    if len(report) > 2_000_000:
        raise HTTPException(413, "Báo cáo quá lớn")
    from src.pdf_report import make_report_pdf
    try:
        content = await run_in_threadpool(make_report_pdf, report)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=report.pdf"})
