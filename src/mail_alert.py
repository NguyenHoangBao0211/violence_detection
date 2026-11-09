"""Email alerts for uploaded videos and saved webcam violence events.

SMTP settings are read from the project .env; see .env.example.
"""
from __future__ import annotations

import os
import ssl
import smtplib
from email.message import EmailMessage

from src.utils.config import project_root


def load_env() -> None:
    path = project_root() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _cfg() -> dict:
    load_env()
    return {
        "enabled": os.environ.get("EMAIL_ALERTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "to": os.environ.get("ALERT_TO", "").strip(),
        "host": os.environ.get("SMTP_HOST", "smtp.gmail.com").strip(),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
    }


def send_violence_alert(*, note: str, peak_prob: float, history_id: int,
                        source: str = "webcam") -> dict:
    """Send once per saved event/upload, returning failure without raising."""
    try:
        cfg = _cfg()
        if not cfg.get("enabled", True):
            return {"sent": False, "skipped": True}
        if not cfg["to"] or not cfg["user"] or not cfg["password"]:
            return {"sent": False, "error": "Chưa cấu hình mail trong file .env"}
        source_name = "video tải lên" if source == "upload" else "webcam" if source == "webcam" else "video camera"
        msg = EmailMessage()
        msg["Subject"] = f"Cảnh báo bạo lực từ {source_name}"
        msg["From"] = cfg["user"]
        msg["To"] = cfg["to"]
        msg.set_content(
            f"Hệ thống vừa phát hiện bạo lực từ {source_name}.\n\n"
            f"{note}\n"
            f"Điểm bạo lực cao nhất: {peak_prob * 100:.1f}%\n"
            f"Mã lịch sử: #{history_id}\n"
            "Mở trang Lịch sử của hệ thống để xem video đã lưu.\n"
        )
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(cfg["user"], cfg["password"])
            smtp.send_message(msg)
    except Exception:
        return {"sent": False, "error": "Không gửi được email; kiểm tra cấu hình SMTP và kết nối mạng"}
    return {"sent": True}
