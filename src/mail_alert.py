"""Send one email when a webcam violence event is saved.

Configure in `.env` at the project root (file is gitignored):

    ALERT_TO=you@gmail.com
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=app-password
"""

from __future__ import annotations

import os
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
        "to": os.environ.get("ALERT_TO", "").strip(),
        "host": os.environ.get("SMTP_HOST", "smtp.gmail.com").strip(),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
    }


def send_violence_alert(*, note: str, peak_prob: float, history_id: int) -> dict:
    cfg = _cfg()
    if not cfg["to"] or not cfg["user"] or not cfg["password"]:
        return {"sent": False, "error": "Chưa cấu hình mail trong file .env"}
    msg = EmailMessage()
    msg["Subject"] = "Cảnh báo bạo lực trên webcam"
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(
        "Webcam vừa phát hiện bạo lực.\n\n"
        f"{note}\n"
        f"Xác suất cao nhất: {peak_prob * 100:.1f}%\n"
        f"Mã lịch sử: #{history_id}\n"
    )
    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as smtp:
            smtp.starttls()
            smtp.login(cfg["user"], cfg["password"])
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001
        return {"sent": False, "error": str(exc), "to": cfg["to"]}
    return {"sent": True, "to": cfg["to"]}
