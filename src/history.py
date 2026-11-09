"""SQLite history for violence detections."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2

from src.utils.config import project_root


def db_path() -> Path:
    path = project_root() / "data" / "history.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def alerts_dir(day: str | None = None) -> Path:
    day = day or datetime.now().strftime("%Y-%m-%d")
    path = project_root() / "storage" / "alerts" / day
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                pred_class TEXT NOT NULL,
                prob REAL NOT NULL,
                video_path TEXT,
                attention_path TEXT,
                note TEXT
            )
            """
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(detections)")}
        if "hidden" not in cols:
            conn.execute(
                "ALTER TABLE detections ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()


def save_video_copy(src: Path, pred_class: str, suffix: str) -> Path:
    dest_dir = alerts_dir()
    stamp = datetime.now().strftime("%H%M%S")
    # Lịch sử luôn dùng đuôi MP4 để trình duyệt phát ổn định.
    dest = dest_dir / f"{stamp}_{pred_class}.mp4"
    if str(src).lower().endswith(".mp4"):
        shutil.copy2(src, dest)
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video để chuyển sang MP4: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        writer = cv2.VideoWriter(
            str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            cap.release()
            writer.release()
        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"Không tạo được file MP4: {dest}")
    return dest


def add_detection(
    *,
    source: str,
    pred_class: str,
    prob: float,
    video_path: str | Path | None = None,
    attention_path: str | Path | None = None,
    note: str | None = None,
) -> int:
    init_db()
    created = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO detections
            (created_at, source, pred_class, prob, video_path, attention_path, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created,
                source,
                pred_class,
                float(prob),
                str(video_path) if video_path else None,
                str(attention_path) if attention_path else None,
                note,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_detections(limit: int = 100, *, include_hidden: bool = False) -> list[dict]:
    init_db()
    where = "" if include_hidden else "WHERE hidden = 0"
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, created_at, source, pred_class, prob, video_path, attention_path, note
            FROM detections
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_detection(det_id: int) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?",
            (det_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_detection(det_id: int) -> dict | None:
    """Hide one row from the history page. Stats keep reading it."""
    row = get_detection(det_id)
    if row is None or row.get("hidden"):
        return None
    with connect() as conn:
        conn.execute("UPDATE detections SET hidden = 1 WHERE id = ?", (det_id,))
        conn.commit()
    return row


def delete_all_detections() -> list[dict]:
    """Hide every visible history row. The database rows stay for statistics."""
    rows = list_detections(100000)
    if not rows:
        return []
    with connect() as conn:
        conn.execute("UPDATE detections SET hidden = 1 WHERE hidden = 0")
        conn.commit()
    return rows
