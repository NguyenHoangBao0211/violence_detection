"""Labelled webcam clips recorded from the Camera page (data/webcam_dataset).

Layout: <root>/violence/*.mp4 and <root>/non_violence/*.mp4.
The split is a stable hash of the file name, so a clip keeps its split when
more clips are added. notebooks/finetune_webcam.py has a copy of split_of();
keep the two in sync.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

LABELS = {"non_violence": 0, "violence": 1}


def split_of(path: str | Path) -> str:
    bucket = int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 10
    return "test" if bucket < 2 else "val" if bucket < 3 else "train"


def list_clips(root: str | Path, split: str | None = None) -> list[tuple[Path, int]]:
    clips = []
    for name, label in LABELS.items():
        for p in sorted((Path(root) / name).glob("*.mp4")):
            if split in (None, "all") or split_of(p) == split:
                clips.append((p, label))
    return clips
