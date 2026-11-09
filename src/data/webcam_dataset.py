"""Labelled webcam clips recorded from the Camera page (data/webcam_dataset).

Layout: <root>/violence/*.mp4 and <root>/non_violence/*.mp4.

Clips cut one per second from a long recording are named <source>_<NNN>.mp4
(NNN = start second). Neighbouring seconds look almost identical, so they are
split by time block inside each source: the first 70% of seconds -> train,
next 15% -> val, last 15% -> test, dropping GAP seconds at each boundary
("gap", used by no split). Other clips (Camera page recordings) keep the
stable file-name hash. notebooks/finetune_webcam.py has a copy of split_of();
keep the two in sync.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

LABELS = {"non_violence": 0, "violence": 1}
SECOND_CLIP = re.compile(r"^(.*)_(\d{3})$")
TRAIN_FRAC, VAL_FRAC, GAP = 0.70, 0.15, 2


@lru_cache(maxsize=None)
def _source_length(folder: Path, source: str) -> int:
    return sum(1 for p in folder.glob(f"{source}_*.mp4") if SECOND_CLIP.match(p.stem))


def split_of(path: str | Path) -> str:
    path = Path(path)
    m = SECOND_CLIP.match(path.stem)
    if m is None:
        bucket = int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 10
        return "test" if bucket < 2 else "val" if bucket < 3 else "train"
    idx, n = int(m.group(2)), _source_length(path.parent, m.group(1))
    val_start = round(n * TRAIN_FRAC)
    test_start = round(n * (TRAIN_FRAC + VAL_FRAC))
    if idx < val_start:
        return "train"
    if val_start + GAP <= idx < test_start:
        return "val"
    if idx >= test_start + GAP:
        return "test"
    return "gap"


def list_clips(root: str | Path, split: str | None = None) -> list[tuple[Path, int]]:
    clips = []
    for name, label in LABELS.items():
        for p in sorted((Path(root) / name).glob("*.mp4")):
            if split in (None, "all") or split_of(p) == split:
                clips.append((p, label))
    return clips
