from __future__ import annotations

import numpy as np


def sample_indices(n_frames: int, num_frames: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("Video has no frames")
    if n_frames >= num_frames:
        return np.linspace(0, n_frames - 1, num_frames, dtype=int)
    idxs = np.arange(n_frames)
    pad = np.full(num_frames - n_frames, n_frames - 1, dtype=int)
    return np.concatenate([idxs, pad])


def sliding_windows(n_frames: int, window: int, stride: int) -> list[np.ndarray]:
    if n_frames <= window:
        return [sample_indices(n_frames, window)]
    starts = list(range(0, n_frames - window + 1, stride))
    if starts[-1] != n_frames - window:
        starts.append(n_frames - window)
    return [np.arange(s, s + window) for s in starts]
