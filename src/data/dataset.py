from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import load_frame_dir, read_video_sampled
from src.data.transforms import ClipTransform


class ViolenceClipDataset(Dataset):
    def __init__(self, csv_path: str | Path, num_frames: int, image_size: int, train: bool):
        self.df = pd.read_csv(csv_path)
        required = {"path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Split CSV missing columns: {missing}")
        self.num_frames = num_frames
        self.transform = ClipTransform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = Path(row["path"])
        if path.is_dir():
            frames = load_frame_dir(path, self.num_frames)
        else:
            frames = read_video_sampled(path, self.num_frames)
        clip = self.transform(frames)
        return {
            "frames": clip,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "path": str(path),
        }
