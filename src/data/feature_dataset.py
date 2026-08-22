from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def clip_id_of(row) -> str:
    if "clip_id" in row.index and pd.notna(row.get("clip_id")):
        return str(row["clip_id"])
    return f"{row['class_name']}__{row['video_id']}"


class FeatureSequenceDataset(Dataset):
    def __init__(self, csv_path: str | Path, feature_dir: str | Path):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = Path(feature_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        cid = clip_id_of(row)
        path = self.feature_dir / f"{cid}.npy"
        feat = np.load(path)
        return {
            "features": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "clip_id": cid,
        }
