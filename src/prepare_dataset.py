from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import extract_frames, list_videos
from src.utils.config import load_config, project_root


def prepare(cfg: dict, extract: bool = True) -> None:
    data_cfg = cfg["data"]
    root = project_root()
    raw_dir = root / data_cfg["raw_dir"]
    processed_dir = root / data_cfg["processed_dir"]
    splits_dir = root / data_cfg["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    classes = data_cfg["classes"]

    records = list_videos(raw_dir, classes)
    if not records:
        raise FileNotFoundError(
            f"No videos found under {raw_dir}. Put files in "
            + " and ".join(str(raw_dir / c) for c in classes)
        )

    rows = []
    for rec in records:
        dest = processed_dir / rec["class_name"] / rec["video_id"]
        if extract:
            n = extract_frames(rec["src"], dest)
            if n == 0:
                print(f"skip empty video: {rec['src']}")
                continue
            path = str(dest)
        else:
            path = rec["src"]
        rows.append(
            {
                "video_id": rec["video_id"],
                "clip_id": f"{rec['class_name']}__{rec['video_id']}",
                "path": path,
                "src": rec["src"],
                "label": rec["label"],
                "class_name": rec["class_name"],
            }
        )

    df = pd.DataFrame(rows)
    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]
    test_ratio = data_cfg["test_ratio"]
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["label"],
        random_state=cfg["seed"],
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=cfg["seed"],
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    df.to_csv(splits_dir / "all.csv", index=False)
    print(f"videos={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames and create dataset splits")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-extract", action="store_true", help="Keep original video paths")
    args = parser.parse_args()
    cfg = load_config(project_root() / args.config)
    prepare(cfg, extract=not args.no_extract)


if __name__ == "__main__":
    main()
