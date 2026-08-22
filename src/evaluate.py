from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ViolenceClipDataset
from src.models.cnn_lstm import CNNLSTM
from src.utils.config import load_config, project_root
from src.utils.metrics import collect_predictions, compute_metrics, confusion, report
from src.utils.viz import plot_confusion_matrix, plot_roc


def evaluate(cfg: dict, checkpoint: str | Path, split: str = "test") -> dict:
    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("cfg", cfg)

    ds = ViolenceClipDataset(
        csv_path=root / cfg["data"]["splits_dir"] / f"{split}.csv",
        num_frames=cfg["data"]["num_frames"],
        image_size=cfg["data"]["image_size"],
        train=False,
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    model = CNNLSTM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    labels = cfg["data"]["classes"]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = confusion(y_true, y_pred)
    text = report(y_true, y_pred, labels)

    out_dir = root / cfg["train"]["log_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, labels, out_dir / f"confusion_{split}.png")
    plot_roc(y_true, y_prob, out_dir / f"roc_{split}.png")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "path": ds.df["path"]}).to_csv(
        out_dir / f"predictions_{split}.csv", index=False
    )
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / f"classification_report_{split}.txt").write_text(text, encoding="utf-8")
    print(text)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN-LSTM on a split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    root = project_root()
    cfg = load_config(root / args.config)
    evaluate(cfg, root / args.checkpoint, args.split)


if __name__ == "__main__":
    main()
