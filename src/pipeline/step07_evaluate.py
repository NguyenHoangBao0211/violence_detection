"""Bước 7 — Đánh giá trên tập test (đặc trưng đã cache)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.feature_dataset import FeatureSequenceDataset
from src.models.cnn_lstm import TemporalHead
from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.utils.config import project_root
from src.utils.metrics import compute_metrics, confusion, report as cls_report
from src.utils.viz import plot_confusion_matrix, plot_roc


def run(cfg: dict) -> dict:
    banner(7)
    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = root / cfg["data"].get("features_dir", "data/features") / "test"
    ckpt = torch.load(root / cfg["train"]["checkpoint_dir"] / "temporal_best.pt", map_location=device)
    cfg = ckpt.get("cfg", cfg)

    ds = FeatureSequenceDataset(root / cfg["data"]["splits_dir"] / "test.csv", feat_dir)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
    head = TemporalHead(input_dim=ds[0]["features"].shape[-1], cfg=cfg).to(device)
    head.load_state_dict(ckpt["temporal"])
    head.eval()

    y_true, y_pred, y_prob, ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = head(batch["features"].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(batch["label"].tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_prob.extend(prob.cpu().tolist())
            ids.extend(batch["clip_id"])

    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    y_prob_a = np.array(y_prob)
    labels = cfg["data"]["classes"]
    metrics = compute_metrics(y_true_a, y_pred_a, y_prob_a)
    cm = confusion(y_true_a, y_pred_a)
    text = cls_report(y_true_a, y_pred_a, labels)

    out = out_dir(cfg)
    plot_confusion_matrix(cm, labels, out / "step07_confusion_test.png")
    if len(np.unique(y_true_a)) == 2:
        plot_roc(y_true_a, y_prob_a, out / "step07_roc_test.png")
    pd.DataFrame({"clip_id": ids, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}).to_csv(
        out / "step07_predictions_test.csv", index=False
    )
    (out / "step07_classification_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print("  metrics:", {k: round(v, 4) for k, v in metrics.items()})
    save_json(out / "step07_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(load_cfg(args.config))


if __name__ == "__main__":
    main()
