"""Bước 6 — Huấn luyện LSTM+attention trên đặc trưng CNN đã cache."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import FeatureSequenceDataset
from src.models.cnn_lstm import CNNLSTM, TemporalHead
from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.utils.config import project_root
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed
from src.utils.viz import plot_history


def _epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train(train)
    y_true, y_pred = [], []
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            feat = batch["features"].to(device)
            labels = batch["label"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(feat)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            bs = labels.size(0)
            total += loss.item() * bs
            n += bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    metrics["loss"] = total / max(n, 1)
    return metrics


def run(cfg: dict) -> dict:
    banner(6)
    set_seed(cfg["seed"])
    root = project_root()
    feat_root = root / cfg["data"].get("features_dir", "data/features")
    splits = root / cfg["data"]["splits_dir"]
    ckpt_dir = root / cfg["train"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FeatureSequenceDataset(splits / "train.csv", feat_root / "train")
    val_ds = FeatureSequenceDataset(splits / "val.csv", feat_root / "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    sample = train_ds[0]["features"]
    head = TemporalHead(input_dim=sample.shape[-1], cfg=cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1, stale = -1.0, 0
    best_path = ckpt_dir / "best.pt"
    patience = cfg["train"].get("patience", 6)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = _epoch(head, train_loader, criterion, opt, device, True)
        va = _epoch(head, val_loader, criterion, opt, device, False)
        sch.step()
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_f1"].append(tr["f1"])
        history["val_f1"].append(va["f1"])
        print(
            f"  epoch {epoch:02d}  train_loss={tr['loss']:.4f} f1={tr['f1']:.3f}  "
            f"val_loss={va['loss']:.4f} f1={va['f1']:.3f} acc={va['accuracy']:.3f}",
            flush=True,
        )
        if va["f1"] >= best_f1:
            best_f1 = va["f1"]
            stale = 0
            torch.save({"temporal": head.state_dict(), "cfg": cfg, "val": va}, ckpt_dir / "temporal_best.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    spatial_ckpt = torch.load(ckpt_dir / "spatial.pt", map_location="cpu")
    full = CNNLSTM(cfg)
    full.spatial.load_state_dict(spatial_ckpt["spatial"])
    full.temporal.load_state_dict(torch.load(ckpt_dir / "temporal_best.pt", map_location="cpu")["temporal"])
    torch.save({"model": full.state_dict(), "cfg": cfg, "val_f1": best_f1}, best_path)

    plot_history(history, out_dir(cfg) / "step06_training_curves.png")
    (out_dir(cfg) / "step06_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    report = {
        "best_val_f1": best_f1,
        "checkpoint": str(best_path),
        "trained_on": "cached ResNet features (CNN frozen)",
        "epochs_ran": len(history["val_f1"]),
    }
    save_json(out_dir(cfg) / "step06_report.json", report)
    print(f"  best.pt  val_f1={best_f1:.4f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(load_cfg(args.config))


if __name__ == "__main__":
    main()
