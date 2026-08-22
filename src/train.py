from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ViolenceClipDataset
from src.models.cnn_lstm import CNNLSTM
from src.utils.config import load_config, project_root
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed
from src.utils.viz import plot_history


def make_loader(csv_path: Path, cfg: dict, train: bool) -> DataLoader:
    ds = ViolenceClipDataset(
        csv_path=csv_path,
        num_frames=cfg["data"]["num_frames"],
        image_size=cfg["data"]["image_size"],
        train=train,
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def param_groups(model: CNNLSTM, cfg: dict) -> list[dict]:
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("spatial."):
            backbone.append(p)
        else:
            rest.append(p)
    lr = cfg["train"]["lr"]
    backbone_lr = cfg["train"].get("backbone_lr", lr * 0.1)
    groups = [{"params": rest, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool) -> dict:
    model.train(train)
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None and device.type == "cuda"):
                logits, _ = model(frames)
                loss = criterion(logits, labels)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train(cfg: dict) -> Path:
    set_seed(cfg["seed"])
    root = project_root()
    splits = root / cfg["data"]["splits_dir"]
    ckpt_dir = root / cfg["train"]["checkpoint_dir"]
    log_dir = root / cfg["train"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(splits / "train.csv", cfg, train=True)
    val_loader = make_loader(splits / "val.csv", cfg, train=False)

    model = CNNLSTM(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("use_amp", True) and device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = -1.0
    best_path = ckpt_dir / "best.pt"
    patience = cfg["train"].get("patience", 8)
    stale = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_m = run_epoch(model, val_loader, criterion, optimizer, device, None, False)
        scheduler.step()
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])
        print(
            f"epoch {epoch:03d}  train_loss={train_m['loss']:.4f} f1={train_m['f1']:.4f}  "
            f"val_loss={val_m['loss']:.4f} f1={val_m['f1']:.4f} acc={val_m['accuracy']:.4f}"
        )
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            stale = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "val": val_m}, best_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}")
                break

    plot_history(history, log_dir / "training_curves.png")
    (log_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best checkpoint: {best_path}  val_f1={best_f1:.4f}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM violence detector")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(project_root() / args.config)
    train(cfg)


if __name__ == "__main__":
    main()
