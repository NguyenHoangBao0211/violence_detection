from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"]
        logits, _ = model(frames)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        y_true.append(labels.numpy())
        y_pred.append(pred.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )
