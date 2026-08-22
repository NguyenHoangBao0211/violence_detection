from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: str | Path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_attention(frames_rgb: list[np.ndarray], weights: np.ndarray, out_path: str | Path) -> None:
    t = len(frames_rgb)
    fig, axes = plt.subplots(1, t, figsize=(2.2 * t, 3.2))
    if t == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(frames_rgb[i])
        ax.set_title(f"{weights[i]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Temporal attention over sampled frames", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_history(history: dict, out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_f1"], label="train F1")
    plt.plot(history["val_f1"], label="val F1")
    plt.title("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
