from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
