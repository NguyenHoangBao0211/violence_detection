"""E6: size and speed of each checkpoint on this machine.

One window = one forward pass on a (1, T, 3, 224, 224) clip, the unit the
Camera page and the upload scorer run. YOLO cropping is not included.

  python -m src.research.benchmark --checkpoint checkpoints/best2.pt checkpoints/best_diff.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.infer import load_detector
from src.utils.config import load_config, project_root


def measure(model, frames: int, size: int, runs: int, warmup: int = 2) -> dict:
    x = torch.randn(1, frames, 3, size, size)
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            model(x)
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    ms = float(np.median(times))
    return {
        "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        # FlopCounterMode counts multiply-adds as 2 FLOPs.
        "gflops_per_window": round(counter.get_total_flops() / 1e9, 1),
        "cpu_ms_per_window": round(ms, 1),
        "windows_per_second": round(1000 / ms, 2),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Do tham so, GFLOPs va do tre CPU cua checkpoint")
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None, help="so luong CPU thread (mac dinh: cua torch)")
    ap.add_argument("--out", default="reports/research/benchmark.json")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    root = project_root()
    results = {}
    for ckpt in args.checkpoint:
        try:
            model, cfg, device = load_detector(root / ckpt, load_config(root / "configs/default.yaml"))
            if device.type != "cpu":
                model = model.cpu()
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint should not stop the table
            print(f"bo qua {ckpt}: {exc}")
            continue
        model.eval()
        frames = int(cfg["data"].get("num_frames", 16))
        size = int(cfg["data"].get("image_size", 224))
        results[ckpt] = {"model_type": cfg.get("model_type", cfg.get("model", {}).get("arch", "?")),
                         "frames": frames, **measure(model, frames, size, args.runs)}
        print(ckpt, results[ckpt])

    results["_machine"] = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
