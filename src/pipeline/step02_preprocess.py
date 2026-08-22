"""Bước 2 — Trích frame và chia train/val/test."""

from __future__ import annotations

import argparse

import pandas as pd

from src.pipeline.common import banner, load_cfg, out_dir, save_json
from src.prepare_dataset import prepare
from src.utils.config import project_root


def run(cfg: dict) -> dict:
    banner(2)
    prepare(cfg, extract=True)
    splits = project_root() / cfg["data"]["splits_dir"]
    summary = {}
    for name in ("train", "val", "test", "all"):
        df = pd.read_csv(splits / f"{name}.csv")
        summary[name] = {
            "n": int(len(df)),
            "per_class": df.groupby("class_name").size().astype(int).to_dict(),
        }
        print(f"  {name}: {summary[name]['n']}  {summary[name]['per_class']}")
    save_json(out_dir(cfg) / "step02_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(load_cfg(args.config))


if __name__ == "__main__":
    main()
