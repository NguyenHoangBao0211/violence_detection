"""Chạy từng bước hoặc cả 8 bước luồng xử lý."""

from __future__ import annotations

import argparse
import traceback

from src.pipeline import step01_collect
from src.pipeline import step02_preprocess
from src.pipeline import step03_clips
from src.pipeline import step04_spatial
from src.pipeline import step05_temporal
from src.pipeline import step06_train
from src.pipeline import step07_evaluate
from src.pipeline import step08_infer
from src.pipeline.common import STEPS, load_cfg, out_dir, save_json


RUNNERS = {
    1: step01_collect.run,
    2: step02_preprocess.run,
    3: step03_clips.run,
    4: step04_spatial.run,
    5: step05_temporal.run,
    6: step06_train.run,
    7: step07_evaluate.run,
    8: step08_infer.run,
}


def run_steps(cfg: dict, only: int | None = None, start: int = 1, end: int = 8) -> dict:
    steps = [only] if only else list(range(start, end + 1))
    results = {}
    for i in steps:
        try:
            results[f"step{i:02d}"] = {"title": STEPS[i], "ok": True, "result": RUNNERS[i](cfg)}
        except Exception as exc:
            results[f"step{i:02d}"] = {"title": STEPS[i], "ok": False, "error": str(exc)}
            print(f"\n  LỖI bước {i}: {exc}")
            traceback.print_exc()
            break
    save_json(
        out_dir(cfg) / "pipeline_summary.json",
        {k: {kk: vv for kk, vv in v.items() if kk != "result"} for k, v in results.items()},
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy luồng 8 bước phát hiện bạo lực")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--step", type=int, choices=list(range(1, 9)), help="Chỉ chạy 1 bước")
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--to-step", type=int, default=8)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    run_steps(cfg, only=args.step, start=args.from_step, end=args.to_step)


if __name__ == "__main__":
    main()
