"""Aggregate training runs into the thesis tables.

Each run is a folder written by the training notebook (one EXPERIMENT and one
TRAIN_SEED): metrics.json + test_predictions.csv (+ robustness.json). Copy the
run folders from Kaggle into one directory, then:

  python -m src.research.aggregate runs/ --reference rgb \
      --manifest C:/Users/ASUS/Downloads/results/resnet50_bilstm_attention/manifest_resolved.csv

Writes to reports/research/: runs.csv, summary.csv, comparisons.csv,
per_source.csv, robustness.csv and report.md.
  - summary: mean +- std over seeds, and a 95% bootstrap CI of the seed-mean
  - comparisons: each configuration minus the reference, paired on the same
    clips and seeds; bootstrap p-value with Holm correction, McNemar per seed
Runs are grouped by (experiment, init_from, train_sources): rows that differ
in initialisation or training sources are different configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.research.stats import bootstrap, ci, clip_metrics, holm, mcnemar, paired_p
from src.utils.config import project_root

HEADLINE = ["accuracy", "f1_macro", "roc_auc"]


# Fast versions for the bootstrap loop (sklearn is ~100x slower per call).
def _acc(y, pred):
    return float((y == pred).mean())


def _f1_macro(y, pred):
    out = []
    for cls in (0, 1):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        out.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(out))


def _auc(y, p):
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


FAST = {
    "accuracy": lambda y, p, pred: _acc(y, pred),
    "f1_macro": lambda y, p, pred: _f1_macro(y, pred),
    "roc_auc": lambda y, p, pred: _auc(y, p),
}


def load_runs(root: Path, manifest: Path | None) -> list[dict]:
    meta = None
    if manifest:
        m = pd.read_csv(manifest)
        meta = m.drop_duplicates("resolved_path").set_index("resolved_path")[["group_id", "source"]]
    runs = []
    for mpath in sorted(root.rglob("metrics.json")):
        preds = mpath.parent / "test_predictions.csv"
        if not preds.is_file():
            print(f"bo qua {mpath.parent}: thieu test_predictions.csv")
            continue
        r = json.loads(mpath.read_text(encoding="utf-8"))
        df = pd.read_csv(preds)
        # Old runs saved only path/label/prob/pred: take group and source from the manifest.
        if meta is not None:
            for col in ("group_id", "source"):
                if col not in df.columns:
                    df[col] = df.path.map(meta[col])
        if "group_id" not in df.columns:
            df["group_id"] = df.path
        df["group_id"] = df.group_id.fillna(df.path)
        if "source" not in df.columns:
            df["source"] = ""
        sources = r.get("train_sources", "all")
        runs.append({
            "dir": mpath.parent,
            "experiment": r["experiment"],
            "seed": int(r.get("seed", 42)),
            "init": r.get("init_from", "?"),
            "train_sources": ",".join(sources) if isinstance(sources, list) else str(sources),
            "threshold": float(r["test"]["threshold"]),
            "latency": r.get("latency_ms", {}),
            "params_million": r.get("params_million"),
            "pred": df.set_index("path"),
            "robustness": json.loads((mpath.parent / "robustness.json").read_text(encoding="utf-8"))
            if (mpath.parent / "robustness.json").is_file() else None,
        })
    return runs


def config_names(runs: list[dict]) -> None:
    """Label each run's configuration; add init/sources only when they vary."""
    vary_init = len({r["init"] for r in runs}) > 1
    vary_src = len({r["train_sources"] for r in runs}) > 1
    for r in runs:
        name = r["experiment"]
        if vary_init:
            name += f"[{r['init']}]"
        if vary_src and r["train_sources"] != "all":
            name += f"[train={r['train_sources']}]"
        r["config"] = name


def md_table(df: pd.DataFrame) -> str:
    """Pivot table as markdown, without pandas' optional tabulate dependency."""
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for idx, vals in df.iterrows():
        rows.append("| " + " | ".join([str(idx)] + ["" if pd.isna(v) else f"{v:.3f}" for v in vals]) + " |")
    return "\n".join(rows)


def fmt(mean, std, lo=None, hi=None) -> str:
    s = f"{mean:.3f}" if np.isnan(std) else f"{mean:.3f} ± {std:.3f}"
    return s if lo is None else f"{s} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Gop ket qua cac lan train thanh bang nghien cuu")
    ap.add_argument("runs_dir", help="thu muc chua cac <experiment>_s<seed>/")
    ap.add_argument("--reference", default="rgb", help="cau hinh moc de so sanh (mac dinh: rgb)")
    ap.add_argument("--manifest", default=None, help="manifest_resolved.csv: group_id/source cho run cu")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/research")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir), Path(args.manifest) if args.manifest else None)
    if not runs:
        raise SystemExit(f"Khong tim thay metrics.json + test_predictions.csv trong {args.runs_dir}")
    config_names(runs)

    # Every comparison must use the same test clips.
    common = sorted(set.intersection(*(set(r["pred"].index) for r in runs)))
    sizes = {len(r["pred"]) for r in runs}
    if len(sizes) > 1 or len(common) != sizes.pop():
        print(f"CANH BAO: cac run co tap test khac nhau; chi dung {len(common)} clip chung.")
    first = runs[0]["pred"].loc[common]
    y = first.label.to_numpy().astype(int)
    groups = first.group_id.to_numpy()
    source = first.source.astype(str).to_numpy()
    for r in runs:
        d = r["pred"].loc[common]
        if not np.array_equal(d.label.to_numpy().astype(int), y):
            raise SystemExit(f"Nhan test khac nhau o {r['dir']}: cac run khong cung manifest.")
        r["p"], r["hard"] = d.prob.to_numpy(), d.pred.to_numpy().astype(int)

    by_config: dict[str, list[dict]] = {}
    for r in runs:
        by_config.setdefault(r["config"], []).append(r)
    for c in by_config:
        by_config[c].sort(key=lambda r: r["seed"])
        seeds = [r["seed"] for r in by_config[c]]
        if len(seeds) != len(set(seeds)):
            print(f"CANH BAO: {c} co seed trung lap {seeds}; kiem tra thu muc run.")

    out = project_root() / args.out
    out.mkdir(parents=True, exist_ok=True)

    # Per run.
    run_rows = []
    for r in runs:
        m = clip_metrics(y, r["p"], r["threshold"])
        run_rows.append({"config": r["config"], "seed": r["seed"], "threshold": r["threshold"],
                         **{k: round(v, 4) for k, v in m.items()},
                         "params_M": r["params_million"], "cpu_ms": r["latency"].get("cpu"),
                         "gpu_ms": r["latency"].get("device"), "dir": str(r["dir"])})
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(out / "runs.csv", index=False)

    # Summary: mean +- std over seeds, CI of the seed-mean.
    summary = []
    for c, rs in by_config.items():
        row = {"config": c, "n_seeds": len(rs)}
        sub = run_df[run_df.config == c]
        for k in ["accuracy", "precision", "recall", "f1_macro", "false_alarm_rate", "roc_auc", "pr_auc"]:
            row[f"{k}_mean"], row[f"{k}_std"] = sub[k].mean(), sub[k].std(ddof=1) if len(sub) > 1 else np.nan
        for k in HEADLINE:
            f = FAST[k]
            samples = bootstrap(lambda idx: np.mean([f(y[idx], r["p"][idx], r["hard"][idx]) for r in rs]),
                                groups, args.n_boot)
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = ci(samples)
        row["params_M"] = sub.params_M.iloc[0]
        row["cpu_ms"] = sub.cpu_ms.mean()
        summary.append(row)
    summary_df = pd.DataFrame(summary).sort_values("f1_macro_mean", ascending=False)
    summary_df.round(4).to_csv(out / "summary.csv", index=False)

    # Comparisons with the reference, paired by seed and clip.
    comparisons = []
    if args.reference not in by_config:
        print(f"CANH BAO: khong co cau hinh moc '{args.reference}'; bo qua so sanh. Co: {sorted(by_config)}")
    else:
        ref = {r["seed"]: r for r in by_config[args.reference]}
        for c, rs in by_config.items():
            if c == args.reference:
                continue
            pairs = [(r, ref[r["seed"]]) for r in rs if r["seed"] in ref]
            if not pairs:
                print(f"CANH BAO: {c} khong co seed chung voi {args.reference}; bo qua.")
                continue
            row = {"config": c, "reference": args.reference, "paired_seeds": len(pairs)}
            for k in ("accuracy", "roc_auc"):
                f = FAST[k]
                diff = lambda idx: np.mean([f(y[idx], a["p"][idx], a["hard"][idx]) - f(y[idx], b["p"][idx], b["hard"][idx])
                                            for a, b in pairs])
                samples = bootstrap(diff, groups, args.n_boot)
                row[f"delta_{k}"] = diff(np.arange(len(y)))
                row[f"delta_{k}_ci_low"], row[f"delta_{k}_ci_high"] = ci(samples)
                row[f"p_{k}"] = paired_p(samples)
            tests = [mcnemar(y, a["hard"], b["hard"]) for a, b in pairs]
            row["mcnemar_p_per_seed"] = ";".join(f"{t['p_value']:.4f}" for t in tests)
            row["mcnemar_significant_seeds"] = f"{sum(t['p_value'] < 0.05 for t in tests)}/{len(tests)}"
            comparisons.append(row)
        if comparisons:
            for k in ("accuracy", "roc_auc"):
                for row, adj in zip(comparisons, holm([row[f"p_{k}"] for row in comparisons])):
                    row[f"p_{k}_holm"] = adj
    comp_df = pd.DataFrame(comparisons)
    if len(comp_df):
        comp_df.round(4).to_csv(out / "comparisons.csv", index=False)

    # Per source (in-domain vs cross-dataset rows).
    per_source = []
    for c, rs in by_config.items():
        for s in sorted(set(source)):
            idx = np.flatnonzero(source == s)
            ms = [clip_metrics(y[idx], r["p"][idx], r["threshold"]) for r in rs]
            row = {"config": c, "source": s or "?", "n": len(idx)}
            for k in ["accuracy", "f1_macro", "roc_auc", "false_alarm_rate"]:
                row[k] = np.nanmean([m[k] for m in ms])
            row["train_sources"] = rs[0]["train_sources"]
            per_source.append(row)
    src_df = pd.DataFrame(per_source)
    src_df.round(4).to_csv(out / "per_source.csv", index=False)

    # Robustness (E5), when the notebook ran it.
    rob = []
    for c, rs in by_config.items():
        conds = {}
        for r in rs:
            for name, m in (r["robustness"] or {}).items():
                conds.setdefault(name, []).append(m)
        for name, ms in conds.items():
            rob.append({"config": c, "condition": name, "n_seeds": len(ms),
                        **{k: np.mean([m[k] for m in ms]) for k in ("accuracy", "f1_macro", "roc_auc")}})
    rob_df = pd.DataFrame(rob)
    if len(rob_df):
        rob_df.round(4).to_csv(out / "robustness.csv", index=False)

    # Markdown report.
    lines = [f"# Ket qua nghien cuu", "",
             f"{len(runs)} run, {len(by_config)} cau hinh, {len(y)} clip test "
             f"({int(y.sum())} bao luc), bootstrap {args.n_boot} lan theo group_id.", "",
             "## Tong hop (mean ± std qua seed [CI 95%])", "",
             "| Cau hinh | Seed | Accuracy | F1-macro | ROC-AUC | Bao nham | Tham so (M) | CPU ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in summary_df.iterrows():
        cells = [fmt(r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_ci_low"], r[f"{k}_ci_high"]) for k in HEADLINE]
        lines.append(f"| {r.config} | {r.n_seeds} | " + " | ".join(cells)
                     + f" | {fmt(r.false_alarm_rate_mean, r.false_alarm_rate_std)} | {r.params_M} | {r.cpu_ms:.0f} |")
    if len(comp_df):
        lines += ["", f"## So voi {args.reference} (hieu = cau hinh − moc)", "",
                  "| Cau hinh | Seed ghep cap | ΔAccuracy [CI 95%] | p (Holm) | ΔROC-AUC [CI 95%] | p (Holm) | McNemar p<0.05 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        # A bootstrap p of 0 only means "below 1 / n_boot".
        pfmt = lambda p: f"< {1 / args.n_boot:g}" if p == 0 else f"{p:.4f}"
        for _, r in comp_df.iterrows():
            lines.append(
                f"| {r.config} | {r.paired_seeds} "
                f"| {r.delta_accuracy:+.3f} [{r.delta_accuracy_ci_low:+.3f}, {r.delta_accuracy_ci_high:+.3f}] | {pfmt(r.p_accuracy_holm)} "
                f"| {r.delta_roc_auc:+.3f} [{r.delta_roc_auc_ci_low:+.3f}, {r.delta_roc_auc_ci_high:+.3f}] | {pfmt(r.p_roc_auc_holm)} "
                f"| {r.mcnemar_significant_seeds} |")
    lines += ["", "## Theo nguon du lieu (F1-macro)", "",
              md_table(src_df.pivot_table(index="config", columns="source", values="f1_macro"))]
    if len(rob_df):
        lines += ["", "## Do ben (F1-macro)", "",
                  md_table(rob_df.pivot_table(index="config", columns="condition", values="f1_macro"))]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
