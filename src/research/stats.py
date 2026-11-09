"""Statistics for comparing violence classifiers on one fixed test set.

Bootstrap resamples whole groups (`group_id`: clips cut from one source video,
or both AIRTLab cameras of one scene), because clips in a group are not
independent; resampling single clips would make the intervals too narrow.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clip_metrics(y, p, thr: float) -> dict:
    """Clip-level metrics; positive class = violence."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    neg = y == 0
    both = len(np.unique(y)) == 2
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "false_alarm_rate": float(pred[neg].mean()) if neg.any() else float("nan"),
        "roc_auc": roc_auc_score(y, p) if both else float("nan"),
        "pr_auc": average_precision_score(y, p) if both else float("nan"),
    }


def _group_index(groups: Sequence) -> list[np.ndarray]:
    _, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, bounds)


def bootstrap(
    stat: Callable[[np.ndarray], float],
    groups: Sequence,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """`stat(idx)` on `n_boot` group-level resamples of the test set.

    `stat` takes an index array into the test set, so one resample can score
    several models (paired comparison) on exactly the same clips.
    """
    members = _group_index(groups)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        out[b] = stat(np.concatenate([members[g] for g in pick]))
    return out


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if not len(samples):
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def paired_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0."""
    diffs = diffs[~np.isnan(diffs)]
    if not len(diffs):
        return float("nan")
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(1.0, 2 * tail))


def mcnemar(y, pred_a, pred_b) -> dict:
    """McNemar test on the clips where exactly one model is right.

    Exact binomial for fewer than 25 discordant clips, else chi-square with
    continuity correction.
    """
    y = np.asarray(y)
    ok_a, ok_b = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b = int((ok_a & ~ok_b).sum())  # only A right
    c = int((~ok_a & ok_b).sum())  # only B right
    if b + c == 0:
        p = 1.0
    elif b + c < 25:
        p = binomtest(b, b + c, 0.5).pvalue
    else:
        p = float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), df=1))
    return {"only_a_right": b, "only_b_right": c, "p_value": float(p)}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    adjusted = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(valid) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
