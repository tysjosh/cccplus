"""Hierarchical bootstrap, Holm correction, AUROC, matched-retention comparison.

Sec. 3.6: "Validation rules are compared on the same candidates using 10,000
hierarchical bootstrap replicates over program/model stratum, source mechanism,
fact or IOI template family, and translator seed; prompts are resampled within
these clusters. Holm correction covers the seven planned contrasts."

Sec. 4.1: "Primary comparisons use false acceptance at matched positive retention
and coverage."

Matched-retention comparison is the reason every rule exposes a single normalised
error: thresholding that error at a common positive-retention level removes the
effect of where each rule's calibrated bound happens to sit, so the remaining
difference in false acceptance is attributable to the rule itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------- metrics
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with ties handled by mid-ranks."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    finite = np.isfinite(scores)
    s = np.where(finite, scores, -np.inf)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # mid-ranks for ties
    su = s[order]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def retention(errors: np.ndarray, labels: np.ndarray, threshold: float = 1.0) -> float:
    """Positive retention: fraction of positives accepted (abstention counts against)."""
    pos = labels == 1
    if not pos.any():
        return float("nan")
    return float((errors[pos] <= threshold).mean())


def false_acceptance(errors: np.ndarray, labels: np.ndarray, threshold: float = 1.0) -> float:
    neg = labels == 0
    if not neg.any():
        return float("nan")
    return float((errors[neg] <= threshold).mean())


def coverage(abstained: np.ndarray) -> float:
    """Fraction of eligible source paths with a non-null candidate."""
    a = np.asarray(abstained, dtype=bool)
    if a.size == 0:
        return float("nan")
    return float((~a).mean())


def threshold_for_retention(errors: np.ndarray, labels: np.ndarray, target: float) -> float:
    """Smallest error threshold achieving at least ``target`` positive retention."""
    pos = np.asarray(errors)[labels == 1]
    pos = pos[np.isfinite(pos)]
    if pos.size == 0 or not np.isfinite(target):
        return float("nan")
    target = float(np.clip(target, 0.0, 1.0))
    if target <= 0:
        return float(-np.inf)
    k = int(np.ceil(target * pos.size))
    k = min(max(k, 1), pos.size)
    return float(np.sort(pos)[k - 1])


def fa_at_matched_retention(
    errors: np.ndarray, labels: np.ndarray, target_retention: float
) -> float:
    thr = threshold_for_retention(errors, labels, target_retention)
    if not np.isfinite(thr):
        return float("nan")
    return false_acceptance(errors, labels, thr)


# ---------------------------------------------------------------- bootstrap
class HierarchicalBootstrap:
    """Nested cluster bootstrap: resample each level with replacement, then items."""

    def __init__(self, level_labels: Sequence[np.ndarray], seed: int = 0):
        if not level_labels:
            raise ValueError("at least one clustering level is required")
        self.n = len(level_labels[0])
        self.levels = [np.asarray(l) for l in level_labels]
        self.rng = np.random.default_rng(seed)
        self._tree = self._build(np.arange(self.n), 0)

    def _build(self, idx: np.ndarray, depth: int):
        if depth >= len(self.levels):
            return ("leaf", idx)
        labels = self.levels[depth][idx]
        groups: Dict[object, List[int]] = {}
        for i, lab in zip(idx, labels):
            groups.setdefault(lab, []).append(int(i))
        children = [self._build(np.asarray(v, dtype=int), depth + 1) for v in groups.values()]
        return ("node", children)

    def _draw(self, node) -> np.ndarray:
        kind = node[0]
        if kind == "leaf":
            idx = node[1]
            if idx.size == 0:
                return idx
            return self.rng.choice(idx, size=idx.size, replace=True)
        children = node[1]
        k = len(children)
        picks = self.rng.integers(0, k, size=k)
        return np.concatenate([self._draw(children[p]) for p in picks]) if k else np.empty(0, dtype=int)

    def indices(self) -> np.ndarray:
        return self._draw(self._tree)

    def run(
        self, statistic: Callable[[np.ndarray], np.ndarray], n_boot: int, progress: bool = False
    ) -> np.ndarray:
        base = np.atleast_1d(np.asarray(statistic(np.arange(self.n)), dtype=float))
        out = np.full((n_boot, base.size), np.nan, dtype=float)
        for b in range(n_boot):
            idx = self.indices()
            try:
                out[b] = np.atleast_1d(np.asarray(statistic(idx), dtype=float))
            except Exception:
                pass
            if progress and (b + 1) % max(1, n_boot // 10) == 0:
                print(f"      bootstrap {b+1}/{n_boot}", flush=True)
        return out


@dataclass
class Estimate:
    """Point estimate with a bootstrap percentile interval and a bootstrap p-value."""

    value: float
    lo: float = float("nan")
    hi: float = float("nan")
    p: float = float("nan")
    p_adjusted: float = float("nan")
    n_boot: int = 0

    def fmt(self, digits: int = 3) -> str:
        if not np.isfinite(self.value):
            return "n/a"
        if np.isfinite(self.lo) and np.isfinite(self.hi):
            return f"{self.value:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"
        return f"{self.value:.{digits}f}"

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def summarise(samples: np.ndarray, point: float, ci: float = 0.95, two_sided_zero: bool = False) -> Estimate:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return Estimate(point)
    alpha = (1.0 - ci) / 2.0
    lo, hi = float(np.quantile(s, alpha)), float(np.quantile(s, 1.0 - alpha))
    p = float("nan")
    if two_sided_zero:
        frac_le = float((s <= 0).mean())
        frac_ge = float((s >= 0).mean())
        p = float(min(1.0, 2.0 * min(frac_le, frac_ge)))
    return Estimate(point, lo, hi, p, float("nan"), int(s.size))


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjustment; NaNs pass through."""
    p = np.asarray([np.nan if v is None else v for v in p_values], dtype=float)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    out = np.full(p.shape, np.nan)
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = idx.size
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out.tolist()
