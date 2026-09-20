"""Turning path measurements into the paper's numbers.

Computes, for every validation rule: positive retention, candidate coverage, false
acceptance, AUROC, and false acceptance at matched positive retention; then the
seven planned paired contrasts with hierarchical-bootstrap confidence intervals
and Holm-adjusted p-values.

The unit of analysis differs for the last rung of the ladder. Pairwise rules are
evaluated on single paths; the multi-model conjunction is evaluated on *pairs* of
paths through two different destinations. To keep the "Multi - pairwise" contrast
meaningful, both are additionally evaluated on the identical pair-item set, where
pairwise means "decide on the first destination alone" and multi-model means
"require both".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stats import (
    Estimate,
    HierarchicalBootstrap,
    auroc,
    coverage,
    fa_at_matched_retention,
    false_acceptance,
    holm,
    retention,
    summarise,
    threshold_for_retention,
)
from .validation import (
    CORRECT_CANDIDATE_SIMILARITY,
    LADDER,
    RULES,
    MultiPath,
    PathMeasurement,
    Rule,
)

NEGATIVE_CASES = ("wrong_variable", "effect_matched_random", "absent", "split_half",
                  "shuffled_pair_control")


# ------------------------------------------------------------------ item sets
@dataclass
class ItemSet:
    """Arrays a rule is scored on, plus clustering for the bootstrap."""

    errors: Dict[str, np.ndarray]
    labels: np.ndarray
    abstained: np.ndarray
    case: np.ndarray
    clusters: List[np.ndarray]
    translator: np.ndarray
    n: int

    def subset(self, mask: np.ndarray) -> "ItemSet":
        return ItemSet(
            {k: v[mask] for k, v in self.errors.items()},
            self.labels[mask], self.abstained[mask], self.case[mask],
            [c[mask] for c in self.clusters], self.translator[mask], int(mask.sum()),
        )


# Cluster levels the path-level bootstrap can actually form. A path's signature is
# measured over every prompt in its split, so a path is not nested inside one template
# family: the paper's `family` level and `resample_within_cluster: prompts` require
# prompt-level resampling, which this implementation does not do (README deviation 6).
# Naming the supported levels here, and rejecting the rest, keeps the manifest from
# advertising a clustering the analysis never performs.
CLUSTER_EXTRACTORS: Dict[str, Callable[[PathMeasurement], object]] = {
    "stratum": lambda m: f"{m.program}|{m.dest_model}",
    "program": lambda m: m.program,
    "dest_model": lambda m: m.dest_model,
    "source_mechanism": lambda m: m.source_mechanism,
    "translator_seed": lambda m: f"{m.translator}|{m.seed}",
    "translator": lambda m: m.translator,
}

DEFAULT_CLUSTER_LEVELS = ("stratum", "source_mechanism", "translator_seed")

UNSUPPORTED_CLUSTER_LEVELS = {
    "family": "a path aggregates every prompt in its split, so it is not nested in one "
              "template family; family clustering needs prompt-level resampling",
    "prompts": "prompt-level resampling is not implemented (README deviation 6)",
}


def resolve_cluster_levels(manifest=None) -> List[str]:
    """Cluster levels for the bootstrap, taken from the manifest when it lists them."""
    if manifest is None:
        return list(DEFAULT_CLUSTER_LEVELS)
    requested = list(manifest.path_of("statistics.cluster_levels", list(DEFAULT_CLUSTER_LEVELS)))
    levels = [lv for lv in requested if lv in CLUSTER_EXTRACTORS]
    bad = [lv for lv in requested if lv not in CLUSTER_EXTRACTORS]
    for lv in bad:
        reason = UNSUPPORTED_CLUSTER_LEVELS.get(lv, "unknown cluster level")
        raise ValueError(
            f"manifest statistics.cluster_levels requests '{lv}', which the path-level "
            f"bootstrap cannot form: {reason}. Remove it from the manifest or implement "
            f"the resampling it requires -- silently dropping it would overstate the "
            f"clustering actually performed."
        )
    if not levels:
        raise ValueError("statistics.cluster_levels resolved to no usable level")
    return levels


def build_item_set(
    records: Sequence[PathMeasurement],
    rules: Sequence[str],
    cluster_levels: Optional[Sequence[str]] = None,
) -> ItemSet:
    errors = {r: np.array([RULES[r].error(m) for m in records], dtype=float) for r in rules}
    levels = list(cluster_levels or DEFAULT_CLUSTER_LEVELS)
    return ItemSet(
        errors=errors,
        labels=np.array([m.label for m in records], dtype=int),
        abstained=np.array([m.abstained for m in records], dtype=bool),
        case=np.array([m.case for m in records], dtype=object),
        clusters=[
            np.array([CLUSTER_EXTRACTORS[lv](m) for m in records], dtype=object)
            for lv in levels
        ],
        translator=np.array([m.translator for m in records], dtype=object),
        n=len(records),
    )


def build_multi_items(
    records: Sequence[PathMeasurement],
    rules: Sequence[str],
    cluster_levels: Optional[Sequence[str]] = None,
) -> Tuple[ItemSet, List[MultiPath]]:
    """Pair the candidates proposed for one source mechanism in two destinations.

    A conjunction item represents a single coherent decision: for a given source
    mechanism, translator and seed, the candidate offered in destination A together
    with the candidate offered in destination B. Both legs must therefore describe
    the *same kind* of candidate, or the item's label describes neither leg:

    * both legs are the planted counterpart -> positive. The planted *case* differs
      between destinations by construction (the same variable is ``permuted`` in one
      model and ``merged`` in another), so the case is deliberately **not** part of
      the key; the label is.
    * both legs are the same species of decoy -> negative.

    Mixed positive/negative pairs are excluded. They conflate two different
    manipulations, so neither reading of the label is well defined, and including
    them lets one positive leg reappear in thousands of negative items -- which
    inflates the item count and breaks the independence the bootstrap assumes.
    """
    by_key: Dict[Tuple, List[PathMeasurement]] = {}
    for m in records:
        by_key.setdefault((m.program, m.source_mechanism, m.translator, m.seed), []).append(m)
    pairs: List[MultiPath] = []
    for _key, ms in by_key.items():
        for a, b in combinations(ms, 2):
            if a.dest_model == b.dest_model:
                continue
            if a.label == 1 and b.label == 1:
                pairs.append(MultiPath(a, b))
            elif a.label == 0 and b.label == 0 and a.case == b.case:
                pairs.append(MultiPath(a, b))
    # A conjunction item spans two destinations, so any destination-specific level
    # collapses to the program it belongs to.
    levels = [lv if lv not in ("stratum", "dest_model") else "program"
              for lv in (cluster_levels or DEFAULT_CLUSTER_LEVELS)]
    if not pairs:
        empty = np.zeros(0)
        return ItemSet({r: empty for r in rules}, empty.astype(int), empty.astype(bool),
                       np.array([], dtype=object),
                       [np.array([], dtype=object) for _ in levels],
                       np.array([], dtype=object), 0), pairs
    errors: Dict[str, np.ndarray] = {}
    for r in rules:
        rule = RULES[r]
        if rule.needs_two_destinations:
            errors[r] = np.array([p.error(rule) for p in pairs], dtype=float)
        else:
            errors[r] = np.array([rule.error(p.left) for p in pairs], dtype=float)
    return (
        ItemSet(
            errors=errors,
            labels=np.array([p.label for p in pairs], dtype=int),
            abstained=np.array([p.abstained for p in pairs], dtype=bool),
            # Negatives share a case, so the shared name is used and stays comparable
            # with NEGATIVE_CASES; positives conjoin two different planted structures.
            case=np.array([p.left.case if p.left.case == p.right.case
                           else f"{p.left.case}+{p.right.case}" for p in pairs], dtype=object),
            clusters=[
                np.array([CLUSTER_EXTRACTORS[lv](p.left) for p in pairs], dtype=object)
                for lv in levels
            ],
            translator=np.array([p.left.translator for p in pairs], dtype=object),
            n=len(pairs),
        ),
        pairs,
    )


# -------------------------------------------------------------------- metrics
def rule_metrics(items: ItemSet, rule: str, target_retention: Optional[float]) -> Dict[str, float]:
    """Metrics at the rule's calibrated bound, plus FA at a preregistered retention.

    ``target_retention`` is a fixed level (manifest: 0.90), not the reference rule's
    own realised retention: matching to a rule whose calibrated retention is low
    collapses every contrast to zero and measures nothing.
    """
    e = items.errors[rule]
    out = {
        "retention": retention(e, items.labels),
        "coverage": coverage(items.abstained),
        "false_acceptance": false_acceptance(e, items.labels),
        "auroc": auroc(-e, items.labels),
    }
    out["fa_at_matched_retention"] = (
        fa_at_matched_retention(e, items.labels, target_retention)
        if target_retention is not None else float("nan")
    )
    return out


def retention_curve(items: ItemSet, rule: str, levels: Sequence[float]) -> Dict[str, float]:
    """FA at each of several matched positive-retention levels."""
    e = items.errors[rule]
    return {f"fa@ret{int(round(l*100))}": fa_at_matched_retention(e, items.labels, l) for l in levels}


def fa_by_case(items: ItemSet, rule: str) -> Dict[str, float]:
    """False acceptance broken down by the kind of negative.

    Reported because the negative classes are not equally hard: an absent mechanism
    produces no effect at all and any intervention-based rule rejects it, whereas an
    effect-matched random subspace is designed to defeat a mean-effect test.
    """
    e = items.errors[rule]
    out: Dict[str, float] = {}
    for case in NEGATIVE_CASES:
        mask = (items.case == case) & (items.labels == 0)
        if mask.any():
            out[case] = float((e[mask] <= 1.0).mean())
    return out


def analyse(
    records: Sequence[PathMeasurement],
    manifest,
    rules: Sequence[str] = tuple(LADDER),
    reference_rule: str = "pairwise_ccc_plus",
    n_boot: Optional[int] = None,
    seed: int = 7777,
) -> Dict[str, object]:
    """Full analysis: per-rule metrics with CIs, plus the planned contrasts."""
    rules = [r for r in rules if r in RULES]
    single_rules = [r for r in rules if not RULES[r].needs_two_destinations]
    cluster_levels = resolve_cluster_levels(manifest)
    items = build_item_set(records, single_rules, cluster_levels)
    multi_items, pairs = build_multi_items(records, rules, cluster_levels)
    n_boot = int(n_boot if n_boot is not None else manifest.path_of("statistics.bootstrap_replicates"))
    ci = float(manifest.path_of("statistics.ci"))
    target = float(manifest.path_of("statistics.matched_retention_target", 0.90))
    levels = list(manifest.path_of("statistics.retention_curve", [0.5, 0.7, 0.8, 0.9, 0.95]))

    ref_ret = target
    ref_ret_multi = target

    # ---- point estimates
    point: Dict[str, Dict[str, float]] = {}
    for r in single_rules:
        point[r] = rule_metrics(items, r, ref_ret)
    if multi_items.n:
        for r in rules:
            if RULES[r].needs_two_destinations:
                point[r] = rule_metrics(multi_items, r, ref_ret_multi)

    # ---- bootstrap
    metric_names = ["retention", "coverage", "false_acceptance", "auroc", "fa_at_matched_retention"]
    def stat_single(idx: np.ndarray) -> np.ndarray:
        sub = items.subset(idx)
        ref = target
        vals = []
        for r in single_rules:
            m = rule_metrics(sub, r, ref)
            vals += [m[k] for k in metric_names]
        return np.asarray(vals, dtype=float)

    boot = HierarchicalBootstrap(items.clusters, seed=seed)
    samples = boot.run(stat_single, n_boot)

    per_rule: Dict[str, Dict[str, Estimate]] = {}
    for i, r in enumerate(single_rules):
        per_rule[r] = {}
        for j, k in enumerate(metric_names):
            col = samples[:, i * len(metric_names) + j]
            per_rule[r][k] = summarise(col, point[r][k], ci)

    if multi_items.n:
        multi_rules = [r for r in rules if RULES[r].needs_two_destinations]

        def stat_multi(idx: np.ndarray) -> np.ndarray:
            sub = multi_items.subset(idx)
            ref = target
            vals = []
            for r in multi_rules:
                m = rule_metrics(sub, r, ref)
                vals += [m[k] for k in metric_names]
            return np.asarray(vals, dtype=float)

        boot_m = HierarchicalBootstrap(multi_items.clusters, seed=seed + 1)
        samples_m = boot_m.run(stat_multi, n_boot)
        for i, r in enumerate(multi_rules):
            per_rule[r] = {}
            for j, k in enumerate(metric_names):
                per_rule[r][k] = summarise(samples_m[:, i * len(metric_names) + j], point[r][k], ci)

    # ---- planned contrasts on FA at matched retention
    planned = [tuple(c) for c in manifest.path_of("planned_contrasts")]
    contrasts = _contrasts(planned, items, multi_items, target, n_boot, seed, ci)
    curves = {r: retention_curve(items, r, levels) for r in single_rules}
    if multi_items.n:
        for r in rules:
            if RULES[r].needs_two_destinations:
                curves[r] = retention_curve(multi_items, r, levels)

    return {
        "per_rule": per_rule,
        "point": point,
        "contrasts": contrasts,
        "fa_by_case": {r: fa_by_case(items, r) for r in single_rules},
        "n_items": items.n,
        "n_positives": int((items.labels == 1).sum()),
        "n_negatives": int((items.labels == 0).sum()),
        "n_multi_items": multi_items.n,
        "n_multi_positives": int((multi_items.labels == 1).sum()) if multi_items.n else 0,
        "matched_retention_target": target,
        "retention_curves": curves,
        "n_boot": n_boot,
        "cluster_levels": list(cluster_levels),
        "prompt_resampling": False,   # README deviation 6; stated, not implied
    }


def _contrasts(
    planned: Sequence[Tuple[str, str]],
    items: ItemSet,
    multi_items: ItemSet,
    target: float,
    n_boot: int,
    seed: int,
    ci: float,
) -> List[Dict[str, object]]:
    """Paired differences in FA at matched retention, bootstrapped on shared items."""
    out: List[Dict[str, object]] = []
    raw_p: List[float] = []
    for a, b in planned:
        use_multi = RULES.get(a, RULES["forward_only"]).needs_two_destinations or \
            RULES.get(b, RULES["forward_only"]).needs_two_destinations
        src = multi_items if use_multi else items
        if src.n == 0 or a not in src.errors or b not in src.errors:
            out.append({"a": a, "b": b, "estimate": Estimate(float("nan")), "unit": "multi" if use_multi else "path"})
            raw_p.append(float("nan"))
            continue

        def diff(idx: np.ndarray) -> np.ndarray:
            sub = src.subset(idx)
            fa_a = fa_at_matched_retention(sub.errors[a], sub.labels, target)
            fa_b = fa_at_matched_retention(sub.errors[b], sub.labels, target)
            return np.asarray([fa_a - fa_b], dtype=float)

        point = float(diff(np.arange(src.n))[0])
        boot = HierarchicalBootstrap(src.clusters, seed=seed + 17 + len(out))
        s = boot.run(diff, n_boot)[:, 0]
        est = summarise(s, point, ci, two_sided_zero=True)
        out.append({"a": a, "b": b, "estimate": est, "unit": "multi" if use_multi else "path"})
        raw_p.append(est.p)

    for rec, p_adj in zip(out, holm(raw_p)):
        rec["estimate"].p_adjusted = p_adj  # type: ignore
    return out


# ---------------------------------------------- coupled vs independent return
def coupling_stats(records: Sequence[PathMeasurement]) -> Dict[str, Dict[str, float]]:
    """Evidence for translator independence (Sec. 3.3).

    If the coupled return passes as often on negatives as on positives, the return
    leg is decoration: recovery was automatic. The independent return should
    separate the two.
    """
    out: Dict[str, Dict[str, float]] = {}
    by_tr: Dict[str, List[PathMeasurement]] = {}
    for m in records:
        if not m.abstained:
            by_tr.setdefault(m.translator, []).append(m)
    for tr, ms in by_tr.items():
        pos = [m for m in ms if m.label == 1]
        neg = [m for m in ms if m.label == 0]
        def rate(group, coupled):
            vals = [m.e_scalar_return(coupled) for m in group]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean([v <= 1.0 for v in vals])) if vals else float("nan")
        def cos(group, coupled):
            legs = [(m.ret_coupled if coupled else m.ret_independent) for m in group]
            vals = [l.subspace_cosine for l in legs if l.available and np.isfinite(l.subspace_cosine)]
            return float(np.mean(vals)) if vals else float("nan")
        out[tr] = {
            "n_positive": len(pos),
            "n_negative": len(neg),
            "coupled_pass_rate": rate(pos, True),
            "independent_pass_rate": rate(pos, False),
            "coupled_pass_rate_negatives": rate(neg, True),
            "independent_pass_rate_negatives": rate(neg, False),
            "coupled_subspace_cosine": cos(ms, True),
            "independent_subspace_cosine": cos(ms, False),
            "coupled_discrimination": rate(pos, True) - rate(neg, True),
            "independent_discrimination": rate(pos, False) - rate(neg, False),
        }
    return out


# --------------------------------------------------------- signature agreement
def signature_agreement(records: Sequence[PathMeasurement]) -> Dict[str, Dict[str, float]]:
    """Prompt-level agreement per translator (Table 5.4)."""
    out: Dict[str, Dict[str, float]] = {}
    by_tr: Dict[str, List[PathMeasurement]] = {}
    for m in records:
        if m.abstained or m.label != 1:
            continue
        by_tr.setdefault(m.translator, []).append(m)
    keys = ("effect_correlation", "sign_agreement", "effect_ratio",
            "worst_setting_error", "calibration_slope", "calibration_intercept")
    for tr, ms in by_tr.items():
        d: Dict[str, float] = {"n": float(len(ms))}
        for k in keys:
            vals = [m.diagnostics.get(k, np.nan) for m in ms]
            vals = [v for v in vals if np.isfinite(v)]
            d[k] = float(np.mean(vals)) if vals else float("nan")
        out[tr] = d
    return out


def rescore_with_bounds(
    records: Sequence[PathMeasurement], bounds_by_program: Dict[str, object]
) -> List[PathMeasurement]:
    """Re-score measurements under different equivalence bounds.

    The eps bounds enter the rules only as normalisers of already-measured distances
    (``d_shape``, ``d_mag``, the return scalar, the representational score), so a
    second bound mode is obtained by swapping the thresholds rather than repeating the
    interventions. That keeps the candidate set, the signatures and tau_min identical
    across modes, which is what makes the two modes comparable at all.

    ``tau_min`` is deliberately untouched: it is fitted from controls and is not a
    function of the bound mode.
    """
    out: List[PathMeasurement] = []
    for m in records:
        b = bounds_by_program.get(m.program)
        if b is None:
            out.append(m)
            continue
        c = copy.copy(m)
        c.bounds = {
            "eps_shape": b.eps_shape, "eps_mag": b.eps_mag,
            "eps_scalar": b.eps_scalar, "eps_representational": b.eps_representational,
        }
        out.append(c)
    return out


SET_VALUED_CASES = ("split", "split_half", "merged", "redundant", "shared", "permuted")


def set_valued_stats(
    records: Sequence[PathMeasurement], rule_name: str = "pairwise_ccc_plus"
) -> Dict[str, Dict[str, float]]:
    """Recovery for the planted split / merged / redundant cases (Sec. 4.1).

    Recall and precision are genuinely set-valued: they are computed over the set of
    candidates a rule *accepts* for one (source mechanism, destination), not by
    averaging a per-path similarity. A variable split over two destination blocks is
    only fully recovered when both are accepted, so accepting one half caps recall at
    one half -- which averaging a per-path similarity cannot express.

    * recall    -- mean over planted blocks of the best similarity among accepted
                   candidates assigned to that block (0 for an uncovered block)
    * precision -- mean over accepted candidates of the similarity to the block they
                   cover (0 for an accepted candidate covering no planted block)
    * set_exact -- fraction of decision contexts in which every planted block is
                   covered and nothing spurious is accepted
    """
    rule = RULES[rule_name]

    # One decision context = one source mechanism seen in one destination by one
    # translator/seed. Candidates within it compete to cover the planted blocks.
    ctx: Dict[Tuple, List[PathMeasurement]] = {}
    for m in records:
        ctx.setdefault(
            (m.program, m.source_mechanism, m.dest_model, m.translator, m.seed), []
        ).append(m)

    agg: Dict[str, Dict[str, List[float]]] = {}
    for _key, ms in ctx.items():
        n_blocks = int(max((m.diagnostics.get("n_truth_blocks", 0.0) for m in ms), default=0.0))
        if n_blocks <= 0:
            continue
        accepted = [m for m in ms if not m.abstained and rule.accepts(m)]

        # best similarity achieved per planted block, over accepted candidates
        covered = [0.0] * n_blocks
        precisions: List[float] = []
        for m in accepted:
            idx = int(m.diagnostics.get("truth_block_index", -1))
            sim = m.truth_subspace_similarity
            sim = float(sim) if np.isfinite(sim) else 0.0
            if 0 <= idx < n_blocks:
                covered[idx] = max(covered[idx], sim)
                precisions.append(sim)
            else:
                precisions.append(0.0)       # accepted, covers no planted block
        recall = float(np.mean(covered))
        precision = float(np.mean(precisions)) if precisions else 0.0
        exact = float(
            all(c >= CORRECT_CANDIDATE_SIMILARITY for c in covered)
            and all(p >= CORRECT_CANDIDATE_SIMILARITY for p in precisions)
            and len(precisions) > 0
        )

        # Attribute the context to the planted case its positives carry, so the row
        # label means "this planted structure" rather than "this decoy kind".
        cases = {m.case for m in ms if m.label == 1} or {m.case for m in ms}
        for case in cases:
            if case not in SET_VALUED_CASES:
                continue
            d = agg.setdefault(case, {"recall": [], "precision": [], "exact": [], "n": []})
            d["recall"].append(recall)
            d["precision"].append(precision)
            d["exact"].append(exact)
            d["n"].append(float(len(ms)))

    # component- and joint-level acceptance stay per-path, as before
    by_case: Dict[str, List[PathMeasurement]] = {}
    for m in records:
        by_case.setdefault(m.case, []).append(m)

    out: Dict[str, Dict[str, float]] = {}
    for case, ms in by_case.items():
        if case not in SET_VALUED_CASES:
            continue
        d = agg.get(case)
        pos = [m for m in ms if m.label == 1]
        out[case] = {
            "n": float(len(ms)),
            "n_contexts": float(len(d["recall"])) if d else 0.0,
            "component_pass": float(np.mean([rule.accepts(m) for m in ms])) if ms else float("nan"),
            "subspace_recall": float(np.mean(d["recall"])) if d else float("nan"),
            "subspace_precision": float(np.mean(d["precision"])) if d else float("nan"),
            "set_exact": float(np.mean(d["exact"])) if d else float("nan"),
            "joint_signature_pass": float(np.mean([rule.accepts(m) for m in pos]))
            if pos else float("nan"),
        }
    return out
