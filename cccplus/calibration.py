"""Estimation and calibration (Sec. 3.6).

Nothing here may look at test labels.

  tau_min^{M,T}   upper quantile of absolute effects produced by null-direction and
                  unrelated-task controls on the calibration split;
  c_{M,T}         median non-zero norm of source-gated calibration effects, with a
                  preregistered floor;
  eps_*           fixed from repeated identity and known-reparameterisation
                  controls, then tightened if necessary so that the false
                  acceptance rate on *labelled calibration negatives* stays under
                  the preregistered maximum.

Two calibration modes are provided. ``pooled`` uses all calibration records.
``leave_one_out`` holds out an entire program, fits the bounds on the remaining
programs, and applies them to the held-out one; this removes item-level overlap
between threshold selection and evaluation, which the pooled mode cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------- records
@dataclass
class ControlRecord:
    """One calibration observation used to set thresholds."""

    kind: str                  # null_direction | unrelated_task | identity |
                               # orthogonal_reparameterization | scale_reparameterization
    program: str
    model: str
    task: str
    abs_mean_effect: float = float("nan")
    signature_norm: float = float("nan")
    d_shape: float = float("nan")
    d_mag: float = float("nan")
    scalar_error: float = float("nan")
    representational: float = float("nan")


@dataclass
class NegativeRecord:
    """A labelled calibration negative: a candidate known not to be the counterpart."""

    program: str
    model: str
    task: str
    d_shape: float
    d_mag: float
    scalar_error: float
    representational: float
    d_shape_return: float = float("nan")
    d_mag_return: float = float("nan")


@dataclass
class PositiveRecord:
    """A labelled calibration positive: the planted counterpart, measured end to end.

    Used only by the ``cycle_aware`` bound mode. Sec. 3.6 fixes the bounds from
    within-model identity and reparameterisation controls, which understate the
    variation a *cross-model* round trip legitimately incurs: an independently
    fitted reverse map does not return the source subspace exactly even when the
    correspondence is real. Calibrating with that in view needs an estimate of how
    far a true correspondence lands, which is what these records provide. They come
    from the calibration split only; no test label is ever consulted.
    """

    program: str
    model: str
    task: str
    d_shape: float
    d_mag: float
    scalar_error: float
    representational: float
    d_shape_return: float = float("nan")
    d_mag_return: float = float("nan")


# ------------------------------------------------------------------ container
@dataclass
class Bounds:
    tau_min: Dict[str, float] = field(default_factory=dict)      # key: f"{model}|{task}"
    scale: Dict[str, float] = field(default_factory=dict)        # c_{M,T}
    eps_shape: float = float("nan")
    eps_mag: float = float("nan")
    eps_scalar: float = float("nan")
    eps_representational: float = float("nan")
    provenance: Dict[str, object] = field(default_factory=dict)

    def tau(self, model: str, task: str, default: float = 0.0) -> float:
        return float(self.tau_min.get(f"{model}|{task}", default))

    def c(self, model: str, task: str, default: float = 1.0) -> float:
        return float(self.scale.get(f"{model}|{task}", default))

    def to_dict(self) -> Dict[str, object]:
        return {
            "tau_min": self.tau_min,
            "scale": self.scale,
            "eps_shape": self.eps_shape,
            "eps_mag": self.eps_mag,
            "eps_scalar": self.eps_scalar,
            "eps_representational": self.eps_representational,
            "provenance": self.provenance,
        }


# ----------------------------------------------------------------- estimators
def _q(values: Iterable[float], q: float, default: float = float("nan")) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return default
    return float(np.quantile(v, q))


def estimate_tau_min(
    controls: Sequence[ControlRecord], quantile: float, kinds: Sequence[str]
) -> Dict[str, float]:
    """Per (model, task) minimum effect size, from control directions only."""
    buckets: Dict[str, List[float]] = {}
    for c in controls:
        if c.kind in kinds and np.isfinite(c.abs_mean_effect):
            buckets.setdefault(f"{c.model}|{c.task}", []).append(c.abs_mean_effect)
    return {k: _q(v, quantile, 0.0) for k, v in buckets.items()}


def estimate_scale(
    gated_norms: Dict[str, Sequence[float]], floor_quantile: float, min_floor: float
) -> Dict[str, float]:
    """c_{M,T}: median non-zero signature norm among source-gated mechanisms."""
    out: Dict[str, float] = {}
    for key, vals in gated_norms.items():
        v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], dtype=float)
        if v.size == 0:
            out[key] = float(min_floor)
            continue
        median = float(np.median(v))
        floor = max(float(np.quantile(v, floor_quantile)), float(min_floor))
        out[key] = float(max(median, floor))
    return out


def _bound_from(
    control_values: Sequence[float],
    negative_values: Sequence[float],
    control_quantile: float,
    max_fa: float,
    fallback: float,
    positive_values: Sequence[float] = (),
    target_retention: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Control-derived bound, tightened to respect the calibration FA budget.

    ``controls_only`` (target_retention = 0) is the literal Sec. 3.6 rule.
    ``cycle_aware`` (target_retention > 0) additionally floors the bound at the
    quantile of labelled calibration *positives* needed to retain that fraction of
    them, which stops a within-model noise floor from rejecting real cross-model
    correspondences. The negative cap still applies, and if the two requirements
    conflict the conflict is recorded rather than silently resolved.
    """
    cand = _q(control_values, control_quantile, fallback)
    if not np.isfinite(cand):
        cand = fallback
    neg = np.asarray([x for x in negative_values if np.isfinite(x)], dtype=float)
    pos = np.asarray([x for x in positive_values if np.isfinite(x)], dtype=float)
    info = {
        "control_quantile_value": float(cand),
        "n_controls": int(len(control_values)),
        "n_negatives": int(neg.size),
        "n_positives": int(pos.size),
    }
    floor = float("nan")
    if target_retention > 0 and pos.size >= 5:
        floor = float(np.quantile(pos, min(target_retention, 1.0)))
        info["positive_floor"] = floor
        cand = max(cand, floor)
    cap = float("nan")
    if neg.size >= 10:
        cap = float(np.quantile(neg, max_fa))
        info["negative_cap"] = cap
        if np.isfinite(cap):
            if np.isfinite(floor) and cap < floor:
                info["cap_conflicts_with_floor"] = True
                cand = floor      # keep the method usable; the realised FA is reported
            else:
                cand = min(cand, cap)
    return float(max(cand, 1e-12)), info


def fit_bounds(
    controls: Sequence[ControlRecord],
    negatives: Sequence[NegativeRecord],
    gated_norms: Dict[str, Sequence[float]],
    manifest,
    positives: Sequence["PositiveRecord"] = (),
    mode: str = "controls_only",
) -> Bounds:
    """Fit every threshold from calibration controls, negatives and (optionally) positives."""
    tau_q = float(manifest.path_of("source_gate.tau_min_estimator.quantile"))
    tau_kinds = list(manifest.path_of("source_gate.tau_min_estimator.controls"))
    ctrl_q = float(manifest.path_of("equivalence_bounds.control_quantile"))
    max_fa = float(manifest.path_of("equivalence_bounds.max_calibration_false_acceptance"))
    eq_kinds = set(manifest.path_of("equivalence_bounds.controls"))

    eq_controls = [c for c in controls if c.kind in eq_kinds]
    target = float(manifest.path_of("equivalence_bounds.target_positive_retention", 0.90)) \
        if mode == "cycle_aware" else 0.0
    # The bound must admit a true correspondence on *both* legs, so the positive
    # evidence pools the forward and return distances.
    pos_shape = [p.d_shape for p in positives] + [p.d_shape_return for p in positives]
    pos_mag = [p.d_mag for p in positives] + [p.d_mag_return for p in positives]
    eps_shape, i_shape = _bound_from(
        [c.d_shape for c in eq_controls], [n.d_shape for n in negatives], ctrl_q, max_fa, 0.2,
        pos_shape, target,
    )
    eps_mag, i_mag = _bound_from(
        [c.d_mag for c in eq_controls], [n.d_mag for n in negatives], ctrl_q, max_fa, 0.2,
        pos_mag, target,
    )
    eps_scalar, i_scalar = _bound_from(
        [c.scalar_error for c in eq_controls], [n.scalar_error for n in negatives], ctrl_q, max_fa, 0.5,
        [p.scalar_error for p in positives], target,
    )
    # Representational rule accepts *above* a similarity threshold, so the FA budget
    # is applied from the other tail.
    # The representational rule accepts *above* a similarity threshold, so the
    # inequalities run the opposite way to the distance bounds: the calibration FA
    # budget puts a *lower* bound on the threshold (lowering it admits more
    # negatives), while the controls put an upper bound (demanding more similarity
    # than genuinely equivalent pairs exhibit would reject them). Taking min() here
    # -- as an earlier version did -- lowers the threshold after the FA constraint
    # and can therefore violate the 5% budget.
    neg_repr = np.asarray([n.representational for n in negatives if np.isfinite(n.representational)], dtype=float)
    repr_info: Dict[str, float] = {"n_negatives": int(neg_repr.size)}
    eps_repr = 0.5
    if neg_repr.size >= 10:
        eps_repr = float(np.quantile(neg_repr, 1.0 - max_fa))   # FA floor
        repr_info["negative_floor"] = eps_repr
    ctrl_repr = [c.representational for c in eq_controls if np.isfinite(c.representational)]
    if ctrl_repr:
        ceiling = _q(ctrl_repr, 1.0 - ctrl_q, float("inf"))     # control ceiling
        repr_info["control_ceiling"] = float(ceiling)
        if np.isfinite(ceiling) and ceiling < eps_repr:
            repr_info["ceiling_conflicts_with_floor"] = True    # keep the FA budget
        else:
            eps_repr = float(min(eps_repr, ceiling)) if np.isfinite(ceiling) else eps_repr

    return Bounds(
        tau_min=estimate_tau_min(controls, tau_q, tau_kinds),
        scale=estimate_scale(
            gated_norms,
            float(manifest.path_of("signature.scale_estimator.floor_quantile")),
            float(manifest.path_of("signature.scale_estimator.min_floor")),
        ),
        eps_shape=eps_shape,
        eps_mag=eps_mag,
        eps_scalar=eps_scalar,
        eps_representational=eps_repr,
        provenance={
            "mode": mode,
            "shape": i_shape, "magnitude": i_mag, "scalar": i_scalar,
            "representational": repr_info,
            "tau_quantile": tau_q, "control_quantile": ctrl_q, "max_calibration_fa": max_fa,
            "n_controls": len(controls), "n_negatives": len(negatives),
        },
    )


class Calibrator:
    """Holds calibration evidence and produces pooled or leave-one-program-out bounds."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.controls: List[ControlRecord] = []
        self.negatives: List[NegativeRecord] = []
        self.positives: List[PositiveRecord] = []
        self.gated_norms: Dict[str, List[float]] = {}
        self.mode: str = "controls_only"

    def add_control(self, rec: ControlRecord) -> None:
        self.controls.append(rec)

    def add_negative(self, rec: NegativeRecord) -> None:
        self.negatives.append(rec)

    def add_positive(self, rec: PositiveRecord) -> None:
        self.positives.append(rec)

    def add_gated_norm(self, model: str, task: str, norm: float) -> None:
        self.gated_norms.setdefault(f"{model}|{task}", []).append(float(norm))

    def pooled(self, mode: Optional[str] = None) -> Bounds:
        b = fit_bounds(self.controls, self.negatives, self.gated_norms, self.manifest,
                       self.positives, mode or self.mode)
        b.provenance["split_mode"] = "pooled"
        return b

    def leave_one_program_out(self, program: str, mode: Optional[str] = None) -> Bounds:
        """Bounds fitted without any record from ``program``."""
        ctrl = [c for c in self.controls if c.program != program]
        neg = [n for n in self.negatives if n.program != program]
        pos = [p for p in self.positives if p.program != program]
        # c_{M,T} is keyed per model and model names are program-scoped, so the
        # held-out program's norms cannot reach its own scale in practice; they are
        # filtered anyway so the leave-one-out invariant holds by construction
        # rather than by a naming coincidence.
        norms = {k: v for k, v in self.gated_norms.items() if not k.startswith(f"{program}/")}
        b = fit_bounds(ctrl, neg, norms, self.manifest, pos, mode or self.mode)
        b.provenance["split_mode"] = "leave_one_program_out"
        b.provenance["held_out_program"] = program
        return b

    def bounds_for(self, program: str, split_mode: str = "leave_one_program_out",
                   mode: Optional[str] = None) -> Bounds:
        return self.pooled(mode) if split_mode == "pooled" else self.leave_one_program_out(program, mode)

    def summary(self) -> Dict[str, object]:
        kinds: Dict[str, int] = {}
        for c in self.controls:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        return {"n_controls": len(self.controls), "control_kinds": kinds,
                "n_negatives": len(self.negatives), "n_positives": len(self.positives)}
