"""The validation ladder: forward gate, destination test, independent round trip.

Equations implemented here
  Eq. 7   candidate            m1^{(j)} = T^F_{1->j}(m1)
  Eq. 8   forward scalar gate  |Delta_bar(Mj, m1^{(j)})| >= tau_min, signs agree
  Eq. 9   destination test     d_shape <= eps_shape and d_mag <= eps_mag
  Eq. 10  return               m1,R^{(j->1)} = T^R_{j->1}(m1^{(j)})
  Eq. 11  return test          d_shape, d_mag of the returned signature
  Eq. 12  scalar return error  |Delta_bar(M1, returned) - Delta_bar(M1, m1)|
  Eq. 13  normalised errors    e^F_j, e^R_j
  Eq. 14  scores               CCC+_j, CCC+_multi

The cumulative ladder of Sec. 4.5 / 5.3, each row adding exactly one condition:

  rule                  Eq.8  Eq.9 shape  Eq.9 mag  return map    return test
  representational       -        -           -          -        similarity only
  round_trip_only        -        -           -       coupled     scalar
  forward_only           x        -           -          -           -
  scalar_ccc             x        -           -       coupled     scalar
  ccc_ind                x        -           -      independent  scalar
  ccc_shape              x        x           -      independent  scalar
  ccc_sig                x        x           x      independent  scalar
  pairwise_ccc_plus      x        x           x      independent  signature
  multi_model_ccc_plus   x        x           x      independent  signature, both j

Every rule exposes a single normalised ``error``: the maximum over its own
conditions of (observed quantity / its bound). A rule accepts exactly when
error <= 1, so sweeping the threshold on ``error`` traces that rule's whole
operating curve, which is what "false acceptance at matched positive retention"
requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .calibration import Bounds
from .interventions import CausalProbe, Setting
from .mechanisms import Mechanism, Site, subspace_similarity
from .signature import (
    CausalSignature,
    ccc_plus_multi,
    ccc_plus_score,
    compare_signatures,
    scalar_return_error,
    signature_diagnostics,
)

BIG = 1e6   # normalised error assigned to a hard failure (abstention, sign flip)

# Preregistered: a proposal counts as the planted counterpart at or above this
# subspace similarity. Lives here rather than in a pipeline module because both the
# candidate labelling and the set-valued recovery metrics have to agree on it.
CORRECT_CANDIDATE_SIMILARITY = 0.70


# --------------------------------------------------------- representational
def subspace_activation_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean canonical correlation between in-subspace activation coordinates.

    This is the acceptance signal available to a purely representational rule: the
    translated feature "activates on the same prompts" without any intervention
    being performed. It is translator-agnostic, so every translator is judged by
    the same representational criterion.
    """
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    if A.shape[0] < 4 or A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    def whiten(X: torch.Tensor) -> Optional[torch.Tensor]:
        U, S, _ = torch.linalg.svd(X, full_matrices=False)
        keep = S > 1e-8 * float(S.max()) if float(S.max()) > 0 else S > 0
        return U[:, keep] if bool(keep.any()) else None
    Ua, Ub = whiten(A), whiten(B)
    if Ua is None or Ub is None:
        return 0.0
    s = torch.linalg.svdvals(Ua.T @ Ub).clamp(0.0, 1.0)
    k = min(A.shape[1], B.shape[1], s.numel())
    return float(s[:k].mean())


# ------------------------------------------------------------- measurements
@dataclass
class ReturnLeg:
    """Result of mapping a candidate back into the source model."""

    available: bool = False
    site: Optional[str] = None
    site_matches_source: bool = False
    mean_effect: float = float("nan")
    scalar_error: float = float("nan")
    d_shape: float = float("nan")
    d_mag: float = float("nan")
    subspace_cosine: float = float("nan")   # returned-subspace cosine (Sec 3.3 diagnostic)
    norm_error: float = float("nan")        # |log| norm gain of the composed map (see below)
    same_atom: Optional[bool] = None

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class PathMeasurement:
    """All quantities needed to evaluate every rule on one translation path.

    A 'path' is (source mechanism in M1) -> (candidate in Mj) -> (return into M1)
    for one translator and one seed.
    """

    program: str
    task: str
    source_model: str
    dest_model: str
    translator: str
    seed: int
    source_mechanism: str
    case: str                       # planted case or decoy kind
    label: int                      # 1 = the candidate is the planted counterpart
    family_key: str = ""
    candidate: Optional[str] = None
    abstained: bool = True
    abstention_reason: str = ""

    # source side
    mean_effect_source: float = float("nan")
    source_gate_passed: bool = False

    # forward leg
    mean_effect_dest: float = float("nan")
    sign_agrees: bool = False
    tau_dest: float = float("nan")
    d_shape_forward: float = float("nan")
    d_mag_forward: float = float("nan")

    # returns
    ret_independent: ReturnLeg = field(default_factory=ReturnLeg)
    ret_coupled: ReturnLeg = field(default_factory=ReturnLeg)

    # representational + diagnostics
    representational: float = float("nan")
    truth_subspace_similarity: float = float("nan")
    diagnostics: Dict[str, float] = field(default_factory=dict)
    collateral: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------- normalised
    def e_forward(self) -> float:
        """Eq. 13 forward error."""
        if self.abstained:
            return BIG
        es, em = self.bounds.get("eps_shape", np.nan), self.bounds.get("eps_mag", np.nan)
        if not (np.isfinite(self.d_shape_forward) and np.isfinite(self.d_mag_forward)):
            return BIG
        return max(self.d_shape_forward / max(es, 1e-12), self.d_mag_forward / max(em, 1e-12))

    def e_return(self, coupled: bool = False) -> float:
        """Eq. 13 return error."""
        leg = self.ret_coupled if coupled else self.ret_independent
        if self.abstained or not leg.available:
            return BIG
        es, em = self.bounds.get("eps_shape", np.nan), self.bounds.get("eps_mag", np.nan)
        if not (np.isfinite(leg.d_shape) and np.isfinite(leg.d_mag)):
            return BIG
        return max(leg.d_shape / max(es, 1e-12), leg.d_mag / max(em, 1e-12))

    def e_gate(self) -> float:
        """Forward scalar gate as a normalised quantity (<= 1 iff Eq. 8 holds)."""
        if self.abstained:
            return BIG
        if not self.sign_agrees:
            return BIG
        denom = abs(self.mean_effect_dest)
        if not np.isfinite(denom) or denom <= 0:
            return BIG
        tau = self.tau_dest
        if not np.isfinite(tau):
            return BIG
        return float(tau / denom)

    def e_scalar_return(self, coupled: bool) -> float:
        leg = self.ret_coupled if coupled else self.ret_independent
        if self.abstained or not leg.available or not np.isfinite(leg.scalar_error):
            return BIG
        return float(leg.scalar_error / max(self.bounds.get("eps_scalar", np.nan), 1e-12))

    def e_representational(self) -> float:
        if self.abstained or not np.isfinite(self.representational):
            return BIG
        eps = self.bounds.get("eps_representational", np.nan)
        if not np.isfinite(eps):
            return BIG
        return float(eps / max(self.representational, 1e-12))

    def ccc_plus_score(self) -> float:
        """Eq. 14 pairwise score."""
        return ccc_plus_score(min(self.e_forward(), BIG), min(self.e_return(False), BIG))

    def to_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("ret_independent", "ret_coupled")}
        d["ret_independent"] = self.ret_independent.to_dict()
        d["ret_coupled"] = self.ret_coupled.to_dict()
        d["ccc_plus_score"] = self.ccc_plus_score()
        d["e_forward"] = self.e_forward()
        d["e_return"] = self.e_return(False)
        d["e_return_coupled"] = self.e_return(True)
        d["e_gate"] = self.e_gate()
        return d


# ---------------------------------------------------------------- the rules
@dataclass
class Rule:
    """A validation rule: a set of normalised conditions combined by max()."""

    name: str
    conditions: Sequence[str]
    needs_two_destinations: bool = False
    description: str = ""
    enforce_source_gate: bool = True

    def error(self, m: PathMeasurement) -> float:
        conds = list(self.conditions)
        if self.enforce_source_gate:
            conds = ["source_gate"] + conds
        vals = [_CONDITIONS[c](m) for c in conds]
        return float(max(vals)) if vals else 0.0

    def accepts(self, m: PathMeasurement) -> bool:
        return self.error(m) <= 1.0

    def score(self, m: PathMeasurement) -> float:
        return float(np.exp(-min(self.error(m), 50.0)))


_CONDITIONS: Dict[str, Callable[[PathMeasurement], float]] = {
    # Sec. 3.1: "A source mechanism is eligible for translation only when
    # |Delta_bar| >= tau_min and its sign agrees with a preregistered task
    # prediction." This is a precondition on the *path*, not one rule's option, so
    # it is enforced for every rule: otherwise a downstream failure could be blamed
    # on translation when the starting mechanism was never causally established.
    "source_gate": lambda m: (0.0 if m.source_gate_passed else BIG),
    "representational": lambda m: m.e_representational(),
    "forward_gate": lambda m: m.e_gate(),
    "dest_shape": lambda m: (
        BIG if m.abstained or not np.isfinite(m.d_shape_forward)
        else m.d_shape_forward / max(m.bounds.get("eps_shape", 1e-12), 1e-12)
    ),
    "dest_mag": lambda m: (
        BIG if m.abstained or not np.isfinite(m.d_mag_forward)
        else m.d_mag_forward / max(m.bounds.get("eps_mag", 1e-12), 1e-12)
    ),
    "scalar_return_coupled": lambda m: m.e_scalar_return(True),
    "scalar_return_independent": lambda m: m.e_scalar_return(False),
    "signature_return_independent": lambda m: m.e_return(False),
}


RULES: Dict[str, Rule] = {
    "representational": Rule(
        "representational", ["representational"],
        description="similarity only, no intervention",
    ),
    "round_trip_only": Rule(
        "round_trip_only", ["scalar_return_coupled"],
        description="coupled scalar return applied to every non-null candidate, no forward gate",
    ),
    "forward_only": Rule(
        "forward_only", ["forward_gate"],
        description="Eq. 8 in the destination only",
    ),
    "scalar_ccc": Rule(
        "scalar_ccc", ["forward_gate", "scalar_return_coupled"],
        description="original scalar CCC: forward gate + coupled scalar return",
    ),
    "ccc_ind": Rule(
        "ccc_ind", ["forward_gate", "scalar_return_independent"],
        description="+ translator independence",
    ),
    "ccc_shape": Rule(
        "ccc_shape", ["forward_gate", "dest_shape", "scalar_return_independent"],
        description="+ destination signature shape",
    ),
    "ccc_sig": Rule(
        "ccc_sig", ["forward_gate", "dest_shape", "dest_mag", "scalar_return_independent"],
        description="+ destination magnitude equivalence (Eq. 9 complete)",
    ),
    "pairwise_ccc_plus": Rule(
        "pairwise_ccc_plus",
        ["forward_gate", "dest_shape", "dest_mag", "signature_return_independent"],
        description="+ independent signature return (Eq. 11)",
    ),
    "multi_model_ccc_plus": Rule(
        "multi_model_ccc_plus",
        ["forward_gate", "dest_shape", "dest_mag", "signature_return_independent"],
        needs_two_destinations=True,
        description="+ conjunction over two independently trained destinations",
    ),
}

LADDER = [
    "representational", "round_trip_only", "forward_only", "scalar_ccc",
    "ccc_ind", "ccc_shape", "ccc_sig", "pairwise_ccc_plus", "multi_model_ccc_plus",
]


# ------------------------------------------------------- multi-model records
@dataclass
class MultiPath:
    """A conjunction item: the same source mechanism through two destinations."""

    left: PathMeasurement
    right: PathMeasurement

    @property
    def label(self) -> int:
        return int(self.left.label == 1 and self.right.label == 1)

    @property
    def abstained(self) -> bool:
        return self.left.abstained or self.right.abstained

    def error(self, rule: Rule) -> float:
        return max(rule.error(self.left), rule.error(self.right))

    def score(self) -> float:
        return ccc_plus_multi(self.left.ccc_plus_score(), self.right.ccc_plus_score())

    def key(self) -> str:
        return (
            f"{self.left.program}|{self.left.source_mechanism}|{self.left.translator}|"
            f"{self.left.seed}|{self.left.dest_model}+{self.right.dest_model}"
        )


# ------------------------------------------------------------- measurement
def measure_path(
    *,
    program: str,
    task_name: str,
    source_probe: CausalProbe,
    dest_probe: CausalProbe,
    settings: Sequence[Setting],
    source_mech: Mechanism,
    source_signature: CausalSignature,
    candidate: Optional[Mechanism],
    reverse_independent: Callable[[Mechanism], Optional[Mechanism]],
    reverse_coupled: Optional[Callable[[Mechanism], Optional[Mechanism]]],
    bounds: Bounds,
    translator: str,
    seed: int,
    case: str,
    label: int,
    truth_blocks: Sequence[Mechanism] = (),
    repr_source: Optional[torch.Tensor] = None,
    repr_dest_acts: Optional[Callable[[Mechanism], torch.Tensor]] = None,
    abstention_reason: str = "",
    family_key: str = "",
    collateral: bool = False,
    weight_scheme: str = "setting_balanced",
    eta: float = 1e-6,
    sig_fn: Optional[Callable[[CausalProbe, Mechanism, float], CausalSignature]] = None,
) -> PathMeasurement:
    """Run the full protocol on one path and record every quantity the rules need."""
    measure = sig_fn or (lambda probe, mech, scale: CausalSignature.measure(probe, mech, settings, scale=scale))
    src_model = source_probe.model.name
    dst_model = dest_probe.model.name
    m = PathMeasurement(
        program=program, task=task_name, source_model=src_model, dest_model=dst_model,
        translator=translator, seed=seed, source_mechanism=source_mech.key(),
        case=case, label=label, family_key=family_key,
        mean_effect_source=source_signature.mean_effect(),
        bounds={
            "eps_shape": bounds.eps_shape, "eps_mag": bounds.eps_mag,
            "eps_scalar": bounds.eps_scalar, "eps_representational": bounds.eps_representational,
        },
    )
    m.tau_dest = bounds.tau(dst_model, task_name)
    m.source_gate_passed = (
        abs(m.mean_effect_source) >= bounds.tau(src_model, task_name)
        and np.sign(m.mean_effect_source) == np.sign(source_probe.task.predicted_sign)
    )
    if candidate is None:
        m.abstained = True
        m.abstention_reason = abstention_reason or "null_translation"
        return m

    m.abstained = False
    m.candidate = candidate.key()

    # ---- forward leg
    c_dest = bounds.c(dst_model, task_name)
    dest_sig = measure(dest_probe, candidate, c_dest)
    m.mean_effect_dest = dest_sig.mean_effect()
    m.sign_agrees = bool(np.sign(m.mean_effect_dest) == np.sign(m.mean_effect_source)) and m.mean_effect_dest != 0.0
    cmp_fwd = compare_signatures(source_signature, dest_sig, bounds.eps_shape, bounds.eps_mag, eta, weight_scheme)
    m.d_shape_forward, m.d_mag_forward = cmp_fwd.shape, cmp_fwd.magnitude
    m.diagnostics.update(signature_diagnostics(source_signature, dest_sig))

    if truth_blocks:
        sims = [
            subspace_similarity(candidate.V, t.V) if str(candidate.site) == str(t.site) else 0.0
            for t in truth_blocks
        ]
        m.truth_subspace_similarity = max(sims)
        # Which planted block this candidate covers, and how many there are. Set-valued
        # recovery is a property of the *set* of accepted candidates for one source
        # mechanism, so the per-path assignment has to be recorded here for the
        # analysis layer to aggregate it (Sec. 4.1).
        m.diagnostics["n_truth_blocks"] = float(len(truth_blocks))
        m.diagnostics["truth_block_index"] = float(int(np.argmax(sims)) if max(sims) > 0.0 else -1)

    if repr_source is not None and repr_dest_acts is not None:
        try:
            m.representational = subspace_activation_similarity(repr_source, repr_dest_acts(candidate))
        except Exception:
            m.representational = float("nan")
    else:
        m.representational = float(candidate.meta.get("representational_score", float("nan")))

    if collateral:
        m.collateral = dest_probe.collateral(candidate, settings[0])

    # ---- return legs
    c_src = bounds.c(src_model, task_name)
    for coupled, fn in ((False, reverse_independent), (True, reverse_coupled)):
        if fn is None:
            continue
        returned = fn(candidate)
        leg = m.ret_coupled if coupled else m.ret_independent
        if returned is None:
            leg.available = False
            continue
        ret_sig = measure(source_probe, returned, c_src)
        cmp_ret = compare_signatures(source_signature, ret_sig, bounds.eps_shape, bounds.eps_mag, eta, weight_scheme)
        leg.available = True
        leg.site = str(returned.site)
        leg.site_matches_source = str(returned.site) == str(source_mech.site)
        leg.mean_effect = ret_sig.mean_effect()
        leg.scalar_error = scalar_return_error(leg.mean_effect, m.mean_effect_source)
        leg.d_shape, leg.d_mag = cmp_ret.shape, cmp_ret.magnitude
        leg.subspace_cosine = (
            subspace_similarity(returned.V, source_mech.V) if leg.site_matches_source else 0.0
        )
        # Norm error must measure the *map's* distortion, not the bases': both V are
        # orthonormal, so their Frobenius norms are sqrt(rank) by construction and
        # differencing them is identically zero. Translators therefore record the
        # norm gain of the mapped subspace before re-orthonormalisation, and the
        # diagnostic is its absolute log deviation from unity.
        gain = returned.meta.get("map_gain")
        leg.norm_error = float(abs(np.log(max(float(gain), 1e-12)))) if gain else float("nan")
        leg.same_atom = bool(returned.meta.get("coupled")) if coupled else None
    return m
