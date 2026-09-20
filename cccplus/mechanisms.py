"""Component mechanisms m = (s, V) and mechanism sites (Sec. 3.1).

"A component mechanism in model M is a tuple m = (s, V). The site s identifies a
layer, sublayer, and token position, while V in R^{d_s x r} contains one or more
activation-space directions. [...] We take the columns of V to be orthonormal.
This definition separates where an intervention occurs from what activation
component it changes."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True, order=True)
class Site:
    """Layer / sublayer / token-position triple.

    ``token`` is a *resolved* index for fixed-length synthetic prompts, or a
    symbolic anchor (e.g. ``"final"``, ``"subject_last"``, ``"io"``) for natural
    prompts where the index depends on tokenisation.
    """

    layer: int
    sublayer: str            # 'resid_pre' | 'attn_out' | 'mlp_out' | 'resid_post' | 'head_out'
    token: Any = "final"     # int index or symbolic anchor
    head: Optional[int] = None   # only for sublayer == 'head_out'

    def __str__(self) -> str:
        h = "" if self.head is None else f".h{self.head}"
        return f"L{self.layer}.{self.sublayer}{h}@{self.token}"

    def to_dict(self) -> Dict[str, Any]:
        return {"layer": self.layer, "sublayer": self.sublayer, "token": self.token, "head": self.head}

    def with_token(self, token: Any) -> "Site":
        return Site(self.layer, self.sublayer, token, self.head)


def orthonormalize(V: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    """Return an orthonormal basis for the column space of ``V`` (d x r).

    Uses a thin QR with a rank check; rank-deficient inputs are reduced to their
    numerical rank, which matters for the 'merged' and 'redundant' planted cases
    where proposed directions can be nearly collinear.
    """
    if V.ndim == 1:
        V = V.unsqueeze(1)
    V = V.to(torch.float64)
    # Drop numerically zero columns before QR so that rank detection is stable.
    norms = V.norm(dim=0)
    keep = norms > tol * max(1.0, float(norms.max()))
    if keep.any():
        V = V[:, keep]
    else:
        raise ValueError("cannot orthonormalize an all-zero matrix")
    Q, R = torch.linalg.qr(V)
    diag = R.diagonal().abs()
    rank_keep = diag > tol * max(1.0, float(diag.max()))
    Q = Q[:, rank_keep]
    # Fix sign convention so that bases are comparable run to run.
    signs = torch.sign(Q.sum(dim=0))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q.to(torch.float32)


@dataclass
class Mechanism:
    """A causal mechanism: a subspace ``V`` (orthonormal columns) at a site."""

    site: Site
    V: torch.Tensor
    model: str = ""
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.V.ndim == 1:
            self.V = self.V.unsqueeze(1)
        self.V = orthonormalize(self.V)

    @property
    def rank(self) -> int:
        return int(self.V.shape[1])

    @property
    def d(self) -> int:
        return int(self.V.shape[0])

    def projector(self) -> torch.Tensor:
        """P = V V^T, the projection used by the directional patch (Eq. 6)."""
        return self.V @ self.V.T

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project activations (..., d) onto the mechanism subspace."""
        return (x @ self.V) @ self.V.T

    def key(self) -> str:
        return f"{self.model}:{self.site}:r{self.rank}" + (f":{self.label}" if self.label else "")

    def fingerprint(self) -> str:
        """Stable identity of (site, subspace), used to cache causal measurements.

        Keyed on the projector rather than the basis, so two bases of the same
        subspace share a fingerprint -- which is correct, because they induce
        identical interventions under Eq. 6.
        """
        import hashlib

        P = (self.projector().detach().to(torch.float32) * 1e4).round().to(torch.int32)
        h = hashlib.sha1(P.numpy().tobytes()).hexdigest()[:16]
        return f"{self.model}|{self.site}|r{self.rank}|{h}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "site": self.site.to_dict(),
            "rank": self.rank,
            "d": self.d,
            "label": self.label,
            "meta": {k: v for k, v in self.meta.items() if _jsonable(v)},
        }


def _jsonable(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool, list, tuple, dict, type(None)))


def subspace_cosines(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal cosines between two orthonormal bases (singular values of A^T B)."""
    A = orthonormalize(A).to(torch.float64)
    B = orthonormalize(B).to(torch.float64)
    s = torch.linalg.svdvals(A.T @ B)
    return s.clamp(0.0, 1.0).to(torch.float32)


def subspace_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean principal cosine: 1.0 iff the subspaces coincide.

    For rank 1 this is |cos(a, b)|, which is the quantity a representational
    acceptance rule would use.
    """
    if A.numel() == 0 or B.numel() == 0:
        return 0.0
    c = subspace_cosines(A, B)
    return float(c.mean())


def principal_angle_error(A: torch.Tensor, B: torch.Tensor) -> float:
    """1 - mean principal cosine; the 'returned-subspace cosine' error of Sec 3.3."""
    return 1.0 - subspace_similarity(A, B)


def set_valued_recovery(
    proposed: Sequence[Mechanism], truth: Sequence[Mechanism]
) -> Dict[str, float]:
    """Subspace recovery for the planted split / merged / redundant cases (Sec 4.1).

    "Set-valued cases are scored by subspace and joint-signature recovery."

    Returns precision / recall style quantities computed on *stacked* subspaces at
    matching sites, so a source variable split over two destination sites is only
    fully recovered when both components are proposed.
    """
    if not truth:
        return {"subspace_recall": float("nan"), "subspace_precision": float("nan"), "site_jaccard": float("nan")}
    prop_sites = {str(m.site) for m in proposed}
    true_sites = {str(m.site) for m in truth}
    inter = prop_sites & true_sites
    union = prop_sites | true_sites
    recalls, precisions = [], []
    for t in truth:
        cands = [m for m in proposed if str(m.site) == str(t.site)]
        recalls.append(max((subspace_similarity(t.V, c.V) for c in cands), default=0.0))
    for p in proposed:
        cands = [m for m in truth if str(m.site) == str(p.site)]
        precisions.append(max((subspace_similarity(p.V, c.V) for c in cands), default=0.0))
    return {
        "subspace_recall": float(sum(recalls) / len(recalls)),
        "subspace_precision": float(sum(precisions) / max(1, len(precisions))),
        "site_jaccard": float(len(inter) / max(1, len(union))),
    }
