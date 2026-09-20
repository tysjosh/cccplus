"""Translation operators and the independence discipline (Sec. 3.3).

    T_{A->B} : M(A) -> M(B) union {null}

"The null output explicitly records that no correspondence was found."

Independence (Sec. 3.3, 'Independent reverse estimation') is enforced
structurally rather than by convention:

  * the probe corpus is split by document *before* either direction is fitted;
  * the forward translator only ever sees the forward half, the reverse only the
    reverse half;
  * each direction gets its own random initialisation and its own parameters;
  * for dictionary methods the reverse leg uses a *second, independently trained*
    dictionary, so atom identifiers are not shared between legs.

The coupled construction (transposed Procrustes map, or the same crosscoder atom
in both directions) is kept as an explicit ablation, because "this construction
can make geometric recovery automatic".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..mechanisms import Mechanism, Site
from ..models.base import HookedModel, PromptBatch


# --------------------------------------------------------------- activations
@dataclass
class ActivationStore:
    """Activations of one model at a set of sites over a shared prompt corpus."""

    model_name: str
    sites: List[Site]
    acts: Dict[str, torch.Tensor]        # str(site) -> [N, d]
    #: Identity of each row in the *original* corpus. Subsetting preserves it, so two
    #: legs fitted on disjoint halves record disjoint ids. Without this, each leg would
    #: report local indices 0..N and two genuinely disjoint fits would appear to share
    #: every row.
    row_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.row_ids is None:
            self.row_ids = np.arange(self.n)

    @property
    def n(self) -> int:
        return int(next(iter(self.acts.values())).shape[0])

    def dim(self, site: Site) -> int:
        return int(self.acts[str(site)].shape[1])

    def get(self, site: Site) -> torch.Tensor:
        return self.acts[str(site)]

    def subset(self, idx: np.ndarray) -> "ActivationStore":
        arr = np.asarray(idx)
        sel = torch.as_tensor(arr, dtype=torch.long)
        ids = None if self.row_ids is None else np.asarray(self.row_ids)[arr]
        return ActivationStore(self.model_name, list(self.sites),
                               {k: v[sel] for k, v in self.acts.items()}, ids)

    def global_ids(self, local: Sequence[int]) -> List[int]:
        """Map local row positions back to original corpus row ids."""
        ids = np.arange(self.n) if self.row_ids is None else np.asarray(self.row_ids)
        return [int(ids[i]) for i in local]

    def concat_dims(self) -> List[Tuple[Site, int, int]]:
        """Layout of the site-concatenated activation vector."""
        out, off = [], 0
        for s in self.sites:
            d = self.dim(s)
            out.append((s, off, off + d))
            off += d
        return out

    def concatenated(self) -> torch.Tensor:
        return torch.cat([self.acts[str(s)] for s in self.sites], dim=1)


def collect_activations(
    model: HookedModel, batch: PromptBatch, sites: Sequence[Site], chunk: int = 512
) -> ActivationStore:
    """Read activations at ``sites`` over a (possibly large) prompt batch."""
    sites = list(sites)
    parts: Dict[str, List[torch.Tensor]] = {str(s): [] for s in sites}
    n = len(batch)
    for start in range(0, n, chunk):
        sub = batch.subset(range(start, min(start + chunk, n)))
        got = model.read(sub, sites)
        for s in sites:
            parts[str(s)].append(got[str(s)].detach())
    return ActivationStore(model.name, sites, {k: torch.cat(v, dim=0) for k, v in parts.items()})


def split_documents(
    n: int,
    forward_share: float,
    seed: int,
    document_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a probe corpus into forward / reverse halves *by document*.

    Whole documents are assigned to one half or the other, so the two legs share no
    text: "The generic translator corpus is split by document before either model is
    fit."

    ``document_ids`` gives the true document of each row and must be supplied for
    natural corpora, where consecutive prompts really do come from shared documents.
    For the synthetic probe corpus every prompt is an independent draw, so each row
    is its own document and the split degenerates to a disjoint row split -- which is
    the correct semantics there, not an approximation of one. Blocking independent
    rows into pseudo-documents (as an earlier version did) would only coarsen the
    split without making it more faithful.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(n) if document_ids is None else np.asarray(document_ids)
    uniq = np.unique(ids)
    order = rng.permutation(len(uniq))
    n_fwd = int(round(forward_share * len(uniq)))
    fwd_docs = set(uniq[order[:n_fwd]].tolist())
    fwd = np.flatnonzero(np.isin(ids, list(fwd_docs)))
    rev = np.flatnonzero(~np.isin(ids, list(fwd_docs)))
    return np.sort(fwd), np.sort(rev)


def fit_fingerprint(store: ActivationStore, indices: Optional[np.ndarray] = None) -> str:
    """Content hash of the rows a translator leg was fitted on.

    Recorded by every translator so that independence can be *verified* rather than
    asserted: two legs sharing a fitting example will produce overlapping
    fingerprints.
    """
    import hashlib

    X = store.concatenated()
    if indices is not None:
        X = X[torch.as_tensor(np.asarray(indices), dtype=torch.long)]
    q = (X.detach().to(torch.float32) * 1e3).round().to(torch.int64)
    return hashlib.sha1(q.numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- translators
class Translator(ABC):
    """A fitted, directional translation operator."""

    name: str = "translator"
    is_behavior_supervised: bool = False

    def __init__(self, source_model: str = "", dest_model: str = "", seed: int = 0):
        self.source_model = source_model
        self.dest_model = dest_model
        self.seed = int(seed)
        self.fit_info: Dict[str, object] = {}

    # ------------------------------------------------------------------ fit
    @abstractmethod
    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "Translator":
        """Fit on paired activations from the half of the corpus assigned to this leg."""

    # ------------------------------------------------------------ translate
    @abstractmethod
    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        """Propose a destination mechanism, or ``None`` to abstain (the null output)."""

    # ------------------------------------------------------- representational
    def representational_score(self, mech: Mechanism, candidate: Optional[Mechanism]) -> float:
        """The translator's own acceptance signal, used by the 'representational' rule.

        This is the quantity a reconstruction- or similarity-based criterion would
        rely on, with no intervention performed.
        """
        if candidate is None:
            return 0.0
        return float(candidate.meta.get("representational_score", 0.0))

    def coupled_reverse(self) -> Optional["Translator"]:
        """The coupled return map (transpose / same-atom), kept as an ablation."""
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "source_model": self.source_model,
            "dest_model": self.dest_model,
            "seed": self.seed,
            "behavior_supervised": self.is_behavior_supervised,
            "fit_info": {k: v for k, v in self.fit_info.items() if isinstance(v, (int, float, str, bool, list))},
        }


# Attributes through which one translator can be built on top of another. A coupled
# leg may wrap the forward map indirectly -- the crosscoder's coupled reverse reaches it
# via `same_atom`/`shared_dict`, not via `forward` -- so provenance has to be followed
# transitively. Missing a hop here is not a cosmetic bug: the same traversal certifies
# the *independent* leg, so a hidden wrapper could report a coupled leg as independent.
_DELEGATE_ATTRS = ("forward", "same_atom", "shared_dict", "_coupled_from", "base", "inner")


def _delegates(t: "Translator") -> List["Translator"]:
    """Every translator ``t`` is built on top of, transitively (excluding ``t``)."""
    seen = {id(t)}
    stack = [t]
    out: List["Translator"] = []
    while stack:
        cur = stack.pop()
        for attr in _DELEGATE_ATTRS:
            d = getattr(cur, attr, None)
            if d is None or id(d) in seen:
                continue
            if not hasattr(d, "translate"):      # only follow translator-like objects
                continue
            seen.add(id(d))
            out.append(d)
            stack.append(d)
    return out


def _is_derived_from(a: "Translator", b: "Translator") -> bool:
    """Is one translator's map derived from the other's fit?

    The coupled Procrustes reverse is the transpose of the forward map, which is a new
    tensor rather than a shared object, so object identity alone would wrongly report
    the coupled leg as independent. The crosscoder's coupled reverse instead wraps the
    forward translator behind one or two attributes, so the search is transitive.
    """
    return (any(id(d) == id(b) for d in _delegates(a))
            or any(id(d) == id(a) for d in _delegates(b)))


def _shares_parameters(a: "Translator", b: "Translator") -> bool:
    """Do two translators reference any identical parameter tensor?"""
    def tensors(t):
        out = set()
        for obj in [t] + _delegates(t):
            mod = getattr(obj, "module", None)
            if mod is not None:
                out |= {id(p) for p in mod.parameters()}
            for m in getattr(obj, "maps", {}).values():
                w = getattr(m, "W", None)
                if w is not None:
                    out.add(id(w))
        return out
    return bool(tensors(a) & tensors(b))


@dataclass
class TranslatorPair:
    """A forward translator plus *independently fitted* and coupled reverse maps.

    ``reverse_independent`` is fitted on the reverse half of the corpus with its own
    initialisation. ``reverse_coupled`` reproduces the original construction, whose
    round trip can be automatic.
    """

    forward: Translator
    reverse_independent: Translator
    reverse_coupled: Optional[Translator] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.forward.name

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        return self.forward.translate(mech)

    def back(self, mech: Optional[Mechanism], coupled: bool = False) -> Optional[Mechanism]:
        if mech is None:
            return None
        tr = self.reverse_coupled if coupled else self.reverse_independent
        if tr is None:
            return None
        return tr.translate(mech)

    def independence_report(self) -> Dict[str, object]:
        """Verified evidence that the two legs are separate estimates.

        Checks three things rather than assuming them: the legs hold distinct
        parameter objects, their recorded fitting rows do not intersect, and their
        fitting-data fingerprints differ.
        """
        f = self.forward.fit_info
        r = self.reverse_independent.fit_info
        fwd_rows = set(f.get("fit_rows", []) or [])
        rev_rows = set(r.get("fit_rows", []) or [])
        shared_params = (_shares_parameters(self.forward, self.reverse_independent)
                         or _is_derived_from(self.forward, self.reverse_independent))
        return {
            "shared_parameters": shared_params,
            "n_shared_fit_rows": len(fwd_rows & rev_rows),
            "fit_rows_recorded": bool(fwd_rows) and bool(rev_rows),
            "forward_fingerprint": f.get("fit_fingerprint"),
            "reverse_fingerprint": r.get("fit_fingerprint"),
            "fingerprints_differ": f.get("fit_fingerprint") != r.get("fit_fingerprint"),
            "forward_n": f.get("n_fit"),
            "reverse_n": r.get("n_fit"),
            "forward_seed": self.forward.seed,
            "reverse_seed": self.reverse_independent.seed,
            "independent": (not shared_params) and len(fwd_rows & rev_rows) == 0,
        }

    def coupling_report(self) -> Dict[str, object]:
        """The coupled leg should, by contrast, show shared parameters."""
        if self.reverse_coupled is None:
            return {"available": False}
        return {
            "available": True,
            "shared_parameters": _shares_parameters(self.forward, self.reverse_coupled),
            "derived_from_forward": _is_derived_from(self.forward, self.reverse_coupled),
            "coupled": bool(self.reverse_coupled.fit_info.get("coupled")),
            "n_fit": self.reverse_coupled.fit_info.get("n_fit"),
        }
