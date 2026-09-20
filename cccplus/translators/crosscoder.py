"""Crosscoders and Dedicated Feature Crosscoders (Sec. 3.3).

Standard crosscoder (Lindsey et al., 2024): a single sparse dictionary jointly
reconstructs activations from both models, giving an operational notion of
correspondence through shared atoms.

  "The atom whose normalized source-side decoder vector has maximum absolute
  cosine with a rank-one source direction is mapped to the corresponding
  normalized destination-side decoder vector. For a rank-r source subspace, we
  select the preregistered number of atoms maximizing captured projection energy
  and orthogonalize their destination decoder vectors."

Dedicated Feature Crosscoder (Jiralerspong and Bricken, 2026): the
parameterisation separates shared atoms from model-specific ones, so features
unique to one architecture need not be forced into shared atoms.

  "We return a non-null translation only when the method classifies the selected
  atom as shared. A model-specific classification or an insufficient number of
  shared atoms is counted as abstention, not forced into the nearest shared
  feature."

One dictionary is trained per (model pair, corpus half) over the concatenation of
all eligible sites, so an atom carries both *where* and *what*: the site with the
largest decoder energy on each side is that atom's site on that side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mechanisms import Mechanism, Site, orthonormalize
from .base import ActivationStore, Translator, fit_fingerprint

Layout = List[Tuple[Site, int, int]]


# ------------------------------------------------------------------ modules
class Crosscoder(nn.Module):
    """Shared sparse dictionary over two models' concatenated site activations."""

    def __init__(self, d_a: int, d_b: int, n_atoms: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.d_a, self.d_b, self.n_atoms = d_a, d_b, n_atoms
        self.W_enc = nn.Parameter(torch.randn(d_a + d_b, n_atoms, generator=g) * (1.0 / (d_a + d_b) ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(n_atoms))
        self.D_a = nn.Parameter(torch.randn(n_atoms, d_a, generator=g) * (1.0 / n_atoms ** 0.5))
        self.D_b = nn.Parameter(torch.randn(n_atoms, d_b, generator=g) * (1.0 / n_atoms ** 0.5))
        self.b_a = nn.Parameter(torch.zeros(d_a))
        self.b_b = nn.Parameter(torch.zeros(d_b))

    def encode(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.cat([xa, xb], dim=1) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return f @ self.D_a + self.b_a, f @ self.D_b + self.b_b

    def loss(self, xa: torch.Tensor, xb: torch.Tensor, l1: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        f = self.encode(xa, xb)
        ha, hb = self.decode(f)
        rec = F.mse_loss(ha, xa) + F.mse_loss(hb, xb)
        norms = self.D_a.norm(dim=1) + self.D_b.norm(dim=1)
        sparse = (f * norms).sum(dim=1).mean()
        total = rec + l1 * sparse
        with torch.no_grad():
            stats = {
                "rec": float(rec),
                "l0": float((f > 0).float().sum(dim=1).mean()),
                "fvu_a": float(((ha - xa) ** 2).sum() / ((xa - xa.mean(0)) ** 2).sum().clamp_min(1e-9)),
                "fvu_b": float(((hb - xb) ** 2).sum() / ((xb - xb.mean(0)) ** 2).sum().clamp_min(1e-9)),
            }
        return total, stats


class DedicatedFeatureCrosscoder(nn.Module):
    """Shared atoms plus per-model dedicated atoms.

    Dedicated atoms read and write a single model, so a mechanism that exists in
    only one architecture can be explained without corrupting a shared atom. The
    higher penalty on dedicated activity makes a shared explanation preferred
    whenever one is available.
    """

    def __init__(self, d_a: int, d_b: int, n_shared: int, n_specific: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.d_a, self.d_b = d_a, d_b
        self.n_shared, self.n_specific = n_shared, n_specific
        self.W_shared = nn.Parameter(torch.randn(d_a + d_b, n_shared, generator=g) * (1.0 / (d_a + d_b) ** 0.5))
        self.b_shared = nn.Parameter(torch.zeros(n_shared))
        self.W_a = nn.Parameter(torch.randn(d_a, n_specific, generator=g) * (1.0 / d_a ** 0.5))
        self.b_a_enc = nn.Parameter(torch.zeros(n_specific))
        self.W_b = nn.Parameter(torch.randn(d_b, n_specific, generator=g) * (1.0 / d_b ** 0.5))
        self.b_b_enc = nn.Parameter(torch.zeros(n_specific))
        self.Ds_a = nn.Parameter(torch.randn(n_shared, d_a, generator=g) * (1.0 / n_shared ** 0.5))
        self.Ds_b = nn.Parameter(torch.randn(n_shared, d_b, generator=g) * (1.0 / n_shared ** 0.5))
        self.Dd_a = nn.Parameter(torch.randn(n_specific, d_a, generator=g) * (1.0 / n_specific ** 0.5))
        self.Dd_b = nn.Parameter(torch.randn(n_specific, d_b, generator=g) * (1.0 / n_specific ** 0.5))
        self.bias_a = nn.Parameter(torch.zeros(d_a))
        self.bias_b = nn.Parameter(torch.zeros(d_b))

    def encode(self, xa: torch.Tensor, xb: torch.Tensor):
        fs = F.relu(torch.cat([xa, xb], dim=1) @ self.W_shared + self.b_shared)
        fa = F.relu(xa @ self.W_a + self.b_a_enc)
        fb = F.relu(xb @ self.W_b + self.b_b_enc)
        return fs, fa, fb

    def loss(self, xa, xb, l1: float, l1_specific: float):
        fs, fa, fb = self.encode(xa, xb)
        ha = fs @ self.Ds_a + fa @ self.Dd_a + self.bias_a
        hb = fs @ self.Ds_b + fb @ self.Dd_b + self.bias_b
        rec = F.mse_loss(ha, xa) + F.mse_loss(hb, xb)
        ns = self.Ds_a.norm(dim=1) + self.Ds_b.norm(dim=1)
        sparse = (fs * ns).sum(dim=1).mean()
        spec = (fa * self.Dd_a.norm(dim=1)).sum(dim=1).mean() + (fb * self.Dd_b.norm(dim=1)).sum(dim=1).mean()
        total = rec + l1 * sparse + l1_specific * spec
        with torch.no_grad():
            stats = {
                "rec": float(rec),
                "l0_shared": float((fs > 0).float().sum(dim=1).mean()),
                "l0_specific": float(((fa > 0).float().sum(dim=1) + (fb > 0).float().sum(dim=1)).mean()),
                "fvu_a": float(((ha - xa) ** 2).sum() / ((xa - xa.mean(0)) ** 2).sum().clamp_min(1e-9)),
                "fvu_b": float(((hb - xb) ** 2).sum() / ((xb - xb.mean(0)) ** 2).sum().clamp_min(1e-9)),
            }
        return total, stats


# ------------------------------------------------------------------ training
def _train(module: nn.Module, Xa: torch.Tensor, Xb: torch.Tensor, loss_kwargs: Dict[str, float],
           steps: int, lr: float, batch: int, seed: int) -> Dict[str, float]:
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed) + 5)
    n = Xa.shape[0]
    stats: Dict[str, float] = {}
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen)
        opt.zero_grad(set_to_none=True)
        total, stats = module.loss(Xa[idx], Xb[idx], **loss_kwargs)
        total.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
    return stats


# ------------------------------------------------------------------ helpers
def _slice_for(layout: Layout, site: Site) -> Optional[Tuple[int, int]]:
    for s, a, b in layout:
        if str(s) == str(site):
            return a, b
    return None


def _site_energies(layout: Layout, vec: torch.Tensor) -> List[Tuple[Site, float]]:
    return [(s, float(vec[a:b].norm())) for s, a, b in layout]


def _best_site(layout: Layout, vec: torch.Tensor) -> Site:
    return max(_site_energies(layout, vec), key=lambda t: (t[1], -t[0].layer))[0]


# ------------------------------------------------------------- crosscoder op
class CrosscoderTranslator(Translator):
    """Atom-matching translator backed by a jointly trained sparse dictionary."""

    name = "crosscoder"

    def __init__(
        self,
        source_model: str = "",
        dest_model: str = "",
        seed: int = 0,
        n_atoms: int = 256,
        l1: float = 3e-3,
        steps: int = 800,
        lr: float = 3e-3,
        batch: int = 256,
        subspace_atoms: int = 4,
        anchor_site: Optional[Site] = None,
    ):
        super().__init__(source_model, dest_model, seed)
        self.n_atoms, self.l1, self.steps, self.lr, self.batch = n_atoms, l1, steps, lr, batch
        self.subspace_atoms = subspace_atoms
        self.anchor_site = anchor_site
        self.module: Optional[Crosscoder] = None
        self.layout_src: Layout = []
        self.layout_dst: Layout = []
        self.mu_src: Optional[torch.Tensor] = None
        self.mu_dst: Optional[torch.Tensor] = None
        self._shared_module = False

    # ------------------------------------------------------------------ fit
    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "CrosscoderTranslator":
        Xa, Xb = source.concatenated(), dest.concatenated()
        self.layout_src, self.layout_dst = source.concat_dims(), dest.concat_dims()
        self.mu_src, self.mu_dst = Xa.mean(0), Xb.mean(0)
        Xa, Xb = Xa - self.mu_src, Xb - self.mu_dst
        self.module = Crosscoder(Xa.shape[1], Xb.shape[1], self.n_atoms, seed=self.seed)
        stats = _train(self.module, Xa, Xb, {"l1": self.l1}, self.steps, self.lr, self.batch, self.seed)
        self.fit_info = {
            "n_fit": int(Xa.shape[0]), "n_atoms": self.n_atoms,
            "fit_rows": sorted(source.global_ids(range(int(Xa.shape[0])))),
            "fit_fingerprint": fit_fingerprint(source), **stats,
        }
        return self

    def attach(self, other: "CrosscoderTranslator", flip: bool) -> "CrosscoderTranslator":
        """Reuse a trained dictionary, optionally with the two sides swapped."""
        self.module = other.module
        self._shared_module = True
        if flip:
            self.layout_src, self.layout_dst = other.layout_dst, other.layout_src
            self.mu_src, self.mu_dst = other.mu_dst, other.mu_src
        else:
            self.layout_src, self.layout_dst = other.layout_src, other.layout_dst
            self.mu_src, self.mu_dst = other.mu_src, other.mu_dst
        self._flipped = flip
        self.fit_info = dict(other.fit_info)
        self.fit_info["attached"] = True
        return self

    @property
    def flipped(self) -> bool:
        return bool(getattr(self, "_flipped", False))

    # ------------------------------------------------------------ dictionary
    def _decoders(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.module is not None
        if self.flipped:
            return self.module.D_b.detach(), self.module.D_a.detach()
        return self.module.D_a.detach(), self.module.D_b.detach()

    def _atom_scores(self, mech: Mechanism) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Absolute cosine and captured-projection energy of every atom at the source site."""
        sl = _slice_for(self.layout_src, mech.site)
        if sl is None:
            return None
        D_src, _ = self._decoders()
        U = D_src[:, sl[0] : sl[1]]                                # [n_atoms, d_s]
        nrm = U.norm(dim=1).clamp_min(1e-9)
        Un = U / nrm[:, None]
        proj = Un @ mech.V                                          # [n_atoms, r]
        energy = (proj ** 2).sum(dim=1)                              # captured projection energy
        cos = proj.abs().max(dim=1).values if mech.rank > 1 else proj.abs().squeeze(1)
        # Atoms with negligible presence at this site cannot be matched.
        energy = torch.where(nrm > 1e-6 * float(nrm.max()), energy, torch.zeros_like(energy))
        return cos, energy

    def atom_budget(self, rank: int) -> int:
        """Atoms allocated to a rank-r source subspace.

        ``subspace_atoms`` is a floor, not a cap: an IOI head mechanism is the whole
        column space of the head's output projection, so its rank is d_head (16-128).
        A fixed four-atom budget could never span it and the translator would abstain
        on every head, which would look like a property of the method rather than of
        the budget.
        """
        return max(int(self.subspace_atoms), int(rank))

    def select_atoms(self, mech: Mechanism) -> Optional[torch.Tensor]:
        scored = self._atom_scores(mech)
        if scored is None:
            return None
        cos, energy = scored
        k = 1 if mech.rank == 1 else min(self.atom_budget(mech.rank), int(self.n_atoms))
        key = cos if mech.rank == 1 else energy
        if float(key.max()) <= 1e-8:
            return None
        # Deterministic tie breaking by feature index.
        order = torch.argsort(key, descending=True, stable=True)
        return order[:k]

    # ------------------------------------------------------------ translate
    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        if self.module is None:
            return None
        atoms = self.select_atoms(mech)
        if atoms is None or len(atoms) == 0:
            return None
        _, D_dst = self._decoders()
        rows = D_dst[atoms]                                          # [k, d_dst_total]
        site = self.anchor_site or _best_site(self.layout_dst, rows.abs().sum(0))
        sl = _slice_for(self.layout_dst, site)
        if sl is None:
            return None
        Vb_raw = rows[:, sl[0] : sl[1]].T                            # [d_dst, k]
        if float(Vb_raw.norm()) < 1e-9:
            return None
        gain = float(Vb_raw.norm()) / max(float(mech.V.norm()), 1e-12)
        Vb = orthonormalize(Vb_raw)
        if Vb.shape[1] == 0:
            return None
        rank_deficit = int(max(0, mech.rank - Vb.shape[1]))
        Vb = Vb[:, : max(mech.rank, 1)]
        cos, _ = self._atom_scores(mech)
        return Mechanism(
            site=site,
            V=Vb,
            model=self.dest_model,
            label=f"{self.name}({mech.label})",
            meta={
                "translator": self.name,
                "atoms": [int(a) for a in atoms],
                "source_site": str(mech.site),
                "max_atom_cosine": float(cos[atoms].max()),
                "representational_score": float(cos[atoms].max()),
                "map_gain": gain,
                "rank_deficit": rank_deficit,
            },
        )

    # --------------------------------------------------------- coupled leg
    def coupled_reverse(self, anchor_site: Optional[Site] = None) -> "SameAtomReverse":
        return SameAtomReverse(self, anchor_site)


class CoupledReverse(Translator):
    """Coupled return leg, strongest form available for the given candidate.

    Sec. 3.3 describes the coupled construction as using "the same crosscoder atom in
    both directions", which makes recovery automatic: the returned subspace is the
    forward atom's source-side vector regardless of the destination candidate's causal
    role. That is only defined when the candidate carries atom provenance, i.e. when it
    was produced by the forward translator. For a candidate supplied from outside (the
    oracle and decoy candidates of the rule-discrimination analysis) there is no atom to
    reuse, so the leg falls back to the same *dictionary* as the forward fit -- still
    coupled, since parameters and fitting data are shared, but not automatic.

    Which form was used is recorded on every returned mechanism, so the two cases are
    never silently pooled.
    """

    name = "crosscoder_coupled"

    def __init__(self, same_atom: "SameAtomReverse", shared_dict: Translator):
        super().__init__(same_atom.source_model, same_atom.dest_model, same_atom.seed)
        self.same_atom = same_atom
        self.shared_dict = shared_dict
        self.fit_info = {"coupled": True, "n_fit": 0,
                         "same_atom_when_available": True}

    @property
    def anchor_site(self):
        return getattr(self.same_atom, "anchor_site", None)

    @anchor_site.setter
    def anchor_site(self, v):
        self.same_atom.anchor_site = v
        if hasattr(self.shared_dict, "anchor_site"):
            self.shared_dict.anchor_site = v

    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "CoupledReverse":
        return self

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        if mech.meta.get("atoms"):
            out = self.same_atom.translate(mech)
            if out is not None:
                out.meta["coupling_form"] = "same_atom"
                return out
        out = self.shared_dict.translate(mech)
        if out is not None:
            out.meta["coupled"] = True
            out.meta["coupling_form"] = "shared_dictionary"
        return out


class SameAtomReverse(Translator):
    """Coupled return: reuse the *same atom* selected on the forward leg.

    This is the construction CCC+ flags as making round-trip recovery automatic:
    the returned subspace is the forward atom's source-side vector, so the return
    test carries no information about the destination candidate's causal role.
    """

    name = "crosscoder_coupled"

    def __init__(self, forward: CrosscoderTranslator, anchor_site: Optional[Site] = None):
        super().__init__(forward.dest_model, forward.source_model, forward.seed)
        self.forward = forward
        self.anchor_site = anchor_site
        self.fit_info = {"coupled": True, "n_fit": 0, "same_atom": True}

    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "SameAtomReverse":
        return self

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        atoms = mech.meta.get("atoms")
        if not atoms:
            return None
        D_src, _ = self.forward._decoders()
        rows = D_src[torch.as_tensor(atoms, dtype=torch.long)]
        site = self.anchor_site
        if site is None:
            src_site = mech.meta.get("source_site")
            site = next((s for s, _, _ in self.forward.layout_src if str(s) == str(src_site)), None)
        if site is None:
            site = _best_site(self.forward.layout_src, rows.abs().sum(0))
        sl = _slice_for(self.forward.layout_src, site)
        if sl is None:
            return None
        Va_raw = rows[:, sl[0] : sl[1]].T
        if float(Va_raw.norm()) < 1e-9:
            return None
        gain = float(Va_raw.norm()) / max(float(mech.V.norm()), 1e-12)
        return Mechanism(
            site=site, V=orthonormalize(Va_raw), model=self.dest_model,
            label=f"same_atom({mech.label})",
            meta={"translator": self.name, "atoms": list(atoms), "coupled": True, "map_gain": gain},
        )


# -------------------------------------------------------------------- DFC op
class DFCTranslator(Translator):
    """Dedicated Feature Crosscoder translator with explicit abstention."""

    name = "dfc"

    def __init__(
        self,
        source_model: str = "",
        dest_model: str = "",
        seed: int = 0,
        n_shared: int = 192,
        n_specific: int = 64,
        l1: float = 3e-3,
        l1_specific: float = 6e-3,
        steps: int = 800,
        lr: float = 3e-3,
        batch: int = 256,
        subspace_atoms: int = 4,
        shared_threshold: float = 0.60,
        anchor_site: Optional[Site] = None,
    ):
        super().__init__(source_model, dest_model, seed)
        self.n_shared, self.n_specific = n_shared, n_specific
        self.l1, self.l1_specific = l1, l1_specific
        self.steps, self.lr, self.batch = steps, lr, batch
        self.subspace_atoms = subspace_atoms
        self.shared_threshold = shared_threshold
        self.anchor_site = anchor_site
        self.module: Optional[DedicatedFeatureCrosscoder] = None
        self.layout_src: Layout = []
        self.layout_dst: Layout = []
        self._flipped = False

    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "DFCTranslator":
        Xa, Xb = source.concatenated(), dest.concatenated()
        self.layout_src, self.layout_dst = source.concat_dims(), dest.concat_dims()
        Xa, Xb = Xa - Xa.mean(0), Xb - Xb.mean(0)
        self.module = DedicatedFeatureCrosscoder(Xa.shape[1], Xb.shape[1], self.n_shared, self.n_specific, self.seed)
        stats = _train(
            self.module, Xa, Xb, {"l1": self.l1, "l1_specific": self.l1_specific},
            self.steps, self.lr, self.batch, self.seed,
        )
        self.fit_info = {
            "n_fit": int(Xa.shape[0]), "n_shared": self.n_shared, "n_specific": self.n_specific,
            "fit_rows": sorted(source.global_ids(range(int(Xa.shape[0])))),
            "fit_fingerprint": fit_fingerprint(source), **stats,
        }
        return self

    def attach(self, other: "DFCTranslator", flip: bool) -> "DFCTranslator":
        self.module = other.module
        self._flipped = flip
        if flip:
            self.layout_src, self.layout_dst = other.layout_dst, other.layout_src
        else:
            self.layout_src, self.layout_dst = other.layout_src, other.layout_dst
        self.fit_info = dict(other.fit_info)
        self.fit_info["attached"] = True
        return self

    @property
    def flipped(self) -> bool:
        return self._flipped

    def _parts(self):
        assert self.module is not None
        m = self.module
        if self.flipped:
            return m.Ds_b.detach(), m.Ds_a.detach(), m.Dd_b.detach()
        return m.Ds_a.detach(), m.Ds_b.detach(), m.Dd_a.detach()

    def shared_scores(self) -> torch.Tensor:
        """Per-atom 'sharedness': balance of decoder energy across the two models."""
        Ds_src, Ds_dst, _ = self._parts()
        a = Ds_src.norm(dim=1)
        b = Ds_dst.norm(dim=1)
        return (torch.minimum(a, b) / torch.maximum(a, b).clamp_min(1e-9))

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        """Return a candidate only if the best-matching atom is classified shared."""
        if self.module is None:
            return None
        sl = _slice_for(self.layout_src, mech.site)
        if sl is None:
            return None
        Ds_src, Ds_dst, Dd_src = self._parts()
        # Compare the source direction against shared *and* dedicated atoms.
        def cosines(D: torch.Tensor) -> torch.Tensor:
            U = D[:, sl[0] : sl[1]]
            n = U.norm(dim=1).clamp_min(1e-9)
            proj = (U / n[:, None]) @ mech.V
            return (proj ** 2).sum(dim=1), proj.abs().max(dim=1).values

        shared_energy, shared_cos = cosines(Ds_src)
        ded_energy, ded_cos = cosines(Dd_src)
        if float(shared_energy.max()) <= 1e-8 and float(ded_energy.max()) <= 1e-8:
            return None
        # Abstain when a dedicated (model-specific) atom explains the direction best.
        if float(ded_energy.max()) > float(shared_energy.max()):
            self.fit_info["last_abstention"] = "model_specific"
            return None
        sharedness = self.shared_scores()
        k = 1 if mech.rank == 1 else min(max(int(self.subspace_atoms), int(mech.rank)), self.n_shared)
        order = torch.argsort(shared_energy if mech.rank > 1 else shared_cos, descending=True, stable=True)
        eligible = [int(i) for i in order if float(sharedness[int(i)]) >= self.shared_threshold][:k]
        if len(eligible) < k:
            self.fit_info["last_abstention"] = "insufficient_shared_atoms"
            return None
        idx = torch.as_tensor(eligible, dtype=torch.long)
        rows = Ds_dst[idx]
        site = self.anchor_site or _best_site(self.layout_dst, rows.abs().sum(0))
        sl_d = _slice_for(self.layout_dst, site)
        if sl_d is None:
            return None
        Vb_raw = rows[:, sl_d[0] : sl_d[1]].T
        if float(Vb_raw.norm()) < 1e-9:
            return None
        gain = float(Vb_raw.norm()) / max(float(mech.V.norm()), 1e-12)
        Vb = orthonormalize(Vb_raw)
        if Vb.shape[1] == 0:
            return None
        return Mechanism(
            site=site, V=Vb[:, : mech.rank], model=self.dest_model, label=f"dfc({mech.label})",
            meta={
                "translator": self.name,
                "atoms": eligible,
                "source_site": str(mech.site),
                "sharedness": float(sharedness[idx].min()),
                "max_atom_cosine": float(shared_cos[idx].max()),
                "representational_score": float(shared_cos[idx].max()),
                "map_gain": gain,
            },
        )

    def coupled_reverse(self, anchor_site: Optional[Site] = None) -> "DFCSameAtomReverse":
        return DFCSameAtomReverse(self, anchor_site)


class DFCSameAtomReverse(Translator):
    name = "dfc_coupled"

    def __init__(self, forward: DFCTranslator, anchor_site: Optional[Site] = None):
        super().__init__(forward.dest_model, forward.source_model, forward.seed)
        self.forward = forward
        self.anchor_site = anchor_site
        self.fit_info = {"coupled": True, "n_fit": 0, "same_atom": True}

    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "DFCSameAtomReverse":
        return self

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        atoms = mech.meta.get("atoms")
        if not atoms:
            return None
        Ds_src, _, _ = self.forward._parts()
        rows = Ds_src[torch.as_tensor(atoms, dtype=torch.long)]
        site = self.anchor_site
        if site is None:
            src_site = mech.meta.get("source_site")
            site = next((s for s, _, _ in self.forward.layout_src if str(s) == str(src_site)), None)
        if site is None:
            site = _best_site(self.forward.layout_src, rows.abs().sum(0))
        sl = _slice_for(self.forward.layout_src, site)
        if sl is None:
            return None
        Va_raw = rows[:, sl[0] : sl[1]].T
        if float(Va_raw.norm()) < 1e-9:
            return None
        gain = float(Va_raw.norm()) / max(float(mech.V.norm()), 1e-12)
        return Mechanism(
            site=site, V=orthonormalize(Va_raw), model=self.dest_model,
            label=f"dfc_same_atom({mech.label})",
            meta={"translator": self.name, "coupled": True, "map_gain": gain},
        )
