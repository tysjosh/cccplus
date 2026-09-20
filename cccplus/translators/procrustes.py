"""Cosine alignment by (semi-)orthogonal Procrustes (Sec. 3.3, 'Cosine alignment').

"We collect paired activations from the training probe set, center them within
each model, and fit an orthogonal (or rectangular semi-orthogonal) Procrustes map
for every preregistered pair of eligible sites. A translated rank-one direction is
normalized after mapping; a translated subspace is re-orthogonalized by QR
decomposition. The candidate site is selected on the calibration split by maximum
held-out cosine similarity, with deterministic site and feature-index tie
breaking. Equal layer indices are not required."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..mechanisms import Mechanism, Site, orthonormalize
from .base import ActivationStore, Translator, fit_fingerprint


@dataclass
class _SiteMap:
    source: Site
    dest: Site
    W: torch.Tensor          # [d_src, d_dst], semi-orthogonal
    mu_src: torch.Tensor     # [d_src]
    mu_dst: torch.Tensor     # [d_dst]
    heldout_cosine: float


def _orthogonal_procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """argmin_{W: semi-orthogonal} ||X W - Y||_F, returning W of shape [dX, dY]."""
    M = X.to(torch.float64).T @ Y.to(torch.float64)     # [dX, dY]
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U @ Vh).to(torch.float32)


def _mean_cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    pn = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-12)
    tn = target / target.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((pn * tn).sum(dim=1).mean())


class ProcrustesTranslator(Translator):
    """Deterministic translator: one Procrustes map per eligible site pair."""

    name = "procrustes"

    def __init__(
        self,
        source_model: str = "",
        dest_model: str = "",
        seed: int = 0,
        heldout_share: float = 0.25,
        coupled_from: Optional["ProcrustesTranslator"] = None,
        anchor_site: Optional[Site] = None,
    ):
        super().__init__(source_model, dest_model, seed)
        self.maps: Dict[str, _SiteMap] = {}
        self.heldout_share = heldout_share
        self._coupled_from = coupled_from
        self.anchor_site = anchor_site          # if set, always translate into this site
        self._src_store: Optional[ActivationStore] = None
        self._dst_store: Optional[ActivationStore] = None

    # ------------------------------------------------------------------ fit
    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "ProcrustesTranslator":
        n = source.n
        if dest.n != n:
            raise ValueError("source and destination stores must share the probe corpus")
        rng = np.random.default_rng(self.seed + 13)
        perm = rng.permutation(n)
        n_hold = max(8, int(round(self.heldout_share * n)))
        hold, train = perm[:n_hold], perm[n_hold:]
        self._src_store, self._dst_store = source, dest
        for s_src in source.sites:
            Xs = source.get(s_src)
            mu_s = Xs[train].mean(dim=0)
            for s_dst in dest.sites:
                Xd = dest.get(s_dst)
                mu_d = Xd[train].mean(dim=0)
                W = _orthogonal_procrustes(Xs[train] - mu_s, Xd[train] - mu_d)
                hc = _mean_cosine((Xs[hold] - mu_s) @ W, Xd[hold] - mu_d)
                self.maps[f"{s_src}->{s_dst}"] = _SiteMap(s_src, s_dst, W, mu_s, mu_d, hc)
        self.fit_info = {
            "n_fit": int(len(train)),
            "n_heldout": int(len(hold)),
            "n_site_pairs": len(self.maps),
            "mean_heldout_cosine": float(np.mean([m.heldout_cosine for m in self.maps.values()])),
            # provenance so translator independence can be verified, not asserted
            "fit_rows": sorted(source.global_ids(train.tolist())),
            "fit_fingerprint": fit_fingerprint(source, train),
        }
        return self

    # ------------------------------------------------------------ translate
    def _select_map(self, site: Site) -> Optional[_SiteMap]:
        cands = [m for m in self.maps.values() if str(m.source) == str(site)]
        if not cands:
            return None
        if self.anchor_site is not None:
            anchored = [m for m in cands if str(m.dest) == str(self.anchor_site)]
            if anchored:
                return anchored[0]
        # Deterministic tie breaking: alignment quality, then layer index, then name.
        cands.sort(key=lambda m: (-m.heldout_cosine, m.dest.layer, str(m.dest)))
        return cands[0]

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        m = self._select_map(mech.site)
        if m is None:
            # No map was fitted for this site: the mechanism sits outside the
            # preregistered site grid. Recorded explicitly so this is never confused
            # with a substantive abstention.
            self.fit_info["last_abstention"] = f"no_map_for_site:{mech.site}"
            return None
        # Row-convention map x_B ~= x_A W means directions push forward by W^T.
        Vb_raw = m.W.T @ mech.V
        if float(Vb_raw.norm()) < 1e-8:
            return None
        # norm gain of the map on this subspace, before re-orthonormalisation
        gain = float(Vb_raw.norm()) / max(float(mech.V.norm()), 1e-12)
        Vb = orthonormalize(Vb_raw)
        if Vb.shape[1] < mech.rank:
            return None
        return Mechanism(
            site=m.dest,
            V=Vb,
            model=self.dest_model,
            label=f"procrustes({mech.label})",
            meta={
                "translator": self.name,
                "source_site": str(mech.site),
                "heldout_cosine": m.heldout_cosine,
                "representational_score": m.heldout_cosine,
                "map_gain": gain,
                "coupled": self._coupled_from is not None,
            },
        )

    # --------------------------------------------------------- coupled leg
    def coupled_reverse(self, anchor_site: Optional[Site] = None) -> "ProcrustesTranslator":
        """Reverse map defined as the transpose of the forward map.

        This is the construction CCC+ singles out as making geometric recovery
        automatic: applying it after the forward map returns the source subspace
        exactly, whatever the destination candidate's causal role.
        """
        rev = ProcrustesTranslator(self.dest_model, self.source_model, self.seed, coupled_from=self)
        rev.anchor_site = anchor_site
        for key, m in self.maps.items():
            rev.maps[f"{m.dest}->{m.source}"] = _SiteMap(
                source=m.dest, dest=m.source, W=m.W.T.contiguous(),
                mu_src=m.mu_dst, mu_dst=m.mu_src, heldout_cosine=m.heldout_cosine,
            )
        rev.fit_info = {"coupled": True, "n_fit": 0, "derived_from_forward": True}
        rev.name = "procrustes_coupled"
        return rev

    def map_activations(self, site_src: Site, site_dst: Site, X: torch.Tensor) -> Optional[torch.Tensor]:
        m = self.maps.get(f"{site_src}->{site_dst}")
        if m is None:
            return None
        return (X - m.mu_src) @ m.W + m.mu_dst
