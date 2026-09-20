"""Behaviour-supervised alignment baselines (Sec. 4.5, Grant 2025).

These are the causal comparison points for CCC+: they *learn* a cross-model
alignment by transferring activity and measuring the resulting behaviour, rather
than proposing a correspondence from representational structure alone.

  MAS            learn an orthonormal destination subspace plus an r x r map, so
                 that importing the source model's in-subspace coordinates
                 produces the interchange behaviour in the destination.
  Latent Stitch  same objective, but the destination map is unconstrained and
                 low-rank (no orthonormality / invertibility requirement).
  Stitch         learn a full linear map between site activations by replacing the
                 destination activation wholesale and matching clean behaviour;
                 the candidate subspace is the image of the source subspace.

Because they consume task labels, the manifest reports them separately from the
unsupervised translators. They are fitted on the hyperparameter split only, never
on calibration or test prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import stable_seed
from ..mechanisms import Mechanism, Site, orthonormalize
from ..models.base import HookedModel, Patch, PromptBatch, resolve_positions
from ..tasks.base import Task, TaskData
from .base import ActivationStore, Translator


@dataclass
class CausalFitContext:
    """Everything a behaviour-supervised translator needs to optimise its map."""

    source_model: HookedModel
    dest_model: HookedModel
    task: Task
    source_data: TaskData          # prompts for the source model (fitting split)
    dest_data: TaskData            # aligned prompts for the destination model
    dest_sites: Sequence[Site]
    interchange_target: torch.Tensor   # [B] label after the variable is imported
    clean_target: Optional[torch.Tensor] = None


class _BehaviorSupervised(Translator):
    is_behavior_supervised = True

    def __init__(self, source_model: str = "", dest_model: str = "", seed: int = 0,
                 steps: int = 200, lr: float = 1e-2, anchor_site: Optional[Site] = None):
        super().__init__(source_model, dest_model, seed)
        self.steps, self.lr = steps, lr
        self.anchor_site = anchor_site
        self.ctx: Optional[CausalFitContext] = None
        self._cache: Dict[str, Optional[Mechanism]] = {}

    def fit(self, source: ActivationStore, dest: ActivationStore, **kwargs) -> "_BehaviorSupervised":
        ctx = kwargs.get("context")
        if ctx is None:
            raise ValueError(f"{self.name} requires a CausalFitContext via fit(context=...)")
        self.ctx = ctx
        self.fit_info = {"n_fit": int(len(ctx.source_data)), "steps": self.steps, "supervised": True}
        return self

    # ------------------------------------------------------------- utilities
    def _source_coords(self, mech: Mechanism) -> torch.Tensor:
        """In-subspace coordinates of the source model on the counterfactual prompts."""
        assert self.ctx is not None
        h = self.ctx.source_model.read_site(self.ctx.source_data.counterfactual, mech.site)
        return (h @ mech.V).detach()

    def _dest_sites(self) -> List[Site]:
        assert self.ctx is not None
        if self.anchor_site is not None:
            return [self.anchor_site]
        return list(self.ctx.dest_sites)

    def translate(self, mech: Mechanism) -> Optional[Mechanism]:
        key = mech.key()
        if key not in self._cache:
            self._cache[key] = self._optimise(mech)
        return self._cache[key]

    def _optimise(self, mech: Mechanism) -> Optional[Mechanism]:  # pragma: no cover - overridden
        raise NotImplementedError


class MASTranslator(_BehaviorSupervised):
    """Model Alignment Search: orthonormal destination subspace + invertible map."""

    name = "mas"

    def _optimise(self, mech: Mechanism) -> Optional[Mechanism]:
        ctx = self.ctx
        assert ctx is not None
        coords = self._source_coords(mech)                       # [B, r]
        r = mech.rank
        target = ctx.interchange_target
        best: Optional[Tuple[float, Site, torch.Tensor]] = None
        for site in self._dest_sites():
            d = ctx.dest_model.site_dim(site)
            g = torch.Generator().manual_seed(stable_seed(self.seed, "mas", str(site)) % (2**31))
            P = torch.nn.Parameter(torch.randn(d, r, generator=g) * (1.0 / d ** 0.5))
            R = torch.nn.Parameter(torch.eye(r) + 0.01 * torch.randn(r, r, generator=g))
            opt = torch.optim.Adam([P, R], lr=self.lr)
            pos = resolve_positions(site.token, ctx.dest_data.clean)
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)
                V = _qr(P)
                mapped = coords @ R.T                             # [B, r]

                def fn(cur: torch.Tensor, V=V, mapped=mapped) -> torch.Tensor:
                    return cur + (mapped - cur @ V) @ V.T

                logits = ctx.dest_model.logits_grad(ctx.dest_data.clean, [Patch(site, fn, pos)])
                loss = F.cross_entropy(logits, target)
                loss.backward()
                opt.step()
            with torch.no_grad():
                V = _qr(P)
                mapped = coords @ R.T

                def fn(cur: torch.Tensor, V=V, mapped=mapped) -> torch.Tensor:
                    return cur + (mapped - cur @ V) @ V.T

                logits = ctx.dest_model.logits(ctx.dest_data.clean, [Patch(site, fn, pos)])
                final = float(F.cross_entropy(logits, target))
            if best is None or final < best[0]:
                best = (final, site, V.detach())
        if best is None:
            return None
        loss, site, V = best
        return Mechanism(
            site=site, V=V, model=self.dest_model, label=f"mas({mech.label})",
            meta={"translator": self.name, "fit_loss": loss, "source_site": str(mech.site),
                  "representational_score": float(np.exp(-loss))},
        )


class LatentStitchTranslator(_BehaviorSupervised):
    """Low-rank stitch: unconstrained destination map, interchange objective."""

    name = "latent_stitch"

    def _optimise(self, mech: Mechanism) -> Optional[Mechanism]:
        ctx = self.ctx
        assert ctx is not None
        coords = self._source_coords(mech)
        r = mech.rank
        target = ctx.interchange_target
        best: Optional[Tuple[float, Site, torch.Tensor]] = None
        for site in self._dest_sites():
            d = ctx.dest_model.site_dim(site)
            g = torch.Generator().manual_seed(stable_seed(self.seed, "latent_stitch", str(site)) % (2**31))
            M = torch.nn.Parameter(torch.randn(d, r, generator=g) * (1.0 / d ** 0.5))
            opt = torch.optim.Adam([M], lr=self.lr)
            pos = resolve_positions(site.token, ctx.dest_data.clean)
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)

                def fn(cur: torch.Tensor, M=M) -> torch.Tensor:
                    Vn = M / M.norm(dim=0, keepdim=True).clamp_min(1e-9)
                    return cur + (coords - cur @ Vn) @ Vn.T

                logits = ctx.dest_model.logits_grad(ctx.dest_data.clean, [Patch(site, fn, pos)])
                loss = F.cross_entropy(logits, target)
                loss.backward()
                opt.step()
            with torch.no_grad():
                Vn = M / M.norm(dim=0, keepdim=True).clamp_min(1e-9)

                def fn(cur: torch.Tensor, Vn=Vn) -> torch.Tensor:
                    return cur + (coords - cur @ Vn) @ Vn.T

                logits = ctx.dest_model.logits(ctx.dest_data.clean, [Patch(site, fn, pos)])
                final = float(F.cross_entropy(logits, target))
            if best is None or final < best[0]:
                best = (final, site, M.detach().clone())
        if best is None:
            return None
        loss, site, M = best
        V = orthonormalize(M)
        if V.shape[1] < mech.rank:
            return None
        return Mechanism(
            site=site, V=V, model=self.dest_model, label=f"latent_stitch({mech.label})",
            meta={"translator": self.name, "fit_loss": loss, "source_site": str(mech.site),
                  "representational_score": float(np.exp(-loss))},
        )


class StitchTranslator(_BehaviorSupervised):
    """Full-activation stitching: learn W so that W h_A replaces h_B and behaviour holds."""

    name = "stitch"

    def _optimise(self, mech: Mechanism) -> Optional[Mechanism]:
        ctx = self.ctx
        assert ctx is not None
        h_src = ctx.source_model.read_site(ctx.source_data.clean, mech.site).detach()
        target = ctx.clean_target if ctx.clean_target is not None else ctx.interchange_target
        best: Optional[Tuple[float, Site, torch.Tensor]] = None
        for site in self._dest_sites():
            d = ctx.dest_model.site_dim(site)
            g = torch.Generator().manual_seed(stable_seed(self.seed, "stitch", str(site)) % (2**31))
            W = torch.nn.Parameter(torch.randn(mech.d, d, generator=g) * (1.0 / mech.d ** 0.5))
            opt = torch.optim.Adam([W], lr=self.lr)
            pos = resolve_positions(site.token, ctx.dest_data.clean)
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)

                def fn(cur: torch.Tensor, W=W) -> torch.Tensor:
                    return h_src @ W                      # wholesale replacement

                logits = ctx.dest_model.logits_grad(ctx.dest_data.clean, [Patch(site, fn, pos)])
                loss = F.cross_entropy(logits, target)
                loss.backward()
                opt.step()
            with torch.no_grad():
                def fn(cur: torch.Tensor, W=W) -> torch.Tensor:
                    return h_src @ W

                logits = ctx.dest_model.logits(ctx.dest_data.clean, [Patch(site, fn, pos)])
                final = float(F.cross_entropy(logits, target))
            if best is None or final < best[0]:
                best = (final, site, W.detach().clone())
        if best is None:
            return None
        loss, site, W = best
        V = orthonormalize(W.T @ mech.V)
        if V.shape[1] < mech.rank:
            return None
        return Mechanism(
            site=site, V=V, model=self.dest_model, label=f"stitch({mech.label})",
            meta={"translator": self.name, "fit_loss": loss, "source_site": str(mech.site),
                  "representational_score": float(np.exp(-loss))},
        )


def _qr(P: torch.Tensor) -> torch.Tensor:
    Q, R = torch.linalg.qr(P)
    return Q * torch.sign(torch.diagonal(R)).clamp(min=-1, max=1)
