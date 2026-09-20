"""Interventions and causal-effect estimation (Sec. 3.1-3.2).

Primary intervention, Eq. 6 (directional counterfactual patch):

    h_patch_s(x) = h_s(x) + a * V V^T ( h_s(x') - h_s(x) )

with the preregistered setting family A = {full (a=1), half (a=1/2)}. Only the
component inside the tested subspace is replaced, and the replacement comes from
a naturally occurring activation difference rather than an architecture-dependent
steering magnitude.

Robustness arms (reported separately, never pooled into acceptance):
  * mean ablation -- replace the in-subspace component with its dataset mean;
  * norm-matched additive steering -- add a fixed in-subspace direction scaled to
    the norm of the counterfactual delta.

Each intervention also reports unrelated-logit change, cross-task effect and
output-distribution KL, "so that a target effect can be distinguished from broad
behavioral disruption."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .mechanisms import Mechanism, Site
from .models.base import HookedModel, Patch, PromptBatch, resolve_positions
from .tasks.base import Task, TaskData

PRIMARY = "counterfactual_patch"


def _additive_patch(add: torch.Tensor):
    """An additive patch that survives a chunked forward pass.

    ``add`` holds one row per prompt, so when the model splits the batch it must be
    sliced to the rows being processed. The ``row_aware`` marker is what tells the model
    it is safe to chunk at all: a patch without it is run in a single pass, because
    silently pairing the wrong rows would corrupt every effect it produces.
    """
    def fn(cur: torch.Tensor, rows: Optional[torch.Tensor] = None, add=add) -> torch.Tensor:
        a = add if rows is None else add.index_select(0, rows.to(add.device))
        return cur + a.to(cur.dtype)

    fn.row_aware = True
    return fn
ROBUSTNESS = ("mean_ablation", "norm_matched_steering")


@dataclass(frozen=True)
class Setting:
    """One element of the preregistered setting family A."""

    kind: str = PRIMARY
    alpha: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.kind}@{self.alpha:g}"

    def __str__(self) -> str:
        return self.key


def settings_from_manifest(manifest, kinds: Sequence[str] = (PRIMARY,)) -> List[Setting]:
    names = manifest.path_of("signature.settings")
    alphas = manifest.path_of("signature.setting_alphas")
    return [Setting(kind, float(alphas[n])) for kind in kinds for n in names]


class CausalProbe:
    """Caches everything needed to evaluate many mechanisms in one model/task.

    One clean forward pass and one counterfactual forward pass are shared by all
    mechanisms; each (mechanism, setting) then costs a single patched forward.
    """

    def __init__(
        self,
        model: HookedModel,
        task: Task,
        data: TaskData,
        sites: Sequence[Site] = (),
        collect_kl: bool = True,
    ):
        self.model = model
        self.task = task
        self.data = data
        self.collect_kl = collect_kl
        self._acts: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._full = bool(getattr(task, "uses_full_scoring", False))
        clean_logits = model.logits(data.clean)
        self._clean_logits = clean_logits
        self._clean_score = (
            task.score_full(model, data) if self._full else task.score(clean_logits, data)
        )
        self._clean_logprobs = torch.log_softmax(clean_logits.to(torch.float32), dim=-1) if collect_kl else None
        self._clean_unrelated = task.unrelated_score(clean_logits, data)
        if sites:
            self.prefetch(sites)

    # --------------------------------------------------------------- caches
    @property
    def clean_score(self) -> torch.Tensor:
        return self._clean_score

    @property
    def clean_accuracy(self) -> float:
        return float((self._clean_score > 0).float().mean())

    def prefetch(self, sites: Sequence[Site]) -> None:
        todo = [s for s in sites if str(s) not in self._acts]
        if not todo:
            return
        clean = self.model.read(self.data.clean, todo)
        cf = self.model.read(self.data.counterfactual, todo)
        for s in todo:
            self._acts[str(s)] = (clean[str(s)], cf[str(s)])

    def acts(self, site: Site) -> Tuple[torch.Tensor, torch.Tensor]:
        if str(site) not in self._acts:
            self.prefetch([site])
        return self._acts[str(site)]

    def delta_acts(self, site: Site) -> torch.Tensor:
        clean, cf = self.acts(site)
        return cf - clean

    # ---------------------------------------------------------- intervention
    def _patch_fn(self, mech: Mechanism, setting: Setting):
        """Build the additive patch for one intervention arm.

        Every arm reduces to ``cur + add`` with a per-prompt ``add``. Note that the
        mean-ablation target and the steering direction are batch statistics: they are
        computed over the whole prompt set *here*, before the forward pass is split into
        chunks, so chunking cannot change them.
        """
        clean, cf = self.acts(mech.site)
        V = mech.V.to(clean.dtype)
        if setting.kind == PRIMARY:
            add = setting.alpha * ((cf - clean) @ V @ V.T)
        elif setting.kind == "mean_ablation":
            mean = clean.mean(dim=0, keepdim=True)
            add = setting.alpha * ((mean - clean) @ V @ V.T)
        elif setting.kind == "norm_matched_steering":
            delta = (cf - clean) @ V @ V.T
            mean_dir = delta.mean(dim=0, keepdim=True)
            unit = mean_dir / mean_dir.norm().clamp_min(1e-12)
            scale = delta.norm(dim=1, keepdim=True)           # match per-prompt patch norm
            add = setting.alpha * scale * unit
        else:
            raise ValueError(f"unknown intervention kind: {setting.kind}")
        return _additive_patch(add)

    # (see module-level _additive_patch for the row-aware closure)

    def patch(self, mech: Mechanism, setting: Setting) -> Patch:
        pos = resolve_positions(mech.site.token, self.data.clean)
        return Patch(site=mech.site, fn=self._patch_fn(mech, setting), positions=pos)

    # --------------------------------------------------------------- effects
    def effect(self, mech: Mechanism, setting: Setting) -> torch.Tensor:
        """Eq. 1: per-prompt causal effect Delta_T(M, m, a; x), shape [B]."""
        patches = [self.patch(mech, setting)]
        if self._full:
            return self.task.score_full(self.model, self.data, patches) - self._clean_score
        logits = self.model.logits(self.data.clean, patches)
        return self.task.score(logits, self.data) - self._clean_score

    def effects(self, mech: Mechanism, settings: Sequence[Setting]) -> Dict[str, torch.Tensor]:
        return {s.key: self.effect(mech, s) for s in settings}

    def collateral(self, mech: Mechanism, setting: Setting) -> Dict[str, float]:
        """Unrelated-logit change, output KL, and (if present) cross-task effect."""
        logits = self.model.logits(self.data.clean, [self.patch(mech, setting)])
        out: Dict[str, float] = {}
        unrelated = self.task.unrelated_score(logits, self.data) - self._clean_unrelated
        out["unrelated_logit_change"] = float(unrelated.abs().mean())
        if self._clean_logprobs is not None:
            lp = torch.log_softmax(logits.to(torch.float32), dim=-1)
            p = self._clean_logprobs.exp()
            kl = (p * (self._clean_logprobs - lp)).sum(dim=-1)
            out["output_kl"] = float(kl.mean())
        if self.data.cross_task is not None:
            cross = CausalProbe(self.model, self.task, self.data.cross_task, collect_kl=False)
            try:
                out["cross_task_effect"] = float(cross.effect(mech, setting).abs().mean())
            except Exception:
                out["cross_task_effect"] = float("nan")
        return out


def null_direction_mechanisms(
    model: HookedModel,
    sites: Sequence[Site],
    rank: int,
    n: int,
    seed: int,
) -> List[Mechanism]:
    """Random-direction controls used to calibrate tau_min (Sec 3.6).

    These are directions with no planted role; the upper quantile of their absolute
    effects sets the minimum effect size a real mechanism must exceed.
    """
    gen = torch.Generator().manual_seed(int(seed))
    out: List[Mechanism] = []
    for i in range(n):
        site = sites[i % len(sites)]
        d = model.site_dim(site)
        V = torch.randn(d, rank, generator=gen)
        out.append(Mechanism(site=site, V=V, model=model.name, label=f"null{i}"))
    return out
