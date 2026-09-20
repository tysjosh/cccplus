"""SIIT training: plant a labelled causal graph inside a small transformer.

Three objectives, following the interchange-intervention training family that
InterpBench's semi-synthetic transformers are built with:

  behaviour  the model reproduces the program's output;
  IIT        importing a planted block from a counterfactual prompt produces the
             *interchange* output, so the block really carries its variable;
  strict     importing the orthogonal complement of every planted block at a site
             leaves the output unchanged, so that site carries no task-relevant
             information outside its planted blocks.

The strict term is what makes the ground truth exact: after training, the planted
subspaces are the only carriers of the labelled variables at their sites, so a
translated candidate either matches a planted subspace or it does not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..mechanisms import Site
from ..models.base import Patch, PromptBatch
from ..models.tiny import TinyConfig, TinyTransformer
from .planted import Placement, PlantedBlock, PlantedGroup, build_placement
from .programs import Program, program_output


# --------------------------------------------------------------------- batches
def make_batch(program: Program, tokens: np.ndarray) -> PromptBatch:
    t = torch.as_tensor(tokens, dtype=torch.long)
    pos = torch.full((t.shape[0],), program.readout_position, dtype=torch.long)
    return PromptBatch(tokens=t, attn_mask=None, positions={"final": pos}, readout="final")


def _blocks_by_site(blocks: Sequence[PlantedBlock]) -> Dict[str, Tuple[Site, torch.Tensor]]:
    """Concatenate the subspaces of blocks that share a site (they are orthogonal)."""
    out: Dict[str, Tuple[Site, torch.Tensor]] = {}
    for b in blocks:
        key = str(b.site)
        if key in out:
            out[key] = (b.site, torch.cat([out[key][1], b.V], dim=1))
        else:
            out[key] = (b.site, b.V)
    return out


def multi_group_patches(
    model: TinyTransformer,
    source_batch: PromptBatch,
    assignments: Sequence[Tuple[torch.Tensor, Sequence[PlantedBlock]]],
    grad: bool = True,
    complement: bool = False,
) -> List[Patch]:
    """Row-wise patches: each row range imports its own blocks from ``source_batch``.

    This lets a single forward pass carry the interchange interventions for *every*
    planted group at once, so all groups receive gradient on every step instead of
    one sampled group per step.
    """
    per_site: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Site]]] = {}
    for rows, blocks in assignments:
        by_site = _blocks_by_site(blocks)
        for key, (site, V) in by_site.items():
            per_site.setdefault(key, []).append((rows, V, site))
    sites = [entries[0][2] for entries in per_site.values()]
    src = model.read_grad(source_batch, sites) if grad else model.read(source_batch, sites)

    patches: List[Patch] = []
    for key, entries in per_site.items():
        site = entries[0][2]
        s_all = src[key]

        def fn(cur: torch.Tensor, entries=entries, s_all=s_all) -> torch.Tensor:
            out = cur
            for rows, V, _ in entries:
                diff = s_all[rows] - cur[rows]
                inside = (diff @ V) @ V.T
                upd = diff - inside if complement else inside
                out = out.index_add(0, rows, upd.to(out.dtype))
            return out

        patches.append(Patch(site=site, fn=fn))
    return patches


def subspace_patches(
    model: TinyTransformer,
    source_batch: PromptBatch,
    blocks: Sequence[PlantedBlock],
    alpha: float = 1.0,
    grad: bool = False,
    complement: bool = False,
) -> List[Patch]:
    """Eq. 6 patches that import ``blocks`` (or their complement) from ``source_batch``.

    ``grad=False`` reads the source activations without building a graph; the
    training signal still flows through the patched forward pass, which is what
    forces the variable into the block.
    """
    by_site = _blocks_by_site(blocks)
    sites = [s for s, _ in by_site.values()]
    src = model.read_grad(source_batch, sites) if grad else model.read(source_batch, sites)
    patches: List[Patch] = []
    for key, (site, V) in by_site.items():
        s = src[key]

        def fn(cur: torch.Tensor, s=s, V=V) -> torch.Tensor:
            diff = s - cur
            inside = (diff @ V) @ V.T
            return cur + alpha * (diff - inside if complement else inside)

        patches.append(Patch(site=site, fn=fn))
    return patches


# --------------------------------------------------------------------- result
@dataclass
class SIITMetrics:
    clean_accuracy: float
    iit_accuracy: Dict[str, float]
    strict_accuracy: Dict[str, float]
    steps: int
    seconds: float
    final_loss: float

    def ok(self, min_clean: float, min_iit: float) -> bool:
        return self.clean_accuracy >= min_clean and min(self.iit_accuracy.values(), default=0.0) >= min_iit

    def to_dict(self) -> Dict[str, object]:
        return {
            "clean_accuracy": self.clean_accuracy,
            "iit_accuracy": self.iit_accuracy,
            "strict_accuracy": self.strict_accuracy,
            "min_iit_accuracy": min(self.iit_accuracy.values(), default=float("nan")),
            "mean_strict_accuracy": float(np.mean(list(self.strict_accuracy.values()))) if self.strict_accuracy else float("nan"),
            "steps": self.steps,
            "seconds": self.seconds,
            "final_loss": self.final_loss,
        }


@dataclass
class SIITModel:
    name: str
    program: str
    model: TinyTransformer
    placement: Placement
    metrics: SIITMetrics
    arch: Dict[str, int] = field(default_factory=dict)

    @property
    def absent_vars(self):
        return self.placement.absent_vars


# -------------------------------------------------------------------- training
def train_siit(
    program: Program,
    arch: Dict[str, int],
    structure: str,
    seed: int,
    steps: int = 3000,
    lr: float = 2e-3,
    batch: int = 192,
    behavior_weight: float = 1.0,
    iit_weight: float = 1.0,
    strict_weight: float = 0.3,
    rank: int = 4,
    name: str = "M",
    log_every: int = 0,
    warmup_frac: float = 0.0,
    ramp_frac: float = 0.0,
) -> SIITModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = TinyConfig(
        d_vocab=program.d_vocab,
        n_ctx=program.seq_len,
        d_vocab_out=program.n_out,
        **{k: int(v) for k, v in arch.items()},
    )
    model = TinyTransformer(cfg, name=name)
    placement = build_placement(
        structure=structure,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        rank=rank,
        seed=seed + 991,
    )
    absent = placement.absent_vars
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    sites = placement.sites()
    t0 = time.time()
    last = 0.0

    n_groups = len(placement.groups)
    iit_per_group = max(16, batch // max(1, n_groups))
    strict_per_site = max(16, batch // max(1, len(sites))) if sites else 0
    # Per-group adaptive weights. Redundant placements in particular need this: if
    # either copy alone suffices, plain averaging lets the optimiser satisfy one
    # copy and neglect the other, which would leave the planted structure only
    # partly realised. Re-weighting towards the worst group forces every planted
    # block to carry its variable.
    group_ema = np.ones(n_groups, dtype=float)
    ema_decay, focus = 0.9, 1.0

    def interchange_scale(step: int) -> float:
        """Curriculum: learn the program first, then localise it.

        With all objectives active from step 0, the harder placements (a split
        variable, a merged pair) can fail to learn the program at all. Holding the
        interchange objectives back until the behaviour is in place, then ramping
        them in, removes that failure mode. ``warmup_frac = 0`` recovers the
        simultaneous schedule.
        """
        if warmup_frac <= 0.0 and ramp_frac <= 0.0:
            return 1.0
        w = warmup_frac * steps
        r = max(ramp_frac * steps, 1.0)
        return float(np.clip((step - w) / r, 0.0, 1.0))

    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        gamma = interchange_scale(step)

        # ---- behaviour
        v = program.sample_values(batch, rng)
        x = program.tokens_for_values(v, rng)
        xb = make_batch(program, x)
        y = torch.as_tensor(program_output(v, absent), dtype=torch.long)
        logits = model.logits_grad(xb)
        loss = behavior_weight * F.cross_entropy(logits, y)

        # ---- IIT on every planted group, in one pass
        clean_parts, cf_parts, tgt_parts, assigns = [], [], [], []
        offset = 0
        for g in placement.groups:
            d = program.interchange_batch(iit_per_group, sorted(g.vars), rng, absent)
            clean_parts.append(d["clean_tokens"])
            cf_parts.append(d["cf_tokens"])
            tgt_parts.append(d["distractor"])
            assigns.append((torch.arange(offset, offset + iit_per_group), g.blocks))
            offset += iit_per_group
        cb = make_batch(program, np.concatenate(clean_parts, axis=0))
        sb = make_batch(program, np.concatenate(cf_parts, axis=0))
        li = model.logits_grad(cb, multi_group_patches(model, sb, assigns, grad=True))
        per_row = F.cross_entropy(
            li, torch.as_tensor(np.concatenate(tgt_parts), dtype=torch.long), reduction="none"
        )
        per_group = per_row.view(n_groups, iit_per_group).mean(dim=1)
        with torch.no_grad():
            group_ema = ema_decay * group_ema + (1 - ema_decay) * per_group.detach().numpy()
            w = np.power(np.maximum(group_ema, 1e-6), focus)
            w = w / w.mean()
        loss = loss + gamma * iit_weight * (torch.as_tensor(w, dtype=per_group.dtype) * per_group).mean()

        # ---- strict localisation at every planted site, in one pass
        if strict_weight > 0 and sites and gamma > 0:
            clean_parts, cf_parts, tgt_parts, assigns = [], [], [], []
            offset = 0
            for site in sites:
                var = int(rng.integers(0, program.n_vars))
                d2 = program.interchange_batch(strict_per_site, [var], rng, absent)
                clean_parts.append(d2["clean_tokens"])
                cf_parts.append(d2["cf_tokens"])
                tgt_parts.append(d2["correct"])
                assigns.append((torch.arange(offset, offset + strict_per_site), placement.blocks_at(site)))
                offset += strict_per_site
            cb2 = make_batch(program, np.concatenate(clean_parts, axis=0))
            sb2 = make_batch(program, np.concatenate(cf_parts, axis=0))
            ls = model.logits_grad(cb2, multi_group_patches(model, sb2, assigns, grad=True, complement=True))
            loss = loss + gamma * strict_weight * F.cross_entropy(
                ls, torch.as_tensor(np.concatenate(tgt_parts), dtype=torch.long)
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"    [{name}/{program.name}/{structure}] step {step+1}/{steps} loss {last:.4f}", flush=True)

    model.eval()
    metrics = evaluate_siit(program, model, placement, rng, n=1024)
    metrics.steps = steps
    metrics.seconds = time.time() - t0
    metrics.final_loss = last
    return SIITModel(
        name=name, program=program.name, model=model, placement=placement, metrics=metrics, arch=dict(arch)
    )


@torch.no_grad()
def evaluate_siit(
    program: Program,
    model: TinyTransformer,
    placement: Placement,
    rng: np.random.Generator,
    n: int = 1024,
) -> SIITMetrics:
    absent = placement.absent_vars
    v = program.sample_values(n, rng)
    x = program.tokens_for_values(v, rng)
    xb = make_batch(program, x)
    y = torch.as_tensor(program_output(v, absent), dtype=torch.long)
    clean_acc = float((model.logits(xb).argmax(-1) == y).float().mean())

    iit: Dict[str, float] = {}
    for g in placement.groups:
        d = program.interchange_batch(n, sorted(g.vars), rng, absent)
        cb = make_batch(program, d["clean_tokens"])
        sb = make_batch(program, d["cf_tokens"])
        pred = model.logits(cb, subspace_patches(model, sb, g.blocks)).argmax(-1)
        tgt = torch.as_tensor(d["distractor"], dtype=torch.long)
        key = f"{g.tag}:{'+'.join(f'v{i}' for i in sorted(g.vars))}:{'|'.join(str(b.site) for b in g.blocks)}"
        iit[key] = float((pred == tgt).float().mean())

    strict: Dict[str, float] = {}
    for site in placement.sites():
        var = int(rng.integers(0, program.n_vars))
        d = program.interchange_batch(n, [var], rng, absent)
        cb = make_batch(program, d["clean_tokens"])
        sb = make_batch(program, d["cf_tokens"])
        comp = subspace_patches(model, sb, placement.blocks_at(site), complement=True)
        pred = model.logits(cb, comp).argmax(-1)
        tgt = torch.as_tensor(d["correct"], dtype=torch.long)
        strict[str(site)] = float((pred == tgt).float().mean())

    return SIITMetrics(clean_acc, iit, strict, 0, 0.0, 0.0)


# ------------------------------------------------------------------ persistence
def save_siit(path, sm: SIITModel) -> None:
    blob = {
        "name": sm.name,
        "program": sm.program,
        "arch": sm.arch,
        "cfg": sm.model.cfg.to_dict(),
        "state": sm.model.state_dict(),
        "metrics": sm.metrics.to_dict(),
        "placement": {
            "structure": sm.placement.structure,
            "absent_vars": sorted(sm.placement.absent_vars),
            "n_vars": sm.placement.n_vars,
            "site_bases": {k: v for k, v in sm.placement.site_bases.items()},
            "blocks": [
                {
                    "site": b.site.to_dict(),
                    "V": b.V,
                    "vars": sorted(b.vars),
                    "slot": b.slot,
                    "tag": b.tag,
                }
                for b in sm.placement.blocks
            ],
        },
    }
    torch.save(blob, path)


def load_siit(path) -> SIITModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyTransformer(TinyConfig(**blob["cfg"]), name=blob["name"])
    model.load_state_dict(blob["state"])
    model.eval()
    blocks = [
        PlantedBlock(site=Site(**b["site"]), V=b["V"], vars=frozenset(b["vars"]), slot=b["slot"], tag=b["tag"])
        for b in blob["placement"]["blocks"]
    ]
    from .planted import _groups_for  # local import to avoid cycles at module load

    placement = Placement(
        structure=blob["placement"]["structure"],
        blocks=blocks,
        groups=_groups_for(blob["placement"]["structure"], blocks),
        site_bases=blob["placement"]["site_bases"],
        absent_vars=set(blob["placement"]["absent_vars"]),
        n_vars=blob["placement"]["n_vars"],
    )
    m = blob["metrics"]
    metrics = SIITMetrics(
        clean_accuracy=m["clean_accuracy"],
        iit_accuracy=m["iit_accuracy"],
        strict_accuracy=m["strict_accuracy"],
        steps=m["steps"],
        seconds=m["seconds"],
        final_loss=m["final_loss"],
    )
    return SIITModel(blob["name"], blob["program"], model, placement, metrics, blob.get("arch", {}))
