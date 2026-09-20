"""Natural-model stage (Sec. 4.2): pretrained LMs, IOI and factual recall.

Design points that matter on real checkpoints:

Directed matrix. "Every checkpoint serves as source, giving six directed mappings
per stratum." With three models that is 3 sources x 2 destinations, and each source
has exactly the two destinations the multi-model conjunction needs.

One model resident at a time. Activations for translator fitting are cached to disk,
so the corpus pass loads each model once and never needs two sets of weights
co-resident. Causal measurement is always within a single model, so the whole
protocol runs with peak memory of one checkpoint. On a 40-80 GB A100 the paper's
larger stratum fits without this, but the option keeps the stage runnable on smaller
cards and makes the corpus pass restartable.

Shared prompts, per-model tokenisation. Signature coordinates are indexed by prompt,
and each model resolves its own anchor positions, because "cross-architecture
token/site alignment is imperfect". Prompts that are not scoreable in *every* model
(multi-token names for IOI, or a fact whose object never outranks its distractor) are
dropped from the shared set so all models are scored on identical prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..mechanisms import Mechanism, Site, orthonormalize
from ..models.hf_lm import HFCausalLM
from ..tasks.base import TaskData
from ..tasks.corpus import Corpus, load_corpus
from ..tasks.factual import (
    FactualRecallTask,
    factual_task_data,
    subject_difference_direction,
)
from ..tasks.ioi import IOIExample, IOITask, ioi_task_data
from ..translators.base import ActivationStore, split_documents


# --------------------------------------------------------------- activations
def collect_corpus_activations(
    model: HFCausalLM,
    corpus: Corpus,
    sites: Sequence[Site],
    batch_size: int = 16,
    max_positions_per_text: int = 4,
    seed: int = 0,
    max_length: int = 128,
) -> ActivationStore:
    """Activations at ``sites`` over the corpus, several positions per passage.

    Sampling a few positions per passage rather than only the last token gives the
    dictionary methods enough independent rows without needing a huge corpus, while
    keeping every row attributable to a single document for the split.
    """
    rng = np.random.default_rng(seed)
    parts: Dict[str, List[torch.Tensor]] = {str(s): [] for s in sites}
    row_docs: List[int] = []
    for start in range(0, len(corpus), batch_size):
        texts = corpus.texts[start : start + batch_size]
        docs = corpus.document_ids[start : start + batch_size]
        batch = model.encode(texts, max_length=max_length)
        lengths = batch.mask.sum(dim=1)
        # choose positions inside each sequence (avoid position 0)
        picks = []
        for i, L in enumerate(lengths.tolist()):
            hi = max(int(L) - 1, 1)
            k = min(max_positions_per_text, hi)
            chosen = rng.choice(np.arange(1, hi + 1), size=k, replace=False) if hi >= 1 else np.array([0])
            picks.append(np.sort(chosen))
        for j in range(max_positions_per_text):
            sel = [(i, p[j]) for i, p in enumerate(picks) if j < len(p)]
            if not sel:
                continue
            idx = torch.as_tensor([i for i, _ in sel], dtype=torch.long)
            pos = torch.as_tensor([int(p) for _, p in sel], dtype=torch.long)
            sub = batch.subset(idx.tolist())
            sub.positions["final"] = pos
            got = model.read(sub, list(sites))
            for s in sites:
                parts[str(s)].append(got[str(s)])
            row_docs.extend(int(docs[i]) for i, _ in sel)
    store = ActivationStore(model.name, list(sites),
                            {k: torch.cat(v, dim=0) for k, v in parts.items()})
    store_docs = np.asarray(row_docs)
    return store, store_docs


def save_store(path: Path, store: ActivationStore, docs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": store.model_name,
                "sites": [s.to_dict() for s in store.sites],
                "acts": store.acts, "docs": docs}, path)


def load_store(path: Path) -> Tuple[ActivationStore, np.ndarray]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sites = [Site(**d) for d in blob["sites"]]
    return ActivationStore(blob["model"], sites, blob["acts"]), blob["docs"]


# ------------------------------------------------------- source mechanisms
def ioi_source_mechanisms(
    model: HFCausalLM, layers: Optional[Sequence[int]] = None, rank: Optional[int] = None
) -> List[Mechanism]:
    """All final-position attention heads (Sec. 4.3).

    "V equal to the residual-stream column space of the head output projection."
    The mechanism therefore lives at ``attn_out`` and its subspace is the head's
    write subspace, which makes Eq. 6 an intervention on exactly that head.
    """
    out: List[Mechanism] = []
    layer_list = list(layers) if layers is not None else list(range(model.n_layers))
    for layer in layer_list:
        for head in range(model.info.n_heads):
            V = model.head_write_subspace(layer, head, rank=rank)
            out.append(Mechanism(
                site=Site(layer, "attn_out", "final"), V=V, model=model.name,
                label=f"head{layer}.{head}",
                meta={"layer": layer, "head": head, "kind": "ioi_head"},
            ))
    return out


def factual_source_mechanisms(
    model: HFCausalLM, data: TaskData, layers: Optional[Sequence[int]] = None, rank: int = 1
) -> List[Mechanism]:
    """Final-subject-token MLP outputs, V = normalised mean clean-cf difference."""
    out: List[Mechanism] = []
    layer_list = list(layers) if layers is not None else list(range(model.n_layers))
    for layer in layer_list:
        site = Site(layer, "mlp_out", "subject_last")
        V = subject_difference_direction(model, data, site, rank=rank)
        out.append(Mechanism(site=site, V=V, model=model.name,
                             label=f"mlp{layer}", meta={"layer": layer, "kind": "fr_mlp"}))
    return out


# ------------------------------------------------------------------ controls
def permuted_destination_control(
    mech: Mechanism, model: HFCausalLM, seed: int
) -> Mechanism:
    """Natural-model control: a task-active subspace at the same site, wrong identity.

    Sec. 4.2 lists "within-stratum permuted destinations" and "task-active but
    semantically different subspaces" as controls. For a head mechanism this is a
    different head at the same layer; for an MLP direction it is a random rotation
    of the same site's difference direction.
    """
    if mech.meta.get("kind") == "ioi_head":
        n_heads = model.info.n_heads
        other = (int(mech.meta["head"]) + 1 + (seed % max(n_heads - 1, 1))) % n_heads
        if other == int(mech.meta["head"]):
            other = (other + 1) % n_heads
        V = model.head_write_subspace(int(mech.meta["layer"]), other, rank=mech.rank)
        return Mechanism(site=mech.site, V=V, model=model.name,
                         label=f"control_head{mech.meta['layer']}.{other}",
                         meta={"kind": "control_other_head", "head": other})
    g = torch.Generator().manual_seed(int(seed))
    V = torch.randn(mech.d, mech.rank, generator=g)
    return Mechanism(site=mech.site, V=V, model=model.name, label="control_random",
                     meta={"kind": "control_random_subspace"})


# ------------------------------------------------------------------ screening
@dataclass
class SharedPrompts:
    """Prompt indices scoreable in every model, with per-model task data."""

    keep: List[int]
    report: Dict[str, Dict[str, float]] = field(default_factory=dict)


def screen_ioi_across_models(
    models: Dict[str, HFCausalLM],
    examples: Sequence[IOIExample],
    task: IOITask,
    min_accuracy: float,
) -> SharedPrompts:
    """Intersect single-token-name examples and check the clean-accuracy gate."""
    keep: Optional[Set[int]] = None
    report: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        ok = {i for i, e in enumerate(examples)
              if m.is_single_token(e.io) and m.is_single_token(e.s)}
        keep = ok if keep is None else (keep & ok)
    keep_list = sorted(keep or set())
    for name, m in models.items():
        sub = [examples[i] for i in keep_list]
        data = ioi_task_data(m, sub)
        acc = 0.0 if data is None else task.accuracy(m.logits(data.clean), data)
        report[name] = {"clean_accuracy": acc, "n": len(keep_list),
                        "meets_gate": bool(acc >= min_accuracy)}
    return SharedPrompts(keep_list, report)


def screen_factual_across_models(
    models: Dict[str, HFCausalLM],
    facts: Sequence,
    task: FactualRecallTask,
) -> SharedPrompts:
    """Keep only facts whose object outranks the distractor in *every* model."""
    per_model: Dict[str, np.ndarray] = {}
    report: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        data = factual_task_data(m, list(facts))
        s = task.score_full(m, data).detach().cpu().numpy()
        per_model[name] = s > 0
        report[name] = {"clean_accuracy": float((s > 0).mean()), "n": len(facts)}
    mask = np.ones(len(facts), dtype=bool)
    for v in per_model.values():
        mask &= v
    return SharedPrompts([int(i) for i in np.flatnonzero(mask)], report)
