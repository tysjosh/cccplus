"""Multi-site circuit extension (Sec. 3.7).

    C = ({(s_l, V_l)}_{l=1..L}, E)

"A multi-site translator maps nodes and edges, may abstain on either, and is
evaluated with joint and path-specific interventions. Its causal signature
concatenates individual-node effects, joint effects, and interaction contrasts

    Gamma_lk = Delta_lk - Delta_l - Delta_k .                          (Eq. 16)

The single-site method is the special case L = 1 and E = empty, so the multi-site
class is strictly more expressive."

The interaction contrast is what a nodewise translation cannot see. Two circuits
can have matching per-node mean effects while combining them differently; Gamma
is the coordinate that separates them, which is why the paper concatenates it into
the signature rather than reporting it separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .interventions import CausalProbe, Setting
from .mechanisms import Mechanism, Site, subspace_similarity
from .models.base import Patch, resolve_positions
from .signature import CausalSignature, compare_signatures, prompt_key_of


@dataclass
class CircuitHypothesis:
    """A set of sites with hypothesised directed dependencies between them."""

    nodes: List[Mechanism]
    edges: List[Tuple[int, int]] = field(default_factory=list)
    model: str = ""
    label: str = ""

    @property
    def L(self) -> int:
        return len(self.nodes)

    def key(self) -> str:
        return f"{self.model}|" + "+".join(n.key() for n in self.nodes) + f"|E{self.edges}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model, "label": self.label,
            "nodes": [n.to_dict() for n in self.nodes], "edges": [list(e) for e in self.edges],
        }


# ------------------------------------------------------------------ patching
def _multi_patch(probe: CausalProbe, mechs: Sequence[Mechanism], setting: Setting) -> List[Patch]:
    """Eq. 6 applied at several sites at once, merging subspaces that share a site."""
    by_site: Dict[str, Tuple[Site, torch.Tensor]] = {}
    for m in mechs:
        key = str(m.site)
        if key in by_site:
            by_site[key] = (m.site, torch.cat([by_site[key][1], m.V], dim=1))
        else:
            by_site[key] = (m.site, m.V)
    patches: List[Patch] = []
    for key, (site, V) in by_site.items():
        clean, cf = probe.acts(site)
        delta = ((cf - clean) @ V) @ V.T
        add = setting.alpha * delta
        pos = resolve_positions(site.token, probe.data.clean)

        def fn(cur: torch.Tensor, add=add) -> torch.Tensor:
            return cur + add.to(cur.dtype)

        patches.append(Patch(site=site, fn=fn, positions=pos))
    return patches


def joint_effect(probe: CausalProbe, mechs: Sequence[Mechanism], setting: Setting) -> torch.Tensor:
    """Per-prompt effect of intervening on several mechanisms simultaneously."""
    if not mechs:
        return torch.zeros_like(probe.clean_score)
    logits = probe.model.logits(probe.data.clean, _multi_patch(probe, mechs, setting))
    return probe.task.score(logits, probe.data) - probe.clean_score


def interaction_contrast(
    probe: CausalProbe, a: Mechanism, b: Mechanism, setting: Setting
) -> torch.Tensor:
    """Eq. 16: Gamma = Delta_ab - Delta_a - Delta_b, per prompt."""
    d_ab = joint_effect(probe, [a, b], setting)
    d_a = joint_effect(probe, [a], setting)
    d_b = joint_effect(probe, [b], setting)
    return d_ab - d_a - d_b


# ----------------------------------------------------------------- signature
def circuit_signature(
    probe: CausalProbe,
    circuit: CircuitHypothesis,
    settings: Sequence[Setting],
    scale: float = 1.0,
    include_interactions: bool = True,
) -> CausalSignature:
    """Concatenated node, joint and interaction coordinates.

    Coordinate blocks, in a fixed order so that two circuits are always compared
    coordinate-for-coordinate: each node alone, then the full joint intervention,
    then one interaction contrast per hypothesised edge.
    """
    rows: List[torch.Tensor] = []
    keys: List[str] = []
    for s in settings:
        for i, node in enumerate(circuit.nodes):
            rows.append(joint_effect(probe, [node], s).to(torch.float64))
            keys.append(f"{s.key}|node{i}")
        if circuit.L > 1:
            rows.append(joint_effect(probe, circuit.nodes, s).to(torch.float64))
            keys.append(f"{s.key}|joint")
        if include_interactions:
            pairs = circuit.edges if circuit.edges else list(combinations(range(circuit.L), 2))
            for (i, j) in pairs:
                rows.append(
                    interaction_contrast(probe, circuit.nodes[i], circuit.nodes[j], s).to(torch.float64)
                )
                keys.append(f"{s.key}|gamma{i}{j}")
    return CausalSignature(
        setting_keys=keys,
        effects=torch.stack(rows, dim=0),
        scale=float(scale),
        model=probe.model.name,
        mechanism=circuit.key(),
        prompt_key=prompt_key_of(probe.data),
    )


def nodewise_signature(
    probe: CausalProbe, circuit: CircuitHypothesis, settings: Sequence[Setting], scale: float = 1.0
) -> CausalSignature:
    """Node effects only: what a nodewise translation can be held to."""
    return circuit_signature(probe, circuit, settings, scale, include_interactions=False)


# --------------------------------------------------------------- translation
def translate_circuit(
    translate: Callable[[Mechanism], Optional[Mechanism]],
    circuit: CircuitHypothesis,
    dest_model: str = "",
) -> Optional[CircuitHypothesis]:
    """Nodewise translation; abstains if any node has no counterpart."""
    nodes: List[Mechanism] = []
    for n in circuit.nodes:
        t = translate(n)
        if t is None:
            return None
        nodes.append(t)
    return CircuitHypothesis(nodes=nodes, edges=list(circuit.edges), model=dest_model,
                             label=f"translated({circuit.label})")


def permute_nodes(circuit: CircuitHypothesis) -> CircuitHypothesis:
    """Node-permuted circuit: right set of sites, wrong correspondence.

    A hard negative for a nodewise criterion. The interaction contrast is symmetric
    under a full swap, so only the *per-node* coordinates expose the error -- which
    is exactly why the signature concatenates both.
    """
    nodes = list(circuit.nodes)
    if len(nodes) > 1:
        nodes = nodes[::-1]
    return CircuitHypothesis(nodes=nodes, edges=list(circuit.edges), model=circuit.model,
                             label=f"permuted({circuit.label})")


# ---------------------------------------------------------------- evaluation
@dataclass
class CircuitResult:
    program: str
    source_model: str
    dest_model: str
    translator: str
    method: str                 # 'nodewise' | 'multisite'
    case: str
    label: int
    abstained: bool = True
    node_recovery: float = float("nan")
    edge_recovery: float = float("nan")
    interaction_error: float = float("nan")
    d_shape: float = float("nan")
    d_mag: float = float("nan")
    accepted: bool = False

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def node_recovery(proposed: CircuitHypothesis, truth: CircuitHypothesis) -> float:
    """Mean subspace similarity of matched nodes (site must agree)."""
    if proposed.L == 0 or truth.L == 0:
        return float("nan")
    sims = []
    for p, t in zip(proposed.nodes, truth.nodes):
        sims.append(subspace_similarity(p.V, t.V) if str(p.site) == str(t.site) else 0.0)
    return float(np.mean(sims))


def edge_recovery(
    probe_dest: CausalProbe,
    proposed: CircuitHypothesis,
    probe_src: CausalProbe,
    source: CircuitHypothesis,
    setting: Setting,
    rel_tol: float = 0.5,
) -> Tuple[float, float]:
    """Does each hypothesised edge survive translation?

    An edge is treated as present when the interaction contrast is non-negligible;
    it is recovered when the destination's contrast agrees with the source's in sign
    and to within ``rel_tol`` in relative magnitude.
    """
    pairs = source.edges if source.edges else list(combinations(range(source.L), 2))
    if not pairs:
        return float("nan"), float("nan")
    ok, errs = [], []
    for (i, j) in pairs:
        g_src = float(interaction_contrast(probe_src, source.nodes[i], source.nodes[j], setting).mean())
        g_dst = float(interaction_contrast(probe_dest, proposed.nodes[i], proposed.nodes[j], setting).mean())
        denom = max(abs(g_src), 1e-9)
        rel = abs(g_dst - g_src) / denom
        errs.append(rel)
        ok.append(float(np.sign(g_dst) == np.sign(g_src) and rel <= rel_tol))
    return float(np.mean(ok)), float(np.mean(errs))
