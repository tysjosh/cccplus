"""Planted causal structure for the controlled stage (Sec. 4.1).

"Each program supplies (i) exact planted channel permutations, splits, merges,
redundancies, and absent-variable cases and (ii) independently initialized SIIT
models varying in depth, width, and head count while preserving the labeled
causal graph."

A *placement* assigns each high-level variable to one or more orthonormal
subspaces of additive sublayer outputs. Because the subspaces are drawn from a
random orthogonal matrix per site, they are not axis-aligned: a translator has to
recover a genuine linear correspondence, not a coordinate permutation.

Structures
  plain             each variable in one rank-r block
  permuted          same, with the variable -> slot map permuted (planted permutation)
  split             one variable spread over two rank-r/2 blocks at two sites;
                    both are needed to transfer it
  redundant         one variable duplicated in two rank-r blocks; either suffices
  merged            two variables share a single rank-r block
  absent            a variable is not represented at all (the program variant
                    ignores it), so the only correct answer is abstention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..mechanisms import Mechanism, Site, orthonormalize

STRUCTURES = ("plain", "permuted", "split_redundant", "merged", "absent")


@dataclass
class PlantedBlock:
    """One planted subspace."""

    site: Site
    V: torch.Tensor                 # [d, r] orthonormal
    vars: FrozenSet[int]            # high-level variables carried by this block
    slot: int = 0                   # index among blocks at this site
    tag: str = "plain"              # plain | split | redundant | merged

    @property
    def rank(self) -> int:
        return int(self.V.shape[1])

    def mechanism(self, model_name: str = "") -> Mechanism:
        return Mechanism(
            site=self.site,
            V=self.V,
            model=model_name,
            label=f"{self.tag}:{'+'.join(f'v{v}' for v in sorted(self.vars))}",
            meta={"vars": sorted(self.vars), "tag": self.tag, "slot": self.slot},
        )


@dataclass
class PlantedGroup:
    """Blocks that must be patched together to transfer ``vars``.

    Split variables give a group of two blocks; redundant variables give two
    single-block groups (plus their union); merged variables give one block that
    transfers two variables at once.
    """

    blocks: List[PlantedBlock]
    vars: FrozenSet[int]
    tag: str = "plain"


@dataclass
class Placement:
    """The full planted structure of one model instance."""

    structure: str
    blocks: List[PlantedBlock]
    groups: List[PlantedGroup]
    site_bases: Dict[str, torch.Tensor]        # str(site) -> [d, d] orthogonal
    absent_vars: Set[int] = field(default_factory=set)
    n_vars: int = 3

    # ------------------------------------------------------------- accessors
    def sites(self) -> List[Site]:
        seen: Dict[str, Site] = {}
        for b in self.blocks:
            seen.setdefault(str(b.site), b.site)
        return list(seen.values())

    def blocks_at(self, site: Site) -> List[PlantedBlock]:
        return [b for b in self.blocks if str(b.site) == str(site)]

    def blocks_for_var(self, var: int) -> List[PlantedBlock]:
        return [b for b in self.blocks if var in b.vars]

    def groups_for_var(self, var: int) -> List[PlantedGroup]:
        return [g for g in self.groups if var in g.vars]

    def complement(self, site: Site) -> torch.Tensor:
        """Orthonormal basis of the complement of every planted block at ``site``."""
        Q = self.site_bases[str(site)]
        used = torch.cat([b.V for b in self.blocks_at(site)], dim=1)
        # blocks are carved from the columns of Q, so complement = unused columns
        proj = (Q.T @ used).abs().sum(dim=1)
        keep = proj < 1e-6
        return Q[:, keep]

    # --------------------------------------------------------- ground truth
    def counterpart(self, var: int) -> Dict[str, object]:
        """Planted truth for a source variable, as seen from this model."""
        if var in self.absent_vars or not self.blocks_for_var(var):
            return {"case": "absent", "blocks": [], "exclusive": False}
        blocks = self.blocks_for_var(var)
        tags = {b.tag for b in blocks}
        exclusive = all(b.vars == frozenset({var}) for b in blocks)
        if "merged" in tags:
            case = "merged"
        elif "split" in tags:
            case = "split"
        elif "redundant" in tags:
            case = "redundant"
        else:
            case = "permuted" if self.structure == "permuted" else "shared"
        return {"case": case, "blocks": blocks, "exclusive": exclusive}

    def decoy_blocks(self, var: int) -> List[PlantedBlock]:
        """Planted blocks of *other* variables: rank- and effect-matched negatives."""
        return [b for b in self.blocks if var not in b.vars]

    def summary(self) -> Dict[str, object]:
        return {
            "structure": self.structure,
            "absent_vars": sorted(self.absent_vars),
            "blocks": [
                {"site": str(b.site), "rank": b.rank, "vars": sorted(b.vars), "tag": b.tag}
                for b in self.blocks
            ],
        }


# ------------------------------------------------------------------ builders
def mechanism_sites(n_layers: int, sublayers: Sequence[str] = ("mlp_out", "attn_out")) -> List[Site]:
    """Sites eligible to host a planted mechanism.

    The final layer's ``mlp_out`` is deliberately left unplanted so the model has
    somewhere to combine the variables into the non-additive output function.
    Attention outputs of early layers are also left free so information can be
    moved to the readout position; that keeps the strict-localisation objective
    satisfiable.
    """
    sites: List[Site] = []
    if "mlp_out" in sublayers:
        for layer in range(max(0, n_layers - 1)):
            sites.append(Site(layer, "mlp_out", "final"))
    if "attn_out" in sublayers:
        for layer in range(1, n_layers):
            sites.append(Site(layer, "attn_out", "final"))
    if not sites:
        raise ValueError("no eligible mechanism sites")
    return sites


def _random_basis(d: int, gen: torch.Generator) -> torch.Tensor:
    A = torch.randn(d, d, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q


def build_placement(
    structure: str,
    n_layers: int,
    d_model: int,
    rank: int,
    seed: int,
    n_vars: int = 3,
    sublayers: Sequence[str] = ("mlp_out", "attn_out"),
) -> Placement:
    """Construct a planted structure of the requested type."""
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure '{structure}'")
    gen = torch.Generator().manual_seed(int(seed))
    sites = mechanism_sites(n_layers, sublayers)
    rng = np.random.default_rng(seed)

    # How many subspace slots each structure needs, and what they carry.
    if structure in ("plain", "permuted"):
        slots: List[Tuple[FrozenSet[int], int, str]] = [
            (frozenset({k}), rank, "plain") for k in range(n_vars)
        ]
        if structure == "permuted":
            perm = rng.permutation(n_vars)
            slots = [(frozenset({int(perm[k])}), rank, "plain") for k in range(n_vars)]
        absent: Set[int] = set()
    elif structure == "split_redundant":
        half = max(1, rank // 2)
        slots = [
            (frozenset({0}), half, "split"),
            (frozenset({0}), half, "split"),
            (frozenset({1}), rank, "redundant"),
            (frozenset({1}), rank, "redundant"),
            (frozenset({2}), rank, "plain"),
        ]
        absent = set()
    elif structure == "merged":
        slots = [
            (frozenset({0}), rank, "plain"),
            (frozenset({1, 2}), rank, "merged"),
        ]
        absent = set()
    else:  # absent
        slots = [(frozenset({0}), rank, "plain"), (frozenset({1}), rank, "plain")]
        absent = {2}

    # Distribute slots over sites; a site may host several orthogonal blocks.
    site_order = [sites[i % len(sites)] for i in range(len(slots))]
    rng.shuffle(site_order)
    # Split / redundant pairs must land on *different* sites.
    for i in range(0, len(slots)):
        pass
    if structure == "split_redundant":
        site_order = _force_distinct_pairs(site_order, sites, [(0, 1), (2, 3)], rng)

    site_bases: Dict[str, torch.Tensor] = {}
    next_col: Dict[str, int] = {}
    blocks: List[PlantedBlock] = []
    for (vars_, r, tag), site in zip(slots, site_order):
        key = str(site)
        if key not in site_bases:
            site_bases[key] = _random_basis(d_model, gen)
            next_col[key] = 0
        start = next_col[key]
        if start + r > d_model:
            raise ValueError(f"site {key} out of capacity")
        V = site_bases[key][:, start : start + r].contiguous()
        next_col[key] = start + r
        blocks.append(PlantedBlock(site=site, V=V, vars=vars_, slot=start // max(r, 1), tag=tag))

    groups = _groups_for(structure, blocks)
    return Placement(
        structure=structure,
        blocks=blocks,
        groups=groups,
        site_bases=site_bases,
        absent_vars=absent,
        n_vars=n_vars,
    )


def _force_distinct_pairs(
    site_order: List[Site], sites: List[Site], pairs: Sequence[Tuple[int, int]], rng
) -> List[Site]:
    out = list(site_order)
    for i, j in pairs:
        if str(out[i]) != str(out[j]):
            continue
        alternatives = [s for s in sites if str(s) != str(out[i])]
        if alternatives:
            out[j] = alternatives[int(rng.integers(0, len(alternatives)))]
    return out


def _groups_for(structure: str, blocks: List[PlantedBlock]) -> List[PlantedGroup]:
    groups: List[PlantedGroup] = []
    if structure == "split_redundant":
        split_blocks = [b for b in blocks if b.tag == "split"]
        red_blocks = [b for b in blocks if b.tag == "redundant"]
        plain_blocks = [b for b in blocks if b.tag == "plain"]
        groups.append(PlantedGroup(split_blocks, frozenset({0}), "split"))
        for b in red_blocks:                      # each alone must suffice
            groups.append(PlantedGroup([b], frozenset({1}), "redundant"))
        groups.append(PlantedGroup(red_blocks, frozenset({1}), "redundant_joint"))
        for b in plain_blocks:
            groups.append(PlantedGroup([b], b.vars, "plain"))
    else:
        for b in blocks:
            groups.append(PlantedGroup([b], b.vars, b.tag))
    return groups
