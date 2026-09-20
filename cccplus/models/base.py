"""Model-agnostic hooking interface.

Every causal test in CCC+ needs exactly two primitives:

  * read the activation vector at a site and token position, and
  * run the model while replacing that activation with a function of itself.

Both the synthetic SIIT transformers (Sec. 4.1) and the pretrained language
models (Sec. 4.2) implement this interface, so `interventions.py`,
`signature.py` and `validation.py` never branch on model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ..mechanisms import Site

# fn(activations[B, d]) -> replacement[B, d]
PatchFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class PromptBatch:
    """A batch of prompts plus the token positions that sites and readouts use.

    ``positions`` maps an anchor name (e.g. ``"final"``, ``"subject_last"``) to a
    per-example index, which is how a single Site definition can be applied to
    prompts of different lengths.
    """

    tokens: torch.Tensor                      # [B, T] int64
    attn_mask: Optional[torch.Tensor] = None  # [B, T] int64/bool, None => all ones
    positions: Dict[str, torch.Tensor] = field(default_factory=dict)
    readout: str = "final"
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def mask(self) -> torch.Tensor:
        if self.attn_mask is None:
            return torch.ones_like(self.tokens)
        return self.attn_mask

    def index(self, anchor) -> torch.Tensor:
        return resolve_positions(anchor, self)

    def subset(self, idx: Sequence[int]) -> "PromptBatch":
        sel = torch.as_tensor(list(idx), dtype=torch.long)
        return PromptBatch(
            tokens=self.tokens[sel],
            attn_mask=None if self.attn_mask is None else self.attn_mask[sel],
            positions={k: v[sel] for k, v in self.positions.items()},
            readout=self.readout,
            meta=dict(self.meta),
        )

    def to(self, device: torch.device) -> "PromptBatch":
        return PromptBatch(
            tokens=self.tokens.to(device),
            attn_mask=None if self.attn_mask is None else self.attn_mask.to(device),
            positions={k: v.to(device) for k, v in self.positions.items()},
            readout=self.readout,
            meta=dict(self.meta),
        )


def resolve_positions(anchor, batch: PromptBatch) -> torch.Tensor:
    """Turn a Site token anchor into a per-example position index [B]."""
    B, T = batch.tokens.shape
    if isinstance(anchor, (int,)) and not isinstance(anchor, bool):
        a = int(anchor)
        if a < 0:
            a = T + a
        return torch.full((B,), a, dtype=torch.long, device=batch.tokens.device)
    if isinstance(anchor, str):
        if anchor in batch.positions:
            return batch.positions[anchor].to(torch.long)
        if anchor == "final":
            # last non-pad token
            lengths = batch.mask.sum(dim=1)
            return (lengths - 1).clamp(min=0).to(torch.long)
        raise KeyError(f"unknown token anchor '{anchor}'; batch has {sorted(batch.positions)}")
    if torch.is_tensor(anchor):
        return anchor.to(torch.long)
    raise TypeError(f"unsupported token anchor: {anchor!r}")


@dataclass
class Patch:
    """One activation replacement: apply ``fn`` at ``site`` on the given positions."""

    site: Site
    fn: PatchFn
    positions: Optional[torch.Tensor] = None   # [B] indices; None => resolve from site.token


class HookedModel(ABC):
    """Minimal read/patch interface required by the CCC+ protocol."""

    name: str = "model"

    # ----------------------------------------------------------- structure
    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @abstractmethod
    def site_dim(self, site: Site) -> int:
        """Activation width at a site (``d_s`` in the paper)."""

    @abstractmethod
    def candidate_sites(self, sublayers: Optional[Sequence[str]] = None) -> List[Site]:
        """Sites eligible to host a mechanism, with an unresolved token anchor."""

    # ----------------------------------------------------------- execution
    @abstractmethod
    def read(self, batch: PromptBatch, sites: Sequence[Site]) -> Dict[str, torch.Tensor]:
        """Activations at each site, keyed by ``str(site)``, shaped [B, d_s]."""

    @abstractmethod
    def logits(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        """Readout-position logits [B, vocab] under zero or more patches."""

    # ----------------------------------------------------------- utilities
    def read_site(self, batch: PromptBatch, site: Site) -> torch.Tensor:
        return self.read(batch, [site])[str(site)]

    @torch.no_grad()
    def log_probs(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        return torch.log_softmax(self.logits(batch, patches).to(torch.float32), dim=-1)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name} L={self.n_layers} d={self.d_model}>"
