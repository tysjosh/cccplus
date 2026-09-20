"""Small hookable decoder-only transformer used by the controlled stage (Sec 4.1).

These are the low-level models that SIIT training plants labelled causal
variables into. Architectures differ in depth, width and head count across
M1/M2/M3 while implementing the same high-level program, which is exactly the
setting CCC+ is meant to be evaluated in: "independently initialized SIIT models
varying in depth, width, and head count while preserving the labeled causal
graph."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mechanisms import Site
from .base import HookedModel, Patch, PromptBatch, resolve_positions

SUBLAYERS = ("resid_pre", "head_out", "attn_out", "resid_mid", "mlp_out", "resid_post")


@dataclass
class TinyConfig:
    d_vocab: int
    n_ctx: int
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16
    d_mlp: int = 128
    d_vocab_out: Optional[int] = None

    @property
    def out_dim(self) -> int:
        return self.d_vocab_out or self.d_vocab

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class _Attention(nn.Module):
    """Causal multi-head attention with optional per-head output decomposition.

    Weights are stored flat so that Q/K/V is a single GEMM; the per-head
    decomposition (needed only when a ``head_out`` site is hooked) is computed
    lazily because it is several times more expensive.
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        h, dh, d = cfg.n_heads, cfg.d_head, cfg.d_model
        self.W_QKV = nn.Parameter(torch.empty(d, 3 * h * dh))
        self.W_O = nn.Parameter(torch.empty(h * dh, d))
        self.b_O = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.W_QKV, std=(1.0 / d) ** 0.5)
        nn.init.normal_(self.W_O, std=(1.0 / (h * dh)) ** 0.5)

    def _z(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Attention-weighted values, [B, H, T, d_head]."""
        cfg = self.cfg
        B, T, _ = x.shape
        H, dh = cfg.n_heads, cfg.d_head
        qkv = x @ self.W_QKV                                  # [B,T,3*H*dh]
        q, k, v = qkv.split(H * dh, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (dh ** 0.5)      # [B,H,T,T]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~causal, neg)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.bool()[:, None, None, :], neg)
        return scores.softmax(dim=-1) @ v                     # [B,H,T,dh]

    def attn_out(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Total attention contribution to the residual stream, [B, T, d_model]."""
        z = self._z(x, attn_mask)
        B, H, T, dh = z.shape
        flat = z.transpose(1, 2).reshape(B, T, H * dh)
        return flat @ self.W_O + self.b_O

    def head_contributions(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-head additive contribution, [B, H, T, d_model] (excludes b_O)."""
        z = self._z(x, attn_mask)
        H, dh, d = self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model
        return torch.einsum("bhtk,hkd->bhtd", z, self.W_O.view(H, dh, d))


class _MLP(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        nn.init.normal_(self.W_in.weight, std=(1.0 / cfg.d_model) ** 0.5)
        nn.init.normal_(self.W_out.weight, std=(1.0 / cfg.d_mlp) ** 0.5)
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.gelu(self.W_in(x)))


class TinyTransformer(nn.Module, HookedModel):
    """Hookable transformer with explicit residual-stream bookkeeping."""

    def __init__(self, cfg: TinyConfig, name: str = "tiny"):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.name = name
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(cfg.d_model),
                        "attn": _Attention(cfg),
                        "ln2": nn.LayerNorm(cfg.d_model),
                        "mlp": _MLP(cfg),
                    }
                )
            )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.out_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    # -------------------------------------------------------------- structure
    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    def site_dim(self, site: Site) -> int:
        return self.cfg.d_model  # every hook point here lives in the residual basis

    def candidate_sites(self, sublayers: Optional[Sequence[str]] = None) -> List[Site]:
        subs = tuple(sublayers) if sublayers else ("attn_out", "mlp_out", "resid_post")
        out: List[Site] = []
        for layer in range(self.cfg.n_layers):
            for sub in subs:
                if sub == "head_out":
                    for h in range(self.cfg.n_heads):
                        out.append(Site(layer, sub, "final", h))
                else:
                    out.append(Site(layer, sub, "final"))
        return out

    # -------------------------------------------------------------- internals
    def _forward_collect(
        self,
        batch: PromptBatch,
        patch_index: Dict[str, Tuple[torch.Tensor, "object"]],
        collect: Sequence[Site] = (),
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the model, applying patches and collecting requested activations.

        ``patch_index`` maps ``str(site)`` -> (positions[B], fn). Patches are applied
        at the natural computation point of the sublayer, so downstream layers see
        the modified value.
        """
        tokens = batch.tokens
        B, T = tokens.shape
        mask = batch.mask
        device = tokens.device
        collected: Dict[str, torch.Tensor] = {}
        wanted = {str(s): s for s in collect}

        def handle(site: Site, value: torch.Tensor) -> torch.Tensor:
            """Collect and optionally patch ``value`` [B, T, d] at ``site``."""
            key = str(site)
            need_collect = key in wanted
            patch = patch_index.get(key)
            if not need_collect and patch is None:
                return value
            pos = None
            if patch is not None:
                pos = patch[0]
            if need_collect:
                cpos = resolve_positions(site.token, batch) if pos is None else pos
                collected[key] = value[torch.arange(B, device=device), cpos].clone()
            if patch is not None:
                fn = patch[1]
                idx = torch.arange(B, device=device)
                cur = value[idx, pos]
                new = fn(cur)
                value = value.clone()
                value[idx, pos] = new.to(value.dtype)
            return value

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        resid = self.embed(tokens) + self.pos_embed(pos_ids.clamp(max=self.cfg.n_ctx - 1))

        attn_mask = None if bool((mask == 1).all()) else mask

        for layer, block in enumerate(self.blocks):
            resid = handle(Site(layer, "resid_pre", "final"), resid)
            normed = block["ln1"](resid)
            head_keys = [str(Site(layer, "head_out", "final", h)) for h in range(self.cfg.n_heads)]
            need_heads = any((k in wanted) or (k in patch_index) for k in head_keys)
            if need_heads:
                head_out = block["attn"].head_contributions(normed, attn_mask)   # [B,H,T,d]
                for h in range(self.cfg.n_heads):
                    key = head_keys[h]
                    if key in wanted or key in patch_index:
                        patched = handle(Site(layer, "head_out", "final", h), head_out[:, h])
                        head_out = head_out.clone()
                        head_out[:, h] = patched
                attn_out = head_out.sum(dim=1) + block["attn"].b_O
            else:
                attn_out = block["attn"].attn_out(normed, attn_mask)
            attn_out = handle(Site(layer, "attn_out", "final"), attn_out)
            resid = resid + attn_out
            resid = handle(Site(layer, "resid_mid", "final"), resid)
            mlp_out = block["mlp"](block["ln2"](resid))
            mlp_out = handle(Site(layer, "mlp_out", "final"), mlp_out)
            resid = resid + mlp_out
            resid = handle(Site(layer, "resid_post", "final"), resid)

        logits = self.unembed(self.ln_f(resid))                          # [B, T, out]
        return logits, collected

    @staticmethod
    def _patch_index(batch: PromptBatch, patches: Sequence[Patch]) -> Dict[str, Tuple[torch.Tensor, object]]:
        idx: Dict[str, Tuple[torch.Tensor, object]] = {}
        for p in patches:
            pos = p.positions if p.positions is not None else resolve_positions(p.site.token, batch)
            idx[str(p.site)] = (pos.to(batch.tokens.device), p.fn)
        return idx

    # -------------------------------------------------------------- interface
    def read(self, batch: PromptBatch, sites: Sequence[Site]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            _, acts = self._forward_collect(batch, {}, collect=sites)
        return acts

    def read_grad(self, batch: PromptBatch, sites: Sequence[Site]) -> Dict[str, torch.Tensor]:
        """Activation read that keeps the autograd graph (used by MAS-style baselines)."""
        _, acts = self._forward_collect(batch, {}, collect=sites)
        return acts

    def logits(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        with torch.no_grad():
            return self._logits_inner(batch, patches)

    def logits_grad(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        return self._logits_inner(batch, patches)

    def _logits_inner(self, batch: PromptBatch, patches: Sequence[Patch]) -> torch.Tensor:
        logits, _ = self._forward_collect(batch, self._patch_index(batch, patches))
        pos = resolve_positions(batch.readout, batch)
        return logits[torch.arange(batch.tokens.shape[0], device=logits.device), pos]

    def full_logits(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        logits, _ = self._forward_collect(batch, self._patch_index(batch, patches))
        return logits

    # -------------------------------------------------------------- persistence
    def save(self, path) -> None:
        torch.save({"cfg": self.cfg.to_dict(), "state": self.state_dict(), "name": self.name}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "TinyTransformer":
        blob = torch.load(path, map_location=map_location)
        model = cls(TinyConfig(**blob["cfg"]), name=blob.get("name", "tiny"))
        model.load_state_dict(blob["state"])
        model.eval()
        return model
