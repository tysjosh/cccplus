"""Pretrained causal-LM adapter for the natural-model stage (Sec. 4.2).

Implements the same ``HookedModel`` interface as the synthetic transformers, so
``interventions.py``, ``signature.py``, ``validation.py`` and ``circuits.py`` run
against Pythia, Gemma-3, Llama-3.2, GPT-2 and friends without modification.

Hook points are submodule boundaries, which exist in every decoder architecture:

  resid_pre    input to a decoder layer
  attn_out     output of the attention block, before the residual add
  mlp_out      output of the MLP block, before the residual add
  resid_post   output of a decoder layer

Attention *heads* are addressed without per-head hooks. A head's causal footprint
is the subspace of the residual stream it writes into, i.e. the column space of its
slice of the output projection, so the paper's IOI mechanism -- "V equal to the
residual-stream column space of the head output projection" -- is obtained by
patching ``attn_out`` restricted to ``head_write_subspace(layer, head)``. This is
exactly Eq. 6 applied to that head and needs no architecture-specific head
plumbing.

Token positions are resolved per model: the same text tokenises differently across
architectures, so each model computes its own anchor indices for a shared prompt
list. Signature coordinates are indexed by prompt, not by token.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..config import stable_seed
from ..mechanisms import Site, orthonormalize
from .base import HookedModel, Patch, PromptBatch, resolve_positions

# Attribute names per architecture family. Probed in order; the first hit wins.
_LAYER_PATHS = (
    "model.layers",                      # Llama, Mistral, Qwen, Gemma-3 1B (text-only)
    "gpt_neox.layers",                   # Pythia
    "transformer.h",                     # GPT-2
    "model.decoder.layers",              # OPT
    # Multimodal wrappers put the decoder behind a vision-language container. Gemma-3 4B
    # and up load as Gemma3ForConditionalGeneration, not Gemma3ForCausalLM, and the exact
    # nesting moved between transformers releases, so all the observed spellings are tried.
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
)

# Submodule names that indicate a non-text tower, which must never be mistaken for the
# decoder stack: a vision encoder also holds a ModuleList of attention+MLP blocks.
_NON_TEXT_HINTS = ("vision", "visual", "image", "audio", "speech", "multi_modal")
_ATTN_NAMES = ("self_attn", "attention", "attn")
_MLP_NAMES = ("mlp", "feed_forward")
_OPROJ_NAMES = ("o_proj", "dense", "c_proj", "out_proj")

SUBLAYERS = ("resid_pre", "attn_out", "mlp_out", "resid_post")

# Prompts per forward pass when logits are needed. The logit tensor is
# batch * tokens * vocab, and modern vocabularies are large enough that an unbatched
# forward over a full prompt set does not fit: Gemma-3's 262k vocab needs ~23 GiB in
# bf16 for 2400 prompts of 20 tokens, Llama-3.2's 128k needs ~11 GiB. Activation reads
# are unaffected (d_model is ~100x smaller than a vocabulary) and stay unbatched.
_LOGIT_CHUNK = int(os.environ.get("CCC_LOGIT_CHUNK", "16"))

# Prompts per forward pass when only activations are wanted. Larger than the logit chunk
# because the collected tensors are one position per prompt, and because the LM head is
# sliced down to a single position for these passes.
_READ_CHUNK = int(os.environ.get("CCC_READ_CHUNK", "32"))


def _resolve(root: torch.nn.Module, dotted: str):
    node = root
    for part in dotted.split("."):
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def _first_attr(module: torch.nn.Module, names: Sequence[str]):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n), n
    return None, ""


def _discover_layers(root: torch.nn.Module, expected_n: Optional[int]):
    """Locate the decoder stack when no known path matches.

    Searches for a ``ModuleList`` whose blocks carry both an attention and an MLP
    submodule. Two guards keep it from picking the wrong stack on a multimodal
    checkpoint, where the vision tower has the same shape: module paths naming a
    non-text tower are skipped, and the length must equal the text config's layer
    count. Returns ``(path, module_list)`` or ``(None, None)``.
    """
    best: Tuple[Optional[str], Optional[torch.nn.ModuleList]] = (None, None)
    for path, mod in root.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) == 0:
            continue
        if any(h in path.lower() for h in _NON_TEXT_HINTS):
            continue
        block = mod[0]
        if _first_attr(block, _ATTN_NAMES)[0] is None:
            continue
        if _first_attr(block, _MLP_NAMES)[0] is None:
            continue
        if expected_n is not None and len(mod) != expected_n:
            continue
        if best[1] is None or len(mod) > len(best[1]):
            best = (path, mod)
    return best


def _transformers_at_least(version: str, major: int, minor: int) -> bool:
    """Compare a transformers version string against (major, minor).

    Tolerates suffixes like '4.57.0.dev0' and short strings like '5.0'.
    """
    parts: List[int] = []
    for chunk in version.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1]) >= (major, minor)


@dataclass
class HFConfigInfo:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int


class HFCausalLM(HookedModel):
    """Hookable wrapper around a Hugging Face causal language model."""

    def __init__(
        self,
        model,
        tokenizer,
        name: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)      # Gemma-3 nests the text config
        expected_n = getattr(text_cfg, "num_hidden_layers", None)
        expected_n = int(expected_n) if expected_n is not None else None

        self._layers = None
        self._layers_path = ""
        for path in _LAYER_PATHS:
            got = _resolve(self.model, path)
            if got is not None and len(got) > 0:
                self._layers, self._layers_path = got, path
                break
        if self._layers is None:
            # Unknown wrapper: fall back to structural discovery rather than failing, so a
            # new multimodal container does not block the stage.
            self._layers_path, self._layers = _discover_layers(self.model, expected_n)
        if self._layers is None:
            raise ValueError(
                f"could not locate decoder layers on {type(model).__name__}; tried "
                f"{list(_LAYER_PATHS)} and structural discovery for "
                f"{expected_n} layers"
            )
        if expected_n is not None and len(self._layers) != expected_n:
            raise ValueError(
                f"located {len(self._layers)} layers at '{self._layers_path}' on "
                f"{type(model).__name__}, but its text config declares {expected_n}. "
                f"Refusing to continue: on a multimodal checkpoint this usually means the "
                f"vision tower was picked up instead of the decoder."
            )

        n_heads = int(getattr(text_cfg, "num_attention_heads"))
        d_model = int(getattr(text_cfg, "hidden_size"))
        self.info = HFConfigInfo(
            n_layers=len(self._layers),
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=int(getattr(text_cfg, "num_key_value_heads", n_heads)),
            d_head=int(getattr(text_cfg, "head_dim", d_model // n_heads)),
        )
        self._subspace_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------ construction
    @classmethod
    def load(
        cls,
        repo: str,
        name: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "HFCausalLM":
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        # `torch_dtype` was renamed to `dtype` in transformers 4.56 and is deprecated
        # there. The controlled-stage environment pins 4.46.3, but Gemma-3 needs >= 4.50,
        # so the loader has to span both spellings: `from_pretrained` swallows **kwargs,
        # which means the wrong name is silently ignored and the model loads in fp32.
        dtype_key = "dtype" if _transformers_at_least(transformers.__version__, 4, 56) \
            else "torch_dtype"
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, trust_remote_code=trust_remote_code,
            **{dtype_key: torch_dtype},
        ).to(device)
        if model.dtype != torch_dtype:
            raise RuntimeError(
                f"requested {torch_dtype} but loaded {model.dtype}; the dtype argument was "
                f"ignored (transformers {transformers.__version__}, tried '{dtype_key}'). "
                f"Loading a multi-billion-parameter checkpoint in the wrong precision "
                f"silently doubles memory, so this fails rather than continues."
            )
        return cls(model, tok, name or repo, torch.device(device), torch_dtype)

    @classmethod
    def from_random_config(cls, kind: str, name: str, device: str = "cpu",
                           seed: Optional[int] = None, **overrides) -> "HFCausalLM":
        """Randomly initialised small model of a given family, for offline testing.

        Exercises the hook wiring, head subspaces and position anchors without
        downloading any weights.

        The weights are seeded from the model's ``name``. ``transformers`` initialises
        from the *global* torch RNG, so without this the weights depended on whatever
        had consumed that RNG earlier in the process: the self-test stage was not
        reproducible between runs, and test outcomes changed with execution order. The
        surrounding RNG state is saved and restored so seeding here cannot perturb a
        caller's stream.
        """
        d = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                 intermediate_size=128, vocab_size=512, max_position_embeddings=64)
        d.update(overrides)
        sd = stable_seed("hf_random_config", kind, name) if seed is None else int(seed)
        rng_state = torch.get_rng_state()
        torch.manual_seed(sd % (2 ** 31))
        try:
            model = cls._build_random(kind, d)
        finally:
            torch.set_rng_state(rng_state)
        tok = _DummyTokenizer(d["vocab_size"])
        obj = cls(model, tok, name, torch.device(device), torch.float32)
        # Recorded so a self-test artifact can be told apart from real weights.
        obj.random_config = {"kind": kind, "seed": sd, **{k: d[k] for k in
                             ("hidden_size", "num_hidden_layers")}}
        return obj

    @staticmethod
    def _build_random(kind: str, d: Dict[str, int]):
        if kind == "llama":
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(num_key_value_heads=d["num_attention_heads"], **d)
            model = LlamaForCausalLM(cfg)
        elif kind == "gpt_neox":
            from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
            cfg = GPTNeoXConfig(rotary_pct=0.25, **d)
            model = GPTNeoXForCausalLM(cfg)
        elif kind == "gpt2":
            from transformers import GPT2Config, GPT2LMHeadModel
            cfg = GPT2Config(n_embd=d["hidden_size"], n_layer=d["num_hidden_layers"],
                             n_head=d["num_attention_heads"], n_inner=d["intermediate_size"],
                             vocab_size=d["vocab_size"], n_positions=d["max_position_embeddings"])
            model = GPT2LMHeadModel(cfg)
        else:
            raise ValueError(f"unknown kind {kind}")
        return model

    # -------------------------------------------------------------- structure
    @property
    def n_layers(self) -> int:
        return self.info.n_layers

    @property
    def d_model(self) -> int:
        return self.info.d_model

    def site_dim(self, site: Site) -> int:
        return self.info.d_model

    def candidate_sites(self, sublayers: Optional[Sequence[str]] = None) -> List[Site]:
        subs = tuple(sublayers) if sublayers else ("attn_out", "mlp_out")
        return [Site(l, s, "final") for l in range(self.n_layers) for s in subs]

    def site_grid(self, sublayers: Sequence[str], n_depths: int, token: object = "final") -> List[Site]:
        """Preregistered site family at evenly spaced relative depths.

        Translator maps are fitted per site pair, so the grid bounds that cost while
        keeping "Equal layer indices are not required" -- relative depth, not absolute
        index, is what is matched across architectures of different depth.
        """
        if n_depths >= self.n_layers:
            layers = list(range(self.n_layers))
        else:
            layers = sorted({int(round(f * (self.n_layers - 1))) for f in
                             [i / max(n_depths - 1, 1) for i in range(n_depths)]})
        return [Site(l, s, token) for l in layers for s in sublayers]

    # ------------------------------------------------------- head subspaces
    def _oproj_weight(self, layer: int) -> torch.Tensor:
        block = self._layers[layer]
        attn, _ = _first_attr(block, _ATTN_NAMES)
        if attn is None:
            raise ValueError(f"no attention module on layer {layer}")
        proj, pname = _first_attr(attn, _OPROJ_NAMES)
        if proj is None:
            raise ValueError(f"no output projection on layer {layer}")
        W = proj.weight
        # Linear stores [out, in]; GPT-2's Conv1D stores [in, out].
        if pname == "c_proj" and W.shape[0] == self.info.d_model and W.shape[1] == self.info.d_model:
            W = W.T
        if W.shape[0] != self.info.d_model:
            W = W.T
        return W                                  # [d_model, n_heads * d_head]

    def head_write_subspace(self, layer: int, head: int, rank: Optional[int] = None) -> torch.Tensor:
        """Orthonormal basis of the residual subspace head ``head`` writes into.

        This is the paper's IOI mechanism: the column space of the head's slice of the
        output projection. Patching ``attn_out`` restricted to it is equivalent to
        intervening on that head's contribution.
        """
        key = (layer, head)
        if key not in self._subspace_cache:
            W = self._oproj_weight(layer).to(torch.float32)
            dh = W.shape[1] // self.info.n_heads
            block = W[:, head * dh : (head + 1) * dh]
            self._subspace_cache[key] = orthonormalize(block.detach().cpu())
        V = self._subspace_cache[key]
        return V[:, :rank] if rank else V

    # ---------------------------------------------------------------- hooking
    def _module_for(self, site: Site):
        block = self._layers[site.layer]
        if site.sublayer in ("resid_pre", "resid_post"):
            return block, site.sublayer
        if site.sublayer == "attn_out":
            mod, _ = _first_attr(block, _ATTN_NAMES)
            return mod, "out"
        if site.sublayer == "mlp_out":
            mod, _ = _first_attr(block, _MLP_NAMES)
            return mod, "out"
        raise ValueError(f"unsupported sublayer '{site.sublayer}' for HF models")

    def _register(self, sites: Sequence[Site], batch: PromptBatch,
                  collect: Optional[Dict[str, torch.Tensor]],
                  patches: Dict[str, Tuple[torch.Tensor, object]],
                  rows: Optional[torch.Tensor] = None):
        handles = []
        B = batch.tokens.shape[0]
        ar = torch.arange(B, device=batch.tokens.device)

        def make(site: Site):
            key = str(site)
            mod, which = self._module_for(site)

            def grab_and_patch(t: torch.Tensor) -> torch.Tensor:
                if collect is not None and key in collect_keys:
                    pos = patches[key][0] if key in patches else resolve_positions(site.token, batch)
                    collect[key] = t[ar, pos].detach().to(torch.float32).cpu()
                if key in patches:
                    pos, fn = patches[key]
                    cur = t[ar, pos]
                    # A patch may close over a per-prompt tensor. When the forward pass is
                    # split into row chunks, such a function has to be told which rows it
                    # is seeing; those declare themselves with `row_aware`.
                    if rows is not None and getattr(fn, "row_aware", False):
                        new = fn(cur.to(torch.float32), rows)
                    else:
                        new = fn(cur.to(torch.float32))
                    # Patch functions close over activations that were read back to the
                    # CPU, so normalise device as well as dtype. Casting dtype alone was
                    # silently fine while every stage ran on CPU.
                    new = new.to(device=t.device, dtype=t.dtype)
                    t = t.clone()
                    t[ar, pos] = new
                return t

            if which == "resid_pre":
                def pre_hook(module, args, kwargs):
                    if not args:
                        return None
                    return (grab_and_patch(args[0]),) + tuple(args[1:]), kwargs
                return mod.register_forward_pre_hook(pre_hook, with_kwargs=True)

            def fwd_hook(module, args, output):
                if isinstance(output, tuple):
                    return (grab_and_patch(output[0]),) + tuple(output[1:])
                return grab_and_patch(output)
            return mod.register_forward_hook(fwd_hook)

        collect_keys = {str(s) for s in sites} if collect is not None else set()
        for s in sites:
            handles.append(make(s))
        return handles

    # ------------------------------------------------------------- interface
    def _no_logits_kwargs(self) -> Dict[str, int]:
        """Ask the model to run the LM head over a single position.

        ``read`` wants hidden states, which arrive through hooks; the logits a causal LM
        computes on the way out are discarded immediately. With a 262k vocabulary that
        discarded tensor is tens of GiB. Transformers takes a parameter to slice the head
        down to the last positions -- the name changed between releases, so both are
        probed, and an unrecognised signature simply means no saving.
        """
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return {}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                return {name: 1}
        return {}

    @torch.no_grad()
    def read(
        self, batch: PromptBatch, sites: Sequence[Site], chunk: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """In-subspace activations at each site, [B, d_model] per site.

        Chunked for the same reason as ``logits``, and with the LM head suppressed: a
        factual-recall calibration split is long enough that the throwaway logits alone
        exceeded GPU memory on the larger stratum. Collected activations are small --
        one position per prompt -- so only the forward pass is split.
        """
        sites = list(sites)
        n = len(batch)
        size = max(1, min(int(chunk or _READ_CHUNK), n))
        fwd_kwargs = self._no_logits_kwargs()
        parts: Dict[str, List[torch.Tensor]] = {}
        for lo in range(0, n, size):
            hi = min(lo + size, n)
            sub = batch.subset(torch.arange(lo, hi))
            got: Dict[str, torch.Tensor] = {}
            handles = self._register(sites, sub, got, {})
            try:
                self.model(input_ids=sub.tokens.to(self.device),
                           attention_mask=sub.mask.to(self.device),
                           use_cache=False, **fwd_kwargs)
            finally:
                for h in handles:
                    h.remove()
            for k, v in got.items():
                parts.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0) for k, v in parts.items()}

    @torch.no_grad()
    def logits(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        """Readout-position logits [B, vocab] under zero or more patches.

        Only the readout row of each prompt is kept, and it is taken *inside* the chunk
        loop, so the full [B, T, V] tensor is never materialised.
        """
        pos_all = resolve_positions(batch.readout, batch)
        parts = []
        for rows, chunk_logits in self.iter_full_logits(batch, patches):
            p = pos_all[rows].to(chunk_logits.device)
            sel = chunk_logits[torch.arange(chunk_logits.shape[0], device=chunk_logits.device), p]
            parts.append(sel.to(torch.float32))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def iter_full_logits(
        self, batch: PromptBatch, patches: Sequence[Patch] = (), chunk: Optional[int] = None
    ):
        """Yield ``(row_indices, logits[chunk, T, V])`` a slice of prompts at a time.

        A single forward over every prompt allocates ``B * T * vocab`` and these
        vocabularies are large: at 2400 prompts and 20 tokens that is 23 GiB in bf16 for
        Gemma-3's 262k vocab and 11 GiB for Llama-3.2's 128k, before any float32 cast.
        Callers that reduce over the vocabulary (a readout row, or a gathered target
        token) should consume this and reduce per chunk, which bounds peak memory at
        ``chunk * T * vocab`` regardless of how many prompts were requested.

        Tune with the ``CCC_LOGIT_CHUNK`` environment variable.
        """
        n = len(batch)
        size = max(1, min(int(chunk or _LOGIT_CHUNK), n))
        # A patch that closes over a per-prompt tensor can only be chunked if it accepts
        # the row indices. Rather than risk slicing one incorrectly, fall back to a single
        # pass when any patch is not row-aware: wrong numbers are worse than a large
        # allocation, and the paths that matter for memory (unpatched screening, and the
        # Eq. 6 interventions) are both row-aware.
        if any(not getattr(p.fn, "row_aware", False) for p in patches):
            size = n
        # Patch positions are resolved once against the full batch, then sliced, so a
        # chunk sees exactly the positions its rows would have seen unchunked.
        resolved = [
            (p, p.positions if p.positions is not None
                else resolve_positions(p.site.token, batch))
            for p in patches
        ]
        sites = [p.site for p in patches]
        for lo in range(0, n, size):
            hi = min(lo + size, n)
            rows = torch.arange(lo, hi)
            sub = batch.subset(rows)
            idx: Dict[str, Tuple[torch.Tensor, object]] = {
                str(p.site): (pos[lo:hi].to(self.device), p.fn) for p, pos in resolved
            }
            handles = self._register(sites, sub, None, idx, rows=rows)
            try:
                out = self.model(input_ids=sub.tokens.to(self.device),
                                 attention_mask=sub.mask.to(self.device), use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            yield rows, out.logits

    @torch.no_grad()
    def full_logits(self, batch: PromptBatch, patches: Sequence[Patch] = ()) -> torch.Tensor:
        """Full [B, T, vocab] logits in float32.

        Retained for callers that genuinely need every position. Prefer
        ``iter_full_logits`` and reduce per chunk: concatenating here reassembles the
        very tensor whose size caused an out-of-memory failure on the larger stratum.
        """
        return torch.cat([c.to(torch.float32) for _rows, c in self.iter_full_logits(batch, patches)],
                         dim=0)

    # -------------------------------------------------------------- tokenising
    def encode(self, texts: Sequence[str], max_length: Optional[int] = None) -> PromptBatch:
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                             truncation=max_length is not None, max_length=max_length)
        tokens = enc["input_ids"]
        mask = enc.get("attention_mask", torch.ones_like(tokens))
        lengths = mask.sum(dim=1)
        return PromptBatch(tokens=tokens, attn_mask=mask,
                           positions={"final": (lengths - 1).clamp(min=0).to(torch.long)},
                           readout="final", meta={"texts": list(texts)})

    def last_token_index_of(self, texts: Sequence[str], spans: Sequence[str]) -> torch.Tensor:
        """Index of the final token of ``span`` inside each text.

        Located by tokenising the prefix up to the end of the span, which is robust to
        the whitespace and sub-word conventions that differ between tokenisers.
        """
        out = []
        for text, span in zip(texts, spans):
            cut = text.find(span)
            if cut < 0:
                raise ValueError(f"span {span!r} not found in {text!r}")
            prefix = text[: cut + len(span)]
            n = len(self.tokenizer(prefix, add_special_tokens=True)["input_ids"])
            out.append(max(n - 1, 0))
        return torch.as_tensor(out, dtype=torch.long)

    def token_id_of(self, word: str, prepend_space: bool = True) -> int:
        """Single-token id for an answer word, as IOI scoring requires."""
        s = (" " + word.strip()) if prepend_space else word.strip()
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        return int(ids[0])

    def is_single_token(self, word: str, prepend_space: bool = True) -> bool:
        s = (" " + word.strip()) if prepend_space else word.strip()
        return len(self.tokenizer(s, add_special_tokens=False)["input_ids"]) == 1

    def free(self) -> None:
        """Release weights; activations are cached to disk so models load one at a time."""
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _DummyTokenizer:
    """Deterministic *word-level* tokenizer for offline adapter tests.

    Word-level rather than byte-level on purpose: IOI scoring requires answer names to
    be single tokens, so a byte-level stand-in would make every example unscoreable and
    the test would exercise nothing. Ids are a stable hash of the token, so the same
    text always yields the same ids across processes.
    """

    _PATTERN = r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]"

    def __init__(self, vocab_size: int):
        import re

        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self._re = re.compile(self._PATTERN)

    def _ids(self, text: str) -> List[int]:
        import hashlib

        out = []
        for tok in self._re.findall(text)[:56]:
            h = int(hashlib.sha1(tok.lower().encode()).hexdigest()[:8], 16)
            out.append(8 + h % (self.vocab_size - 8))
        return out or [8]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        seqs = [self._ids(t) for t in texts]
        if max_length is not None and truncation:
            seqs = [s[:max_length] for s in seqs]
        if return_tensors is None:
            return {"input_ids": seqs[0] if isinstance(text, str) else seqs}
        n = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.as_tensor(s)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}
