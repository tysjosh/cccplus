"""Six Tracr-style programs with labelled causal variables (Sec. 4.1).

"The controlled stage uses six InterpBench/Tracr programs with at least two
labeled causal variables."

Each program has the same skeleton, which is what makes exact ground truth
possible:

  input    BOS followed by K contiguous groups of ``group_len`` tokens
  variable v_k = F(tokens of group k), a ternary aggregate over that group
  output   y = G(v_0, v_1, v_2) in {0..8}, at the final position

Programs differ in the aggregate F (counting a marker, running max, distinct
count, bracket balance, arg-max position, induction-style copy). Because the
groups are disjoint, every variable is *independently controllable*: a matched
counterfactual changes exactly one variable and leaves the rest of the prompt
intact, which is the synthetic analogue of swapping two names in IOI.

G is non-additive,

  G(v0, v1, v2) = 3 * ((v0 + v2) mod 3) + ((v1 + 2*v2) mod 3),

so the three variables have *different prompt-level causal signatures* even
though their mean effects are comparable. That is precisely the configuration in
which a mean-effect test cannot distinguish them but a signature test can.
Clamping v2 to 0 yields the restricted variant G' = 3*v0 + v1, used to plant
'absent mechanism' destinations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

BOS = 0
N_VARS = 3
VAR_CARD = 3
GROUP_LEN = 4
N_OUT = 9
D_VOCAB = 12   # 0 = BOS, 1..11 usable symbols


# --------------------------------------------------------------- aggregates F
def _f_count_marker(g: Tuple[int, ...]) -> int:
    return min(sum(1 for t in g if t == 1), 2)


def _f_running_max(g: Tuple[int, ...]) -> int:
    return min((max(g) - 1) // 2, 2)


def _f_count_high(g: Tuple[int, ...]) -> int:
    """Bucketed count of high-valued tokens (a two-bin histogram query).

    Chosen over a distinct-token count because 'all tokens equal' is realised by
    only a handful of tuples, starving one variable value of training diversity
    (pools of 4 vs 256 here), and over a max-multiplicity count because that
    requires all-pairs comparison within the group and is not reliably learnable at
    this model scale. Counting a token *class* keeps the histogram character while
    staying distinct from ``frac_prevs``, which counts one specific token.
    """
    return min(sum(1 for t in g if t >= 5), 2)


def _f_balance(g: Tuple[int, ...]) -> int:
    # 1 = '(', 2 = ')', 3 = filler
    bal = sum(1 for t in g if t == 1) - sum(1 for t in g if t == 2)
    return 0 if bal < 0 else (1 if bal == 0 else 2)


def _f_argmax_pos(g: Tuple[int, ...]) -> int:
    return min(int(np.argmax(np.asarray(g))), 2)


def _f_induction(g: Tuple[int, ...]) -> int:
    # value copied from the token that follows the first marker (token id 1)
    for i, t in enumerate(g):
        if t == 1 and i + 1 < len(g):
            return (g[i + 1] - 1) % 3
    return 2


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    alphabet: Tuple[int, ...]
    feature: Callable[[Tuple[int, ...]], int]
    description: str


SPECS: Tuple[ProgramSpec, ...] = (
    ProgramSpec("frac_prevs", (1, 2, 3, 4), _f_count_marker,
                "bucketed count of the marker token in each group"),
    ProgramSpec("running_max", (1, 2, 3, 4, 5, 6), _f_running_max,
                "bucketed maximum token value in each group"),
    ProgramSpec("token_hist", (1, 2, 3, 4, 5, 6), _f_count_high,
                "bucketed count of high-valued tokens in each group"),
    ProgramSpec("dyck1_balance", (1, 2, 3), _f_balance,
                "sign of the bracket balance of each group"),
    ProgramSpec("sort_key", (1, 2, 3, 4, 5, 6), _f_argmax_pos,
                "position of the maximum token within each group"),
    ProgramSpec("induction_copy", (1, 2, 3, 4, 5), _f_induction,
                "token following the first marker in each group"),
)


def program_output(v: np.ndarray, absent: Optional[Set[int]] = None) -> np.ndarray:
    """G(v0, v1, v2); variables in ``absent`` are clamped to 0."""
    v = np.asarray(v)
    v0, v1, v2 = v[:, 0].copy(), v[:, 1].copy(), v[:, 2].copy()
    if absent:
        if 0 in absent:
            v0[:] = 0
        if 1 in absent:
            v1[:] = 0
        if 2 in absent:
            v2[:] = 0
    return (3 * ((v0 + v2) % 3) + ((v1 + 2 * v2) % 3)).astype(np.int64)


class Program:
    """A program plus balanced pools of group tokens for each variable value."""

    def __init__(self, spec: ProgramSpec, group_len: int = GROUP_LEN, n_vars: int = N_VARS):
        self.spec = spec
        self.name = spec.name
        self.group_len = group_len
        self.n_vars = n_vars
        self.seq_len = 1 + n_vars * group_len
        self.d_vocab = D_VOCAB
        self.n_out = N_OUT
        self.pools: Dict[int, np.ndarray] = {}
        for combo in itertools.product(spec.alphabet, repeat=group_len):
            val = int(spec.feature(combo))
            self.pools.setdefault(val, []).append(combo)
        self.pools = {k: np.asarray(v, dtype=np.int64) for k, v in sorted(self.pools.items())}
        missing = [v for v in range(VAR_CARD) if v not in self.pools or len(self.pools[v]) == 0]
        if missing:
            raise ValueError(f"program {spec.name}: no group tokens realise values {missing}")

    # ------------------------------------------------------------- positions
    def group_slice(self, k: int) -> slice:
        start = 1 + k * self.group_len
        return slice(start, start + self.group_len)

    @property
    def readout_position(self) -> int:
        return self.seq_len - 1

    # --------------------------------------------------------------- sampling
    def tokens_for_values(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Build prompts realising the given [B, K] variable values."""
        B = values.shape[0]
        toks = np.zeros((B, self.seq_len), dtype=np.int64)
        toks[:, 0] = BOS
        for k in range(self.n_vars):
            for val in range(VAR_CARD):
                m = values[:, k] == val
                if not m.any():
                    continue
                pool = self.pools[val]
                pick = rng.integers(0, len(pool), size=int(m.sum()))
                toks[np.ix_(np.flatnonzero(m), np.arange(self.group_slice(k).start, self.group_slice(k).stop))] = pool[pick]
        return toks

    def sample_values(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, VAR_CARD, size=(n, self.n_vars))

    def variable_values(self, tokens: np.ndarray) -> np.ndarray:
        """Recover [B, K] variable values from prompts (used for assertions)."""
        tokens = np.asarray(tokens)
        out = np.zeros((tokens.shape[0], self.n_vars), dtype=np.int64)
        for k in range(self.n_vars):
            grp = tokens[:, self.group_slice(k)]
            out[:, k] = [self.spec.feature(tuple(int(t) for t in row)) for row in grp]
        return out

    def counterfactual_values(self, values: np.ndarray, var: int, rng: np.random.Generator) -> np.ndarray:
        """Change only ``var``, uniformly among its other values."""
        out = values.copy()
        shift = rng.integers(1, VAR_CARD, size=values.shape[0])
        out[:, var] = (values[:, var] + shift) % VAR_CARD
        return out

    def make_pair(
        self, n: int, var: int, rng: np.random.Generator, absent: Optional[Set[int]] = None
    ) -> Dict[str, np.ndarray]:
        """Matched (x, x') pairs differing only in variable ``var``.

        ``correct`` is y(x) and ``distractor`` is y(x'): under a faithful patch of
        the mechanism carrying ``var``, the model should move from the former to the
        latter, so the preregistered effect sign is negative.
        """
        v = self.sample_values(n, rng)
        vp = self.counterfactual_values(v, var, rng)
        x = self.tokens_for_values(v, rng)
        # keep every other group byte-identical: rebuild x' from x, swapping group `var`
        xp = x.copy()
        sl = self.group_slice(var)
        target = vp[:, var]
        for val in range(VAR_CARD):
            m = target == val
            if not m.any():
                continue
            pool = self.pools[val]
            pick = rng.integers(0, len(pool), size=int(m.sum()))
            xp[np.ix_(np.flatnonzero(m), np.arange(sl.start, sl.stop))] = pool[pick]
        return {
            "clean_tokens": x,
            "cf_tokens": xp,
            "clean_values": v,
            "cf_values": vp,
            "correct": program_output(v, absent),
            "distractor": program_output(vp, absent),
            "var": np.full(n, var, dtype=np.int64),
        }

    def fill_group(
        self, tokens: np.ndarray, k: int, values: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """In-place rewrite of group ``k`` so that variable ``k`` takes ``values``."""
        sl = self.group_slice(k)
        cols = np.arange(sl.start, sl.stop)
        for val in range(VAR_CARD):
            m = values == val
            if not m.any():
                continue
            pool = self.pools[val]
            pick = rng.integers(0, len(pool), size=int(m.sum()))
            tokens[np.ix_(np.flatnonzero(m), cols)] = pool[pick]
        return tokens

    def interchange_batch(
        self,
        n: int,
        variables: Sequence[int],
        rng: np.random.Generator,
        absent: Optional[Set[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Clean / source prompt pairs differing in exactly ``variables``.

        Returns the clean output and the interchange output, i.e. the label the
        model must produce when the listed variables are imported from the source
        prompt. This is the supervision signal for IIT training and the definition
        of the 'distractor' answer at evaluation time.
        """
        v = self.sample_values(n, rng)
        vp = v.copy()
        for k in variables:
            shift = rng.integers(1, VAR_CARD, size=n)
            vp[:, k] = (v[:, k] + shift) % VAR_CARD
        x = self.tokens_for_values(v, rng)
        xp = x.copy()
        for k in variables:
            self.fill_group(xp, k, vp[:, k], rng)
        return {
            "clean_tokens": x,
            "cf_tokens": xp,
            "clean_values": v,
            "cf_values": vp,
            "correct": program_output(v, absent),
            "distractor": program_output(vp, absent),
        }

    def sample_plain(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.tokens_for_values(self.sample_values(n, rng), rng)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Program {self.name} K={self.n_vars} L={self.seq_len}>"


PROGRAMS: Dict[str, Program] = {}


def get_program(name: str) -> Program:
    if name not in PROGRAMS:
        spec = next((s for s in SPECS if s.name == name), None)
        if spec is None:
            raise KeyError(f"unknown program '{name}'; available: {[s.name for s in SPECS]}")
        PROGRAMS[name] = Program(spec)
    return PROGRAMS[name]


def all_program_names() -> List[str]:
    return [s.name for s in SPECS]
