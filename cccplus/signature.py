"""Standardised causal signatures and the CCC+ distances (Sec. 3.1, Eqs. 1-5, 13-14).

Eq. 2   mean effect            Delta_bar(M, m) = mean_x Delta_T(M, m, a_full; x)
Eq. 3   signature              s(M, m) = { Delta_T(M, m, a; x) / c_{M,T} }_{(a,x) in A x D}
Eq. 4   shape distance         || s/max(|s|_w, eta) - t/max(|t|_w, eta) ||_w
Eq. 5   magnitude distance     | log( (|t|_w + eta) / (|s|_w + eta) ) |
Eq. 13  normalised path error  e = max{ d_shape/eps_shape , d_mag/eps_mag }
Eq. 14  score                  CCC+_j = exp(-(e^F_j + e^R_j)/2),
                               CCC+_multi = sqrt(CCC+_2 * CCC+_3)

The shape term keeps prompt- and setting-level heterogeneity; the magnitude term
stops "a weak and an overwhelmingly strong effect from being declared equivalent
merely because their signs agree".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .interventions import PRIMARY, CausalProbe, Setting
from .mechanisms import Mechanism

# Eq. 5 is written with a bare log in the paper but is used as a distance
# (d_mag <= eps_mag), so the absolute value is the intended reading.


@dataclass
class CausalSignature:
    """Per-(setting, prompt) causal effects with the response scale c_{M,T}."""

    setting_keys: List[str]
    effects: torch.Tensor                     # [n_settings, n_prompts], raw effects
    scale: float = 1.0                        # c_{M,T}
    model: str = ""
    mechanism: Optional[str] = None
    prompt_key: str = ""                      # identity of the (ordered) prompt set
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------ constructors
    @classmethod
    def measure(
        cls,
        probe: CausalProbe,
        mech: Mechanism,
        settings: Sequence[Setting],
        scale: float = 1.0,
    ) -> "CausalSignature":
        rows = [probe.effect(mech, s) for s in settings]
        return cls(
            setting_keys=[s.key for s in settings],
            # Signatures are analysis artifacts: small, compared, cached, hashed and
            # serialised. Pinning them to the CPU at construction keeps every downstream
            # consumer on one device, rather than inheriting whichever device the model
            # happened to run on.
            effects=torch.stack(
                [r.detach().to(device="cpu", dtype=torch.float64) for r in rows], dim=0
            ),
            scale=float(scale),
            model=probe.model.name,
            mechanism=mech.key(),
            prompt_key=prompt_key_of(probe.data),
        )

    # ------------------------------------------------------------- properties
    @property
    def n_settings(self) -> int:
        return int(self.effects.shape[0])

    @property
    def n_prompts(self) -> int:
        return int(self.effects.shape[1])

    def vector(self) -> torch.Tensor:
        """Eq. 3: flattened standardised signature, ordered (setting, prompt)."""
        return (self.effects / max(self.scale, 1e-12)).reshape(-1)

    def mean_effect(self, setting: Optional[str] = None) -> float:
        """Eq. 2 at the full counterfactual-patch setting by default."""
        key = setting or self._full_key()
        i = self.setting_keys.index(key)
        return float(self.effects[i].mean())

    def per_prompt(self, setting: Optional[str] = None) -> torch.Tensor:
        key = setting or self._full_key()
        return self.effects[self.setting_keys.index(key)]

    def _full_key(self) -> str:
        for k in self.setting_keys:
            if k.startswith(PRIMARY) and k.endswith("@1"):
                return k
        return self.setting_keys[0]

    def rescaled(self, scale: float) -> "CausalSignature":
        return CausalSignature(
            list(self.setting_keys), self.effects.clone(), float(scale), self.model,
            self.mechanism, self.prompt_key, dict(self.meta),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "mechanism": self.mechanism,
            "settings": self.setting_keys,
            "scale": self.scale,
            "mean_effect": self.mean_effect(),
            "n_prompts": self.n_prompts,
        }


# ------------------------------------------------------------------- weights
def signature_weights(
    n_settings: int, n_prompts: int, scheme: str = "setting_balanced"
) -> torch.Tensor:
    """Non-negative weights over signature coordinates summing to one."""
    if scheme == "uniform":
        w = torch.ones(n_settings * n_prompts, dtype=torch.float64)
    elif scheme == "setting_balanced":
        per = torch.full((n_settings, n_prompts), 1.0 / (n_settings * n_prompts), dtype=torch.float64)
        w = per.reshape(-1)
    else:
        raise ValueError(f"unknown weight scheme: {scheme}")
    return w / w.sum()


def weighted_norm(z: torch.Tensor, w: torch.Tensor) -> float:
    """||z||_w = sqrt( sum_q w_q z_q^2 ).

    Signatures are CPU artifacts (see ``CausalSignature.measure``) while weights are
    built on the CPU too, so this is normally a no-op. The coercion is kept because the
    effects that feed a signature originate on whatever device the model ran on, and a
    device mismatch here is a crash rather than a wrong number.
    """
    z = z.to(device="cpu", dtype=torch.float64)
    w = w.to(device="cpu", dtype=torch.float64)
    return float(torch.sqrt(torch.clamp((w * z * z).sum(), min=0.0)))


# ----------------------------------------------------------------- distances
def d_shape(s: torch.Tensor, t: torch.Tensor, w: torch.Tensor, eta: float) -> float:
    """Eq. 4: weighted distance between direction-normalised signatures."""
    s = s.to(device="cpu", dtype=torch.float64)
    t = t.to(device="cpu", dtype=torch.float64)
    if s.shape != t.shape:
        raise ValueError(f"signature shapes differ: {tuple(s.shape)} vs {tuple(t.shape)}")
    sn = max(weighted_norm(s, w), eta)
    tn = max(weighted_norm(t, w), eta)
    return weighted_norm(s / sn - t / tn, w)


def d_mag(s: torch.Tensor, t: torch.Tensor, w: torch.Tensor, eta: float) -> float:
    """Eq. 5: absolute log-ratio of weighted signature norms."""
    sn = weighted_norm(s.to(torch.float64), w)
    tn = weighted_norm(t.to(torch.float64), w)
    return float(abs(np.log((tn + eta) / (sn + eta))))


@dataclass
class SignatureComparison:
    """Both distances plus their threshold-normalised version (Eq. 13)."""

    shape: float
    magnitude: float
    eps_shape: float
    eps_mag: float

    @property
    def normalised_error(self) -> float:
        return max(self.shape / max(self.eps_shape, 1e-12), self.magnitude / max(self.eps_mag, 1e-12))

    @property
    def passes(self) -> bool:
        return (self.shape <= self.eps_shape) and (self.magnitude <= self.eps_mag)

    def to_dict(self) -> Dict[str, float]:
        return {
            "d_shape": self.shape,
            "d_mag": self.magnitude,
            "eps_shape": self.eps_shape,
            "eps_mag": self.eps_mag,
            "normalised_error": self.normalised_error,
            "passes": bool(self.passes),
        }


def prompt_key_of(data) -> str:
    """Identity of a prompt set: its ordered family labels and counterfactual pairing.

    Deliberately *not* based on token ids: across architectures the same prompts
    tokenise differently, and d_shape compares coordinate-for-coordinate, so what
    must match is the prompt ordering and pairing, not the tokenisation.
    """
    import hashlib

    import numpy as _np

    fam = _np.asarray(getattr(data, "family", _np.zeros(0)))
    var = getattr(data, "meta", {}).get("var", "")
    payload = f"{var}|{len(fam)}|" + ",".join(str(x) for x in fam.tolist())
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def assert_comparable(s: CausalSignature, t: CausalSignature) -> None:
    """Refuse to compare signatures whose coordinates do not correspond.

    Eqs. 4-5 are coordinatewise: a silent mismatch in setting order, prompt order or
    counterfactual pairing would produce a well-formed but meaningless distance. A
    shape check alone does not catch that, so all three are verified.
    """
    if list(s.setting_keys) != list(t.setting_keys):
        raise ValueError(
            f"signature setting families differ: {s.setting_keys} vs {t.setting_keys}"
        )
    if s.effects.shape != t.effects.shape:
        raise ValueError(f"signature shapes differ: {tuple(s.effects.shape)} vs {tuple(t.effects.shape)}")
    if s.prompt_key and t.prompt_key and s.prompt_key != t.prompt_key:
        raise ValueError(
            "signatures were measured on different prompt sets or pairings "
            f"({s.prompt_key} vs {t.prompt_key}); d_shape would be meaningless"
        )


def compare_signatures(
    s: CausalSignature,
    t: CausalSignature,
    eps_shape: float,
    eps_mag: float,
    eta: float,
    weight_scheme: str = "setting_balanced",
) -> SignatureComparison:
    assert_comparable(s, t)
    sv, tv = s.vector(), t.vector()
    w = signature_weights(s.n_settings, s.n_prompts, weight_scheme)
    return SignatureComparison(
        shape=d_shape(sv, tv, w, eta),
        magnitude=d_mag(sv, tv, w, eta),
        eps_shape=eps_shape,
        eps_mag=eps_mag,
    )


# -------------------------------------------------------------------- scores
def ccc_plus_score(e_forward: float, e_return: float) -> float:
    """Eq. 14, pairwise."""
    return float(np.exp(-(e_forward + e_return) / 2.0))


def ccc_plus_multi(score_2: float, score_3: float) -> float:
    """Eq. 14, geometric mean over the two destinations."""
    return float(np.sqrt(max(score_2, 0.0) * max(score_3, 0.0)))


def scalar_return_error(mean_effect_returned: float, mean_effect_source: float) -> float:
    """Eq. 12: |Delta_bar(M1, returned) - Delta_bar(M1, source)|."""
    return float(abs(mean_effect_returned - mean_effect_source))


# --------------------------------------------------- prompt-level diagnostics
def signature_diagnostics(s: CausalSignature, t: CausalSignature) -> Dict[str, float]:
    """Reported in Sec. 5.4: effect correlation, sign agreement, effect ratio,
    worst-template error, and calibration slope/intercept (Sec. 3.6)."""
    a = s.vector().numpy()
    b = t.vector().numpy()
    out: Dict[str, float] = {}
    if a.std() > 1e-12 and b.std() > 1e-12:
        out["effect_correlation"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out["effect_correlation"] = float("nan")
    out["sign_agreement"] = float(np.mean(np.sign(a) == np.sign(b)))
    denom = np.abs(a).mean()
    out["effect_ratio"] = float(np.abs(b).mean() / denom) if denom > 1e-12 else float("nan")
    # Calibration of destination effects against source effects.
    if a.std() > 1e-12:
        A = np.stack([a, np.ones_like(a)], axis=1)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        out["calibration_slope"] = float(coef[0])
        out["calibration_intercept"] = float(coef[1])
    else:
        out["calibration_slope"] = float("nan")
        out["calibration_intercept"] = float("nan")
    # Worst per-setting (template-family-level when families are the prompt groups).
    se = s.effects.numpy()
    te = t.effects.numpy()
    scale = max(np.abs(se).mean(), 1e-12)
    per_setting = np.abs(se - te).mean(axis=1) / scale
    out["worst_setting_error"] = float(per_setting.max())
    return out


def worst_family_error(
    s: CausalSignature, t: CausalSignature, family: np.ndarray
) -> float:
    """Worst-template error (Sec 3.6): largest per-family normalised discrepancy."""
    se = s.effects.numpy()
    te = t.effects.numpy()
    scale = max(np.abs(se).mean(), 1e-12)
    worst = 0.0
    for f in sorted(set(family.tolist())):
        m = family == f
        if not m.any():
            continue
        worst = max(worst, float(np.abs(se[:, m] - te[:, m]).mean() / scale))
    return worst
