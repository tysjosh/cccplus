"""Frozen-manifest loading and reproducibility helpers (Sec. 4)."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "manifest" / "manifest.yaml"


class FrozenManifest:
    """Read-only view over the manifest, with a content hash for provenance.

    The hash is recorded next to every result file so that a table can always be
    traced back to the exact configuration that produced it.
    """

    def __init__(self, data: Dict[str, Any], raw: str, path: Optional[Path] = None):
        self._data = data
        self._raw = raw
        self.path = path

    # -------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> "FrozenManifest":
        p = Path(path) if path is not None else DEFAULT_MANIFEST
        raw = p.read_text()
        return cls(yaml.safe_load(raw), raw, p)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self._raw.encode()).hexdigest()[:16]

    # -------------------------------------------------------------- access
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def path_of(self, dotted: str, default: Any = "__raise__") -> Any:
        """Fetch a nested value, e.g. ``manifest.path_of('signature.eta')``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__raise__":
                    raise KeyError(f"manifest key not found: {dotted}")
                return default
            node = node[part]
        return node

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))  # deep copy, plain types

    # -------------------------------------------------------------- outputs
    def out_dir(self, which: str = "results_dir") -> Path:
        d = REPO_ROOT / self.path_of(f"outputs.{which}")
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass(frozen=True)
class RunContext:
    """Per-run bookkeeping: manifest, device, and the active random seed."""

    manifest: FrozenManifest
    device: torch.device
    seed: int

    @classmethod
    def create(
        cls,
        manifest: Optional[FrozenManifest] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "RunContext":
        mf = manifest or FrozenManifest.load()
        dev = torch.device(device or pick_device())
        sd = seed if seed is not None else int(mf.path_of("seeds.master"))
        set_all_seeds(sd)
        return cls(mf, dev, sd)


def pick_device() -> str:
    """CPU by default.

    The models used here are small; MPS gives no reliable speedup for the many
    tiny hooked forward passes that interventions require, and float
    reproducibility on CPU is easier to guarantee across runs.
    """
    if os.environ.get("CCC_DEVICE"):
        return os.environ["CCC_DEVICE"]
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - not used on this host
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    """Deterministic seed derived from arbitrary labels.

    Used so that e.g. the crosscoder for (program, M1, M2, seed 3) always gets the
    same initialisation regardless of iteration order elsewhere.
    """
    key = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def write_json(path: os.PathLike, payload: Any, manifest: Optional[FrozenManifest] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body: Any = payload
    if isinstance(payload, dict) and manifest is not None:
        body = dict(payload)
        body["_manifest_hash"] = manifest.hash
    p.write_text(json.dumps(body, indent=2, default=_json_default))
    return p


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
