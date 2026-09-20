"""Tasks and matched counterfactual data (Sec. 3.1-3.2).

A task supplies:

  * a scalar behavioural score ``f_T(y)``  -- correct-minus-distractor logit
    difference for IOI-like tasks, a length-normalised sequence log-probability
    difference for factual recall;
  * clean prompts with *matched* counterfactuals that "change the task-relevant
    variable while preserving the template";
  * a family label per example, so partitions never split a template family
    (Sec. 4.3);
  * the preregistered sign the causal effect must take (Sec. 3.1 source gate).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..models.base import PromptBatch


@dataclass
class TaskData:
    """Clean / counterfactual prompt pairs and their answer tokens."""

    clean: PromptBatch
    counterfactual: PromptBatch
    correct: torch.Tensor                 # [B] token id scored as "correct"
    distractor: torch.Tensor              # [B] relation- or role-matched distractor
    family: np.ndarray                    # [B] group label for partitioning
    unrelated: Optional[torch.Tensor] = None      # [B, k] control tokens
    cross_task: Optional["TaskData"] = None       # for the cross-task control
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.clean)

    def subset(self, idx: Sequence[int]) -> "TaskData":
        sel = torch.as_tensor(list(idx), dtype=torch.long)
        return TaskData(
            clean=self.clean.subset(idx),
            counterfactual=self.counterfactual.subset(idx),
            correct=self.correct[sel],
            distractor=self.distractor[sel],
            family=self.family[np.asarray(list(idx))],
            unrelated=None if self.unrelated is None else self.unrelated[sel],
            cross_task=self.cross_task,
            meta=dict(self.meta),
        )


class Task(ABC):
    """Behavioural score plus the preregistered effect-sign prediction."""

    name: str = "task"
    predicted_sign: float = -1.0   # patching the mechanism should *reduce* the score
    #: Set by tasks whose score is not a function of one readout position's logits.
    #: Factual recall needs this: its score is a length-normalised *sequence*
    #: log-probability difference, so it must run the model itself over two
    #: continuations rather than read a single logit vector.
    uses_full_scoring: bool = False

    @abstractmethod
    def score(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:
        """Scalar behavioural score per example, [B]."""

    def score_full(self, model, data: TaskData, patches=()) -> torch.Tensor:
        """Behavioural score computed by running ``model`` directly, [B].

        Only called when ``uses_full_scoring`` is True.
        """
        raise NotImplementedError

    def unrelated_score(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:
        """Mean absolute change basis for the unrelated-logit control (Sec 3.2)."""
        if data.unrelated is None:
            return torch.zeros(logits.shape[0], device=logits.device)
        idx = data.unrelated.to(logits.device)
        gathered = torch.gather(logits, 1, idx)
        return gathered.mean(dim=1)

    def accuracy(self, logits: torch.Tensor, data: TaskData) -> float:
        """Fraction of prompts where the correct answer outranks the distractor."""
        return float((self.score(logits, data) > 0).float().mean())


class LogitDiffTask(Task):
    """f_T = logit[correct] - logit[distractor] (IOI and the synthetic programs)."""

    def __init__(self, name: str = "logit_diff", predicted_sign: float = -1.0):
        self.name = name
        self.predicted_sign = predicted_sign

    def score(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:
        logits = logits.to(torch.float32)
        c = data.correct.to(logits.device)
        d = data.distractor.to(logits.device)
        return logits.gather(1, c[:, None]).squeeze(1) - logits.gather(1, d[:, None]).squeeze(1)


def partition_by_family(
    family: np.ndarray,
    fractions: Dict[str, float],
    seed: int,
) -> Dict[str, np.ndarray]:
    """Group-disjoint split (Sec 4.3: 20/20/20/40 grouped by family).

    Families are shuffled once with a fixed seed and then assigned to partitions in
    order, so no family appears in two partitions.
    """
    rng = np.random.default_rng(seed)
    fams = np.array(sorted(set(family.tolist())))
    rng.shuffle(fams)
    names = list(fractions.keys())
    weights = np.array([fractions[n] for n in names], dtype=float)
    weights = weights / weights.sum()
    # Deterministic allocation of whole families to partitions.
    cuts = (np.cumsum(weights) * len(fams)).round().astype(int)
    cuts[-1] = len(fams)
    out: Dict[str, np.ndarray] = {}
    start = 0
    for name, end in zip(names, cuts):
        chosen = set(fams[start:end].tolist())
        out[name] = np.array([i for i, f in enumerate(family.tolist()) if f in chosen], dtype=int)
        start = end
    return out
