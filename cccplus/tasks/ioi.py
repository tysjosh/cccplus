"""Indirect-object identification (Sec. 4.2).

"IOI uses 600 screening pairs and 2,400 analysis pairs from 12 template families
with split-disjoint names; a stratum requires at least 90% clean accuracy in every
model."

Behavioural score: the correct-minus-distractor logit difference, i.e.
logit[IO] - logit[S] at the position that should emit the indirect object.

Matched counterfactual: the two names exchange their grammatical roles while the
template is preserved, so the *repeated* name becomes the indirect object and the
answer flips. The preregistered effect sign is therefore negative -- a faithful
patch of a name-mover/S-inhibition mechanism moves probability from IO to S.

Splits are disjoint in **names**, not templates: with only twelve templates,
partitioning by template would leave too few per split, and name leakage is the
contamination that actually matters for this task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..models.base import PromptBatch
from .base import LogitDiffTask, TaskData

# Twelve template families, in the style of Wang et al. (2023). Each ends immediately
# before the slot where the indirect object must be produced.
TEMPLATES: Tuple[str, ...] = (
    "Then, {a} and {b} went to the {place}. {s} gave a {obj} to",
    "Then, {a} and {b} had a lot of fun at the {place}. {s} gave a {obj} to",
    "Then, {a} and {b} were working at the {place}. {s} decided to give a {obj} to",
    "Then, {a} and {b} were thinking about going to the {place}. {s} wanted to give a {obj} to",
    "Then, {a} and {b} had a long argument, and afterwards {s} said to",
    "After {a} and {b} went to the {place}, {s} gave a {obj} to",
    "When {a} and {b} got a {obj} at the {place}, {s} decided to give it to",
    "While {a} and {b} were commuting to the {place}, {s} gave a {obj} to",
    "The {place} {a} and {b} went to had a {obj}. {s} gave it to",
    "Friends {a} and {b} found a {obj} at the {place}. {s} gave it to",
    "{a} and {b} are working at the {place}. {s} decided to give a {obj} to",
    "{a} and {b} went to the {place}. Because {s} was tired, {s} gave the {obj} to",
)

NAMES: Tuple[str, ...] = (
    "John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Anna", "Emily",
    "Kate", "Rose", "Alex", "Paul", "Sarah", "Peter", "Laura", "Mark", "Jessica",
    "Robert", "Julia", "Michael", "Helen", "David", "Nancy", "Brian", "Rachel",
    "Kevin", "Linda", "Steven", "Diana", "Jacob", "Clara", "Oliver", "Grace", "Henry",
)
PLACES: Tuple[str, ...] = ("store", "garden", "restaurant", "school", "hospital",
                           "office", "station", "museum", "market", "library")
OBJECTS: Tuple[str, ...] = ("ring", "kiss", "bone", "basketball", "computer", "necklace",
                            "drink", "snack", "book", "flower")


@dataclass
class IOIExample:
    text: str
    cf_text: str
    io: str                 # correct answer: the indirect object
    s: str                  # distractor: the repeated subject
    template: int
    name_group: int
    order: str              # 'ABBA' or 'BABA'


def build_ioi_examples(
    n: int,
    seed: int,
    names: Sequence[str] = NAMES,
    templates: Sequence[str] = TEMPLATES,
    n_name_groups: int = 18,
) -> List[IOIExample]:
    """Generate matched IOI pairs with a name-group label for partitioning."""
    rng = np.random.default_rng(seed)
    pool = list(names)
    rng.shuffle(pool)
    # Disjoint name groups; a group is the partitioning unit, so no name can appear
    # in two splits.
    groups: List[List[str]] = []
    per = max(2, len(pool) // n_name_groups)
    for i in range(0, len(pool) - 1, per):
        g = pool[i : i + per]
        if len(g) >= 2:
            groups.append(g)
    out: List[IOIExample] = []
    for i in range(n):
        gi = int(rng.integers(0, len(groups)))
        g = groups[gi]
        a, b = rng.choice(len(g), size=2, replace=False)
        n1, n2 = g[a], g[b]
        t = int(rng.integers(0, len(templates)))
        place = PLACES[int(rng.integers(0, len(PLACES)))]
        obj = OBJECTS[int(rng.integers(0, len(OBJECTS)))]
        order = "ABBA" if rng.random() < 0.5 else "BABA"
        # clean: n2 is the repeated subject, so the answer is n1
        first, second = (n1, n2) if order == "ABBA" else (n2, n1)
        tpl = templates[t]
        text = tpl.format(a=first, b=second, s=n2, place=place, obj=obj)
        # counterfactual: roles exchanged, template identical -> answer flips to n2
        cf_text = tpl.format(a=first, b=second, s=n1, place=place, obj=obj)
        out.append(IOIExample(text, cf_text, io=n1, s=n2, template=t, name_group=gi, order=order))
    return out


class IOITask(LogitDiffTask):
    """f_T = logit[IO] - logit[S]."""

    def __init__(self):
        super().__init__(name="ioi_logit_diff", predicted_sign=-1.0)


def ioi_task_data(model, examples: Sequence[IOIExample]) -> Optional[TaskData]:
    """Tokenise for one model and attach its own answer-token ids.

    Examples whose names are not single tokens for this tokenizer are dropped, since
    a logit difference over multi-token names is not well defined. The caller is
    responsible for intersecting the surviving indices across models so that every
    model is scored on the same prompts.
    """
    keep = [i for i, e in enumerate(examples)
            if model.is_single_token(e.io) and model.is_single_token(e.s)]
    if not keep:
        return None
    ex = [examples[i] for i in keep]
    clean = model.encode([e.text for e in ex])
    cf = model.encode([e.cf_text for e in ex])
    correct = torch.as_tensor([model.token_id_of(e.io) for e in ex], dtype=torch.long)
    distractor = torch.as_tensor([model.token_id_of(e.s) for e in ex], dtype=torch.long)
    unrelated = torch.as_tensor(
        [[model.token_id_of(nm) for nm in _unrelated_names(e, 4)] for e in ex], dtype=torch.long
    )
    data = TaskData(
        clean=clean, counterfactual=cf, correct=correct, distractor=distractor,
        family=np.asarray([e.name_group for e in ex]),
        unrelated=unrelated,
        meta={"var": "ioi", "kept_indices": keep,
              "templates": [e.template for e in ex]},
    )
    return data


def _unrelated_names(e: IOIExample, k: int) -> List[str]:
    """Names absent from the prompt: the unrelated-logit control of Sec. 3.2."""
    out = [n for n in NAMES if n not in (e.io, e.s)][:k]
    return out


def single_token_mask(model, examples: Sequence[IOIExample]) -> np.ndarray:
    return np.asarray([model.is_single_token(e.io) and model.is_single_token(e.s)
                       for e in examples], dtype=bool)


def screen_ioi(model, examples: Sequence[IOIExample], task: IOITask) -> Dict[str, float]:
    """Clean accuracy gate: "at least 90% clean accuracy in every model"."""
    data = ioi_task_data(model, examples)
    if data is None:
        return {"clean_accuracy": 0.0, "n": 0}
    logits = model.logits(data.clean)
    return {"clean_accuracy": task.accuracy(logits, data), "n": len(data),
            "mean_logit_diff": float(task.score(logits, data).mean())}
