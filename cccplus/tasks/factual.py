"""Factual recall (Sec. 4.2).

"Factual recall uses 600 facts, two screening paraphrases, and four analysis
paraphrases per fact. Primary analysis uses only facts whose correct object
outranks a relation-matched distractor in every model."

Behavioural score: the **length-normalised sequence log-probability difference**
between the correct object and a relation-matched distractor,

    f_T = (1/|o|) sum_t log p(o_t | prefix, o_<t)
        - (1/|d|) sum_t log p(d_t | prefix, d_<t)

which needs two teacher-forced passes per prompt, so this task sets
``uses_full_scoring`` and computes its own score rather than reading one logit
vector.

Matched counterfactual: the subject is replaced by another subject *of the same
relation*, so the template and relation are preserved and only the queried entity
changes. The distractor is that counterfactual subject's object, which makes it
relation-matched by construction rather than by a similarity heuristic.

The mechanism site is the final subject token's MLP output, with V the normalised
mean clean-counterfactual difference (Sec. 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..models.base import PromptBatch
from .base import Task, TaskData

# Relation -> (paraphrase templates, facts). Two screening + four analysis templates
# per relation, as the paper specifies. Held inline so the stage runs without any
# download; `load_counterfact` can extend it with the standard ROME data.
RELATIONS: Dict[str, Dict[str, object]] = {
    "capital": {
        "templates": [
            "The capital of {s} is",                      # screening
            "{s}'s capital city is",                      # screening
            "The capital city of {s} is called",          # analysis
            "If you travel to {s}, the capital you land in is",
            "Geography question: the capital of {s} is",
            "The seat of government of {s} is located in",
        ],
        "facts": [
            ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Greece", "Athens"),
            ("Poland", "Warsaw"), ("Norway", "Oslo"), ("Portugal", "Lisbon"), ("Austria", "Vienna"),
            ("Ireland", "Dublin"), ("Cuba", "Havana"), ("Peru", "Lima"), ("Chile", "Santiago"),
            ("Hungary", "Budapest"), ("Thailand", "Bangkok"), ("Sweden", "Stockholm"),
            ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Turkey", "Ankara"),
            ("Iraq", "Baghdad"), ("Iran", "Tehran"), ("Morocco", "Rabat"), ("Ethiopia", "Addis"),
        ],
    },
    "language": {
        "templates": [
            "The official language of {s} is",
            "In {s}, people mainly speak",
            "The language most widely spoken in {s} is",
            "A visitor to {s} would need to speak",
            "Linguistics question: the dominant language of {s} is",
            "Most residents of {s} communicate in",
        ],
        "facts": [
            ("France", "French"), ("Japan", "Japanese"), ("Italy", "Italian"), ("Brazil", "Portuguese"),
            ("Germany", "German"), ("Russia", "Russian"), ("China", "Chinese"), ("Greece", "Greek"),
            ("Poland", "Polish"), ("Turkey", "Turkish"), ("Sweden", "Swedish"), ("Finland", "Finnish"),
            ("Israel", "Hebrew"), ("Egypt", "Arabic"), ("Korea", "Korean"), ("Thailand", "Thai"),
            ("Vietnam", "Vietnamese"), ("Hungary", "Hungarian"), ("Norway", "Norwegian"),
            ("Denmark", "Danish"), ("Romania", "Romanian"), ("Iran", "Persian"),
        ],
    },
    "continent": {
        "templates": [
            "{s} is located on the continent of",
            "The continent containing {s} is",
            "Geographically, {s} belongs to the continent of",
            "On a world map, {s} appears in",
            "Atlas entry: {s} is part of",
            "The landmass that contains {s} is",
        ],
        "facts": [
            ("Brazil", "America"), ("Nigeria", "Africa"), ("Japan", "Asia"), ("France", "Europe"),
            ("Kenya", "Africa"), ("India", "Asia"), ("Peru", "America"), ("Norway", "Europe"),
            ("Egypt", "Africa"), ("Vietnam", "Asia"), ("Chile", "America"), ("Poland", "Europe"),
            ("Ghana", "Africa"), ("Nepal", "Asia"), ("Bolivia", "America"), ("Portugal", "Europe"),
        ],
    },
    "instrument": {
        "templates": [
            "{s} is famous for playing the",
            "The instrument most associated with {s} is the",
            "Music history records that {s} performed on the",
            "Concert programmes list {s} as a player of the",
            "In recordings, {s} is heard on the",
            "{s} built a career around the",
        ],
        "facts": [
            ("Miles Davis", "trumpet"), ("Jimi Hendrix", "guitar"), ("Glenn Gould", "piano"),
            ("John Coltrane", "saxophone"), ("Yo-Yo Ma", "cello"), ("Ravi Shankar", "sitar"),
            ("Charlie Parker", "saxophone"), ("Itzhak Perlman", "violin"),
            ("Jaco Pastorius", "bass"), ("Buddy Rich", "drums"), ("Art Tatum", "piano"),
            ("Wynton Marsalis", "trumpet"),
        ],
    },
    "field": {
        "templates": [
            "{s} is best known as a",
            "By profession, {s} was a",
            "Encyclopaedias describe {s} as a",
            "The career of {s} was that of a",
            "In biographies, {s} is called a",
            "{s} spent a working life as a",
        ],
        "facts": [
            ("Marie Curie", "physicist"), ("Ada Lovelace", "mathematician"),
            ("Vincent van Gogh", "painter"), ("Ludwig van Beethoven", "composer"),
            ("Charles Darwin", "biologist"), ("Virginia Woolf", "novelist"),
            ("Alan Turing", "mathematician"), ("Frida Kahlo", "painter"),
            ("Igor Stravinsky", "composer"), ("Niels Bohr", "physicist"),
            ("Jane Austen", "novelist"), ("Rosalind Franklin", "chemist"),
        ],
    },
}


@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    cf_subject: str
    cf_obj: str          # relation-matched distractor
    template: str
    template_idx: int

    SLOT = "{s}"

    def prompt(self) -> str:
        return self._fill(self.subject)

    def cf_prompt(self) -> str:
        return self._fill(self.cf_subject)

    def _fill(self, subject: str) -> str:
        """Substitute the subject without interpreting the rest of the template.

        ``str.format`` is the wrong tool here: templates come from CounterFact, and any
        record containing a brace that is not the subject slot makes it raise (or, worse,
        interpret dataset text as a format field). Plain replacement treats the template
        as data, which is what it is.
        """
        if self.SLOT not in self.template:
            raise ValueError(
                f"template for ({self.subject!r}, {self.relation!r}) has no {self.SLOT} "
                f"slot: {self.template!r}"
            )
        return self.template.replace(self.SLOT, subject)


def build_facts(
    n: int, seed: int, analysis_only: bool = True, relations: Optional[Dict] = None
) -> List[Fact]:
    """Facts crossed with paraphrases; the distractor is another subject's object."""
    rel = relations or RELATIONS
    rng = np.random.default_rng(seed)
    out: List[Fact] = []
    for rname, spec in rel.items():
        templates: List[str] = list(spec["templates"])            # type: ignore
        tpl_idx = range(2, len(templates)) if analysis_only else range(len(templates))
        facts: List[Tuple[str, str]] = list(spec["facts"])        # type: ignore
        for ti in tpl_idx:
            for si, (s, o) in enumerate(facts):
                # relation-matched counterfactual subject with a *different* object
                alts = [(s2, o2) for (s2, o2) in facts if o2 != o and s2 != s]
                if not alts:
                    continue
                s2, o2 = alts[int(rng.integers(0, len(alts)))]
                out.append(Fact(s, rname, o, s2, o2, templates[ti], ti))
    rng.shuffle(out)
    return out[:n]


class FactualRecallTask(Task):
    """Length-normalised sequence log-probability difference."""

    name = "factual_recall_logprob_diff"
    predicted_sign = -1.0
    uses_full_scoring = True

    def score(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("factual recall scores sequences; use score_full")

    def score_full(self, model, data: TaskData, patches=()) -> torch.Tensor:
        cont = data.meta["continuations"]           # type: ignore[index]
        good = _sequence_logprob(model, cont["correct"], patches)
        bad = _sequence_logprob(model, cont["distractor"], patches)
        return good - bad

    def unrelated_score(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:
        if data.unrelated is None:
            return torch.zeros(logits.shape[0], device=logits.device)
        idx = data.unrelated.to(logits.device)
        return torch.gather(logits, 1, idx).mean(dim=1)

    def accuracy(self, logits: torch.Tensor, data: TaskData) -> float:  # pragma: no cover
        raise NotImplementedError("use accuracy_full")

    def accuracy_full(self, model, data: TaskData) -> float:
        return float((self.score_full(model, data) > 0).float().mean())


@dataclass
class ContinuationBatch:
    """Prompt+continuation tokens with the span to be scored."""

    batch: PromptBatch
    start: torch.Tensor          # [B] first continuation index
    length: torch.Tensor         # [B] number of continuation tokens


def _sequence_logprob(model, cb: ContinuationBatch, patches=()) -> torch.Tensor:
    """Mean log p of the continuation tokens, teacher forced, under ``patches``."""
    # Reduced per chunk rather than from one [B, T, V] tensor: this score is the reason
    # the natural stage is slow, and on a 262k-vocab model the assembled tensor does not
    # fit in GPU memory at all. Only the gathered target logprobs survive the loop.
    out = []
    for rows, chunk in model.iter_full_logits(cb.batch, patches):
        logprobs = torch.log_softmax(chunk.to(torch.float32), dim=-1)
        tokens = cb.batch.tokens[rows].to(logprobs.device)
        T = logprobs.shape[1]
        ar = torch.arange(T, device=logprobs.device)[None, :]
        start = cb.start[rows].to(logprobs.device)[:, None]
        length = cb.length[rows].to(logprobs.device)[:, None]
        mask = (ar >= start) & (ar < start + length)          # continuation positions
        # log p(token at t) is read from the distribution at t-1
        tgt = tokens.clamp(min=0)
        lp = logprobs[:, :-1].gather(2, tgt[:, 1:, None]).squeeze(2)   # [chunk, T-1]
        m = mask[:, 1:].to(lp.dtype)
        denom = m.sum(dim=1).clamp(min=1.0)
        out.append(((lp * m).sum(dim=1) / denom).detach())     # length-normalised
    return torch.cat(out, dim=0)


def _continuation_batch(model, prompts: Sequence[str], objects: Sequence[str]) -> ContinuationBatch:
    texts, starts, lengths = [], [], []
    for p, o in zip(prompts, objects):
        cont = " " + o.strip()
        n_prompt = len(model.tokenizer(p, add_special_tokens=True)["input_ids"])
        n_cont = len(model.tokenizer(cont, add_special_tokens=False)["input_ids"])
        texts.append(p + cont)
        starts.append(n_prompt)
        lengths.append(max(n_cont, 1))
    batch = model.encode(texts)
    return ContinuationBatch(batch, torch.as_tensor(starts, dtype=torch.long),
                             torch.as_tensor(lengths, dtype=torch.long))


def factual_task_data(model, facts: Sequence[Fact]) -> TaskData:
    """Build clean/counterfactual prompts plus the two scored continuations.

    ``clean`` and ``counterfactual`` hold the *prompts* (this is where Eq. 6 patches
    are applied, at the final subject token). The continuations used for scoring live
    in ``meta`` and share those prompts.
    """
    prompts = [f.prompt() for f in facts]
    cf_prompts = [f.cf_prompt() for f in facts]
    clean = model.encode(prompts)
    cf = model.encode(cf_prompts)
    # final subject token position, per model's own tokenisation
    clean.positions["subject_last"] = model.last_token_index_of(prompts, [f.subject for f in facts])
    cf.positions["subject_last"] = model.last_token_index_of(cf_prompts, [f.cf_subject for f in facts])
    correct_cb = _continuation_batch(model, prompts, [f.obj for f in facts])
    distractor_cb = _continuation_batch(model, prompts, [f.cf_obj for f in facts])
    for cb, src in ((correct_cb, clean), (distractor_cb, clean)):
        cb.batch.positions["subject_last"] = src.positions["subject_last"]
    unrelated = torch.as_tensor(
        [[model.token_id_of(w) for w in ("the", "and", "of", "a")] for _ in facts], dtype=torch.long
    )
    return TaskData(
        clean=clean, counterfactual=cf,
        correct=torch.zeros(len(facts), dtype=torch.long),      # unused; sequence scored
        distractor=torch.zeros(len(facts), dtype=torch.long),
        family=np.asarray([f"{f.relation}|{f.subject}" for f in facts]),
        unrelated=unrelated,
        meta={"var": "factual_recall",
              "continuations": {"correct": correct_cb, "distractor": distractor_cb},
              "relations": [f.relation for f in facts],
              "subjects": [f.subject for f in facts]},
    )


def subject_difference_direction(model, data: TaskData, site, rank: int = 1) -> torch.Tensor:
    """V from "the normalized mean clean-counterfactual difference" (Sec. 4.3).

    For rank > 1 the leading principal directions of the per-prompt difference matrix
    are used, which is the natural extension and the paper's rank-four robustness
    condition.
    """
    h_clean = model.read_site(data.clean, site)
    h_cf = model.read_site(data.counterfactual, site)
    diff = (h_cf - h_clean).to(torch.float32)
    if rank <= 1:
        v = diff.mean(dim=0)
        return (v / v.norm().clamp_min(1e-9)).unsqueeze(1)
    centred = diff - diff.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centred, full_matrices=False)
    return Vh[:rank].T.contiguous()


def screen_factual(model, facts: Sequence[Fact], task: FactualRecallTask) -> Dict[str, float]:
    """Gate of Sec. 4.2: the correct object must outrank the distractor."""
    data = factual_task_data(model, facts)
    s = task.score_full(model, data)
    return {"clean_accuracy": float((s > 0).float().mean()), "n": len(facts),
            "mean_logprob_diff": float(s.mean())}


# --------------------------------------------------------------- CounterFact
# The paper's factual-recall setting follows Meng et al. (2022), whose CounterFact
# dataset is the canonical source. It is preferred over the inline table because it
# supplies (i) enough facts to reach the specified 600, (ii) paraphrase prompts, and
# (iii) a target_new that is already a relation-matched distractor. The inline table
# remains as an offline fallback; which source was used is recorded with the results.
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"


def load_counterfact(
    n: int,
    seed: int,
    path: Optional[str] = None,
    url: str = COUNTERFACT_URL,
    cache_dir: Optional[str] = None,
    max_paraphrases: int = 4,
) -> Tuple[List[Fact], Dict[str, object]]:
    """Load CounterFact and convert it to relation-matched ``Fact`` records.

    Returns the facts and a provenance dict. Raises on failure so the caller can
    decide whether to fall back rather than silently changing the dataset.
    """
    import json
    import os
    import urllib.request

    blob = None
    local = path
    if local is None and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        local = os.path.join(cache_dir, "counterfact.json")
    if local and os.path.exists(local):
        blob = json.load(open(local))
    else:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        blob = json.loads(raw)
        if local:
            open(local, "wb").write(raw)

    # index objects by relation so a donor subject can be found for each distractor
    by_relation: Dict[str, Dict[str, str]] = {}
    for rec in blob:
        rr = rec["requested_rewrite"]
        by_relation.setdefault(rr["relation_id"], {}).setdefault(rr["target_true"]["str"].strip(),
                                                                rr["subject"])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blob))
    facts: List[Fact] = []
    seen = set()
    n_no_slot = 0
    for i in order:
        rec = blob[int(i)]
        rr = rec["requested_rewrite"]
        subj = rr["subject"]
        rel = rr["relation_id"]
        obj = rr["target_true"]["str"].strip()
        dist = rr["target_new"]["str"].strip()
        if not obj or not dist or obj == dist:
            continue
        key = (subj, rel)
        if key in seen:
            continue
        donor = by_relation.get(rel, {}).get(dist)
        if donor is None or donor == subj:
            continue
        # requested_rewrite prompt uses '{}' for the subject; paraphrases are full text
        templates = [rr["prompt"].replace("{}", Fact.SLOT)]
        for p in rec.get("paraphrase_prompts", [])[: max_paraphrases - 1]:
            if subj in p:
                templates.append(p.replace(subj, Fact.SLOT))
        # A template with no subject slot would silently score a prompt that never names
        # its subject, so such records are dropped and counted rather than carried.
        templates = [t for t in templates if Fact.SLOT in t]
        if not templates:
            n_no_slot += 1
            continue
        seen.add(key)
        for ti, tpl in enumerate(templates[:max_paraphrases]):
            facts.append(Fact(subj, rel, obj, donor, dist, tpl, ti))
        if len({(f.subject, f.relation) for f in facts}) >= n:
            break
    prov = {
        "source": "counterfact",
        "url": url if not (local and os.path.exists(local)) else local,
        "n_records": len(blob),
        "n_facts": len({(f.subject, f.relation) for f in facts}),
        "n_prompts": len(facts),
        "n_dropped_no_subject_slot": n_no_slot,
    }
    return facts, prov


def build_fact_set(
    n: int,
    seed: int,
    source: str = "auto",
    cache_dir: Optional[str] = None,
    path: Optional[str] = None,
) -> Tuple[List[Fact], Dict[str, object]]:
    """Facts from CounterFact when available, otherwise the inline table."""
    if source in ("auto", "counterfact"):
        try:
            return load_counterfact(n, seed, path=path, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            if source == "counterfact":
                raise
            print(f"    [factual] CounterFact unavailable ({exc}); using the inline table")
    facts = build_facts(n * 4, seed)
    prov = {
        "source": "builtin",
        "n_facts": len({(f.subject, f.relation) for f in facts}),
        "n_prompts": len(facts),
        "note": "inline fallback; fewer distinct facts than the paper's 600",
    }
    return facts, prov
