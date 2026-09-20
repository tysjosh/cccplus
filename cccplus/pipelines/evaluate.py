"""Measurement engine: fit translators, build candidates, measure every path.

Cost control: a causal signature depends only on (model, prompt split, subspace),
never on which translator proposed it, so signatures are cached by the
mechanism's projector fingerprint. Return legs from different translators that
happen to land on the same subspace are therefore measured once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..calibration import Bounds, NegativeRecord, PositiveRecord
from ..config import stable_seed
from ..interventions import CausalProbe, Setting
from ..mechanisms import Mechanism, Site, subspace_similarity
from ..signature import CausalSignature
from ..translators import BEHAVIOR_SUPERVISED, REGISTRY, UNSUPERVISED
from ..translators.base import ActivationStore, Translator, TranslatorPair
from ..translators.causal_baselines import CausalFitContext
from ..translators.crosscoder import CoupledReverse, CrosscoderTranslator, DFCTranslator
from ..translators.procrustes import ProcrustesTranslator
from ..validation import PathMeasurement, measure_path, subspace_activation_similarity
from .controlled import (
    CandidateItem,
    ProgramBundle,
    build_oracle_candidates,
    eligible_sites,
    planted_truth,
    source_mechanisms,
)

from ..validation import CORRECT_CANDIDATE_SIMILARITY   # preregistered, shared with analysis


# ------------------------------------------------------------ signature cache
class SignatureCache:
    def __init__(self, settings: Sequence[Setting]):
        self.settings = list(settings)
        self._cache: Dict[Tuple[str, str], CausalSignature] = {}
        self.hits = 0
        self.misses = 0

    def measure(self, probe: CausalProbe, mech: Mechanism, scale: float, tag: str = "") -> CausalSignature:
        key = (tag or f"{probe.model.name}|{id(probe)}", mech.fingerprint())
        sig = self._cache.get(key)
        if sig is None:
            self.misses += 1
            sig = CausalSignature.measure(probe, mech, self.settings, scale=1.0)
            self._cache[key] = sig
        else:
            self.hits += 1
        return sig.rescaled(scale)

    def binder(self, tag: str) -> Callable[[CausalProbe, Mechanism, float], CausalSignature]:
        return lambda probe, mech, scale: self.measure(probe, mech, scale, tag)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache)}


# ---------------------------------------------------------- translator fitting
def fit_translator_pair(
    name: str,
    src_store_fwd: ActivationStore,
    dst_store_fwd: ActivationStore,
    src_store_rev: ActivationStore,
    dst_store_rev: ActivationStore,
    src_model: str,
    dst_model: str,
    seed: int,
    manifest,
    shuffle_destination: bool = False,
) -> TranslatorPair:
    """Fit forward and *independent* reverse legs, plus the coupled ablation.

    The forward leg never touches the reverse half of the corpus and vice versa;
    the two legs get different seeds and, for dictionary methods, entirely separate
    dictionaries.
    """
    if shuffle_destination:
        # Negative control: destroy the pairing between the two models' activations.
        rng = np.random.default_rng(seed + 77)
        perm = rng.permutation(dst_store_fwd.n)
        dst_store_fwd = dst_store_fwd.subset(perm)
        perm_r = rng.permutation(dst_store_rev.n)
        dst_store_rev = dst_store_rev.subset(perm_r)

    if name == "procrustes":
        fwd = ProcrustesTranslator(src_model, dst_model, seed=seed).fit(src_store_fwd, dst_store_fwd)
        rev = ProcrustesTranslator(dst_model, src_model, seed=seed + 5001).fit(dst_store_rev, src_store_rev)
        coupled = fwd.coupled_reverse()
        return TranslatorPair(fwd, rev, coupled, meta={"family": name})

    if name == "crosscoder":
        cfg = manifest.path_of("translators.crosscoder")
        kw = dict(
            n_atoms=int(cfg["n_atoms"]), l1=float(cfg["l1_penalty"]), steps=int(cfg["steps"]),
            lr=float(cfg["lr"]), batch=int(cfg["batch"]), subspace_atoms=int(cfg["subspace_atoms"]),
        )
        fwd = CrosscoderTranslator(src_model, dst_model, seed=seed, **kw).fit(src_store_fwd, dst_store_fwd)
        rev = CrosscoderTranslator(dst_model, src_model, seed=seed + 5001, **kw).fit(dst_store_rev, src_store_rev)
        shared = CrosscoderTranslator(dst_model, src_model, seed=seed).attach(fwd, flip=True)
        shared.name = "crosscoder_shared_dict"
        coupled = CoupledReverse(fwd.coupled_reverse(), shared)
        return TranslatorPair(fwd, rev, coupled, meta={"family": name})

    if name == "dfc":
        cfg = manifest.path_of("translators.dfc")
        kw = dict(
            n_shared=int(cfg["n_shared_atoms"]), n_specific=int(cfg["n_specific_atoms_per_model"]),
            l1=float(cfg["l1_penalty"]), l1_specific=float(cfg["specific_penalty"]),
            steps=int(cfg["steps"]), lr=float(cfg["lr"]), batch=int(cfg["batch"]),
            subspace_atoms=int(cfg["subspace_atoms"]),
            shared_threshold=float(cfg["shared_classification_threshold"]),
        )
        fwd = DFCTranslator(src_model, dst_model, seed=seed, **kw).fit(src_store_fwd, dst_store_fwd)
        rev = DFCTranslator(dst_model, src_model, seed=seed + 5001, **kw).fit(dst_store_rev, src_store_rev)
        shared = DFCTranslator(dst_model, src_model, seed=seed).attach(fwd, flip=True)
        shared.name = "dfc_shared_dict"
        coupled = CoupledReverse(fwd.coupled_reverse(), shared)
        coupled.name = "dfc_coupled"
        return TranslatorPair(fwd, rev, coupled, meta={"family": name})

    raise ValueError(f"{name} is behaviour-supervised; fit it with fit_causal_translator")


def fit_causal_translator(
    name: str, src_model: str, dst_model: str, seed: int, manifest, context
) -> TranslatorPair:
    """Behaviour-supervised baselines: forward and reverse are separate optimisations."""
    cfg = manifest.path_of(f"causal_baselines.{name}")
    cls = REGISTRY[name]
    fwd = cls(src_model, dst_model, seed=seed, steps=int(cfg["steps"]), lr=float(cfg["lr"]))
    fwd.fit(None, None, context=context["forward"])
    rev = cls(dst_model, src_model, seed=seed + 5001, steps=int(cfg["steps"]), lr=float(cfg["lr"]))
    rev.fit(None, None, context=context["reverse"])
    return TranslatorPair(fwd, rev, None, meta={"family": name, "supervised": True})


# ------------------------------------------------------------------- helpers
def _repr_dest_fn(bundle: ProgramBundle, dest_inst: str) -> Callable[[Mechanism], torch.Tensor]:
    store = bundle.store(dest_inst, "all")

    def fn(mech: Mechanism) -> torch.Tensor:
        return store.get(mech.site) @ mech.V

    return fn


def _repr_source(bundle: ProgramBundle, src_inst: str, mech: Mechanism) -> torch.Tensor:
    return bundle.store(src_inst, "all").get(mech.site) @ mech.V


def label_translator_candidate(
    candidate: Optional[Mechanism], truth_blocks: Sequence[Mechanism]
) -> Tuple[int, float]:
    """Is the proposed subspace the planted counterpart? (preregistered threshold)"""
    if candidate is None or not truth_blocks:
        return 0, 0.0
    sim = max(
        subspace_similarity(candidate.V, t.V) if str(candidate.site) == str(t.site) else 0.0
        for t in truth_blocks
    )
    return int(sim >= CORRECT_CANDIDATE_SIMILARITY), float(sim)


# ---------------------------------------------------------------- evaluator
@dataclass
class EvalConfig:
    settings: Sequence[Setting]
    weight_scheme: str
    eta: float
    reverse_anchor: bool = True


class ControlledEvaluator:
    """Runs Part A (rule discrimination) and Part B (translator evaluation)."""

    def __init__(self, bundle: ProgramBundle, manifest, cfg: EvalConfig, seed: int,
                 cache_dir=None):
        self.bundle = bundle
        self.mf = manifest
        self.cfg = cfg
        self.seed = seed
        self.cache = SignatureCache(cfg.settings)
        self.cache_dir = cache_dir
        self.src_inst = "src"
        self.dest_insts = [k for k in bundle.models if k != "src"]
        self.src_mechs = source_mechanisms(bundle.model("src"))
        self._pairs: Dict[Tuple[str, str, int, bool], TranslatorPair] = {}
        self._supervised: Dict[Tuple[str, str, int, int], TranslatorPair] = {}
        self._oracle: Dict[Tuple[str, int, str], List[CandidateItem]] = {}

    def path_allowed(self, dest_inst: str, var: int) -> bool:
        """Restrict the absent-variable destination to the variable it is missing.

        ``dst_absent`` is trained on the restricted program, so for the variables it
        *does* share its answer key differs from the source's wherever the dropped
        variable is non-zero. Comparing signatures across two different behavioural
        metrics would conflate a metric difference with a mechanism difference, so
        this instance contributes only the absent-case path it exists to provide.
        For that path the comparison is well posed: the source mechanism has a real
        effect and the destination provably has none.
        """
        absent = self.bundle.absent(dest_inst)
        return (var in absent) if absent else True

    # ------------------------------------------------------------ translators
    def pair(self, dest_inst: str, name: str, seed: int, shuffled: bool = False) -> TranslatorPair:
        key = (dest_inst, name, seed, shuffled)
        if key not in self._pairs:
            b = self.bundle
            cached = self._load_pair(dest_inst, name, seed, shuffled)
            if cached is not None:
                self._pairs[key] = cached
            else:
                pair = fit_translator_pair(
                    name,
                    b.store(self.src_inst, "forward"), b.store(dest_inst, "forward"),
                    b.store(self.src_inst, "reverse"), b.store(dest_inst, "reverse"),
                    b.model(self.src_inst).model.name, b.model(dest_inst).model.name,
                    stable_seed(self.seed, b.program.name, dest_inst, name, seed, shuffled) % (2**31),
                    self.mf, shuffle_destination=shuffled,
                )
                self._pairs[key] = pair
                self._save_pair(dest_inst, name, seed, shuffled, pair)
        return self._pairs[key]

    # ------------------------------------------- behaviour-supervised baselines
    def causal_context(self, dest_inst: str, var: int) -> Dict[str, CausalFitContext]:
        """Fit contexts for the behaviour-supervised baselines, both directions.

        Fitted on the **hyperparameter** split only, never on calibration or test
        prompts, so a supervised baseline gets no more label access than the
        unsupervised translators get corpus access.

        The interchange target is the program's output once ``var`` is imported from the
        counterfactual prompt. That is exactly ``TaskData.distractor``: the prompt sets
        differ from the clean values in the column for ``var`` alone, so the
        counterfactual answer key *is* the interchange label.
        """
        b = self.bundle
        src, dst = b.model(self.src_inst), b.model(dest_inst)
        src_data = b.probe(self.src_inst, var, "hyperparameter").data
        dst_data = b.probe(dest_inst, var, "hyperparameter").data
        fwd = CausalFitContext(
            source_model=src.model, dest_model=dst.model, task=b.task,
            source_data=src_data, dest_data=dst_data,
            dest_sites=eligible_sites(dst),
            interchange_target=dst_data.distractor, clean_target=dst_data.correct,
        )
        rev = CausalFitContext(
            source_model=dst.model, dest_model=src.model, task=b.task,
            source_data=dst_data, dest_data=src_data,
            dest_sites=eligible_sites(src),
            interchange_target=src_data.distractor, clean_target=src_data.correct,
        )
        return {"forward": fwd, "reverse": rev}

    def supervised_pair(self, dest_inst: str, var: int, name: str, seed: int) -> TranslatorPair:
        """A fitted behaviour-supervised pair. Not disk-cached: the fit holds live
        model handles, and the per-mechanism optimisation happens inside translate()."""
        key = (dest_inst, name, seed, var)
        if key not in self._supervised:
            b = self.bundle
            self._supervised[key] = fit_causal_translator(
                name,
                b.model(self.src_inst).model.name, b.model(dest_inst).model.name,
                stable_seed(self.seed, b.program.name, dest_inst, name, seed, var) % (2**31),
                self.mf, self.causal_context(dest_inst, var),
            )
        return self._supervised[key]

    def measure_supervised_paths(
        self, split: str, bounds: Bounds, seeds_by_name: Dict[str, List[int]]
    ) -> List[PathMeasurement]:
        """Part C: candidates proposed by the behaviour-supervised baselines.

        Reported apart from Part B because these translators have no coupled return
        leg, so pooling them would make a rule that needs one look like it failed when
        the construction simply does not exist for them.
        """
        out: List[PathMeasurement] = []
        b = self.bundle
        prog = b.program.name
        for var, src_mech in self.src_mechs.items():
            src_probe = b.probe(self.src_inst, var, split)
            src_sig = self.cache.measure(
                src_probe, src_mech, bounds.c(src_probe.model.name, b.task.name),
                f"{src_probe.model.name}|{id(src_probe)}",
            )
            repr_src = _repr_source(b, self.src_inst, src_mech)
            for dest_inst in self.dest_insts:
                if not self.path_allowed(dest_inst, var):
                    continue
                dest_probe = b.probe(dest_inst, var, split)
                repr_fn = _repr_dest_fn(b, dest_inst)
                truth = planted_truth(b.model(dest_inst), var)
                truth_mechs = [x.mechanism(b.model(dest_inst).model.name) for x in truth["blocks"]]
                exists = truth["case"] != "absent"
                for name in [n for n in seeds_by_name if n in BEHAVIOR_SUPERVISED]:
                    for tseed in seeds_by_name[name]:
                        pair = self.supervised_pair(dest_inst, var, name, tseed)
                        cand = pair.translate(src_mech)
                        f_ind, f_cou = self._reverse_fns(pair, src_mech.site)
                        correct, sim = label_translator_candidate(cand, truth_mechs)
                        case, label = (truth["case"], correct) if exists else ("absent", 0)
                        m = measure_path(
                            program=prog, task_name=b.task.name,
                            source_probe=src_probe, dest_probe=dest_probe,
                            settings=self.cfg.settings,
                            source_mech=src_mech, source_signature=src_sig,
                            candidate=cand,
                            reverse_independent=f_ind, reverse_coupled=f_cou,
                            bounds=bounds, translator=name, seed=tseed,
                            case=case, label=label, truth_blocks=truth_mechs,
                            repr_source=repr_src, repr_dest_acts=repr_fn,
                            family_key=f"{prog}|v{var}",
                            weight_scheme=self.cfg.weight_scheme, eta=self.cfg.eta,
                            sig_fn=self._sig_fn(),
                        )
                        m.diagnostics["candidate_similarity_to_truth"] = sim
                        m.diagnostics["counterpart_exists"] = float(exists)
                        m.diagnostics["var"] = float(var)
                        m.diagnostics["behavior_supervised"] = 1.0
                        m.truth_subspace_similarity = sim
                        out.append(m)
        return out

    # ---------------------------------------------------------- disk cache
    def _pair_path(self, dest_inst: str, name: str, seed: int, shuffled: bool):
        if self.cache_dir is None:
            return None
        cfg_hash = stable_seed(
            str(self.mf.path_of(f"translators.{name}", {})),
            str(self.mf.path_of("controlled.mechanism_rank")),
            str(self.mf.path_of("controlled.n_prompts.probe")),
        ) % (2**31)
        tag = f"{self.bundle.program.name}__{dest_inst}__{name}__s{seed}"
        tag += "__shuf" if shuffled else ""
        return self.cache_dir / f"{tag}__{cfg_hash}.pt"

    def _load_pair(self, dest_inst: str, name: str, seed: int, shuffled: bool):
        p = self._pair_path(dest_inst, name, seed, shuffled)
        if p is None or not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            return None

    def _save_pair(self, dest_inst: str, name: str, seed: int, shuffled: bool, pair) -> None:
        p = self._pair_path(dest_inst, name, seed, shuffled)
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pair, p)
        except Exception:
            pass

    # -------------------------------------------------------------- candidates
    def oracle_candidates(self, dest_inst: str, var: int, split: str) -> List[CandidateItem]:
        key = (dest_inst, var, split)
        if key not in self._oracle:
            self._oracle[key] = build_oracle_candidates(
                self.bundle, self.src_inst, dest_inst, var,
                self.bundle.probe(dest_inst, var, split),
                stable_seed(self.seed, self.bundle.program.name, dest_inst, var, "decoy") % (2**31),
            )
        return self._oracle[key]

    # ------------------------------------------------------------- reverse fns
    def _reverse_fns(self, pair: TranslatorPair, anchor: Optional[Site]):
        def set_anchor(tr: Optional[Translator]):
            if tr is not None and hasattr(tr, "anchor_site"):
                tr.anchor_site = anchor if self.cfg.reverse_anchor else None
            return tr

        ind = set_anchor(pair.reverse_independent)
        cou = set_anchor(pair.reverse_coupled)
        f_ind = (lambda m: ind.translate(m)) if ind is not None else (lambda m: None)
        f_cou = (lambda m: cou.translate(m)) if cou is not None else None
        return f_ind, f_cou

    # ------------------------------------------------------------------ part A
    def measure_rule_discrimination(
        self, split: str, bounds: Bounds, translators: Sequence[str], seeds_by_name: Dict[str, List[int]]
    ) -> List[PathMeasurement]:
        """Same candidate set for every rule; only the reverse map varies."""
        out: List[PathMeasurement] = []
        b = self.bundle
        prog = b.program.name
        for var, src_mech in self.src_mechs.items():
            src_probe = b.probe(self.src_inst, var, split)
            src_sig = self.cache.measure(
                src_probe, src_mech, bounds.c(src_probe.model.name, b.task.name),
                f"{src_probe.model.name}|{id(src_probe)}",
            )
            repr_src = _repr_source(b, self.src_inst, src_mech)
            for dest_inst in self.dest_insts:
                if not self.path_allowed(dest_inst, var):
                    continue
                dest_probe = b.probe(dest_inst, var, split)
                repr_fn = _repr_dest_fn(b, dest_inst)
                items = self.oracle_candidates(dest_inst, var, split)
                for name in translators:
                    for tseed in seeds_by_name.get(name, [0]):
                        pair = self.pair(dest_inst, name, tseed)
                        f_ind, f_cou = self._reverse_fns(pair, src_mech.site)
                        for item in items:
                            out.append(
                                measure_path(
                                    program=prog, task_name=b.task.name,
                                    source_probe=src_probe, dest_probe=dest_probe,
                                    settings=self.cfg.settings,
                                    source_mech=src_mech, source_signature=src_sig,
                                    candidate=item.mechanism,
                                    reverse_independent=f_ind, reverse_coupled=f_cou,
                                    bounds=bounds, translator=name, seed=tseed,
                                    case=item.case, label=item.label,
                                    truth_blocks=item.truth,
                                    repr_source=repr_src, repr_dest_acts=repr_fn,
                                    family_key=f"{prog}|v{var}",
                                    weight_scheme=self.cfg.weight_scheme, eta=self.cfg.eta,
                                    sig_fn=self._sig_fn(),
                                )
                            )
                            out[-1].diagnostics["set_valued"] = float(item.set_valued)
                            out[-1].diagnostics["var"] = float(var)
        return out

    def _sig_fn(self, *_ignored):
        """Signatures are cached per (probe, subspace); the probe fixes model+split."""
        cache = self.cache

        def fn(probe: CausalProbe, mech: Mechanism, scale: float) -> CausalSignature:
            return cache.measure(probe, mech, scale, tag=f"{probe.model.name}|{id(probe)}")

        return fn

    # ------------------------------------------------------------------ part B
    def measure_translator_paths(
        self, split: str, bounds: Bounds, seeds_by_name: Dict[str, List[int]],
        include_shuffled: bool = True, causal_contexts=None,
    ) -> List[PathMeasurement]:
        """Candidates proposed by the translators themselves (coverage + abstention)."""
        out: List[PathMeasurement] = []
        b = self.bundle
        prog = b.program.name
        for var, src_mech in self.src_mechs.items():
            src_probe = b.probe(self.src_inst, var, split)
            src_sig = self.cache.measure(
                src_probe, src_mech, bounds.c(src_probe.model.name, b.task.name),
                f"{src_probe.model.name}|{id(src_probe)}",
            )
            repr_src = _repr_source(b, self.src_inst, src_mech)
            for dest_inst in self.dest_insts:
                if not self.path_allowed(dest_inst, var):
                    continue
                dest_probe = b.probe(dest_inst, var, split)
                repr_fn = _repr_dest_fn(b, dest_inst)
                truth = planted_truth(b.model(dest_inst), var)
                truth_mechs = [x.mechanism(b.model(dest_inst).model.name) for x in truth["blocks"]]
                exists = truth["case"] != "absent"
                for name, seeds in seeds_by_name.items():
                    # Behaviour-supervised baselines consume task labels and have no
                    # coupled return construction, so they are measured separately by
                    # measure_supervised_paths rather than pooled in here.
                    if name not in UNSUPERVISED:
                        continue
                    variants = [False, True] if include_shuffled else [False]
                    for shuffled in variants:
                        for tseed in seeds:
                            pair = self.pair(dest_inst, name, tseed, shuffled=shuffled)
                            cand = pair.translate(src_mech)
                            f_ind, f_cou = self._reverse_fns(pair, src_mech.site)
                            correct, sim = label_translator_candidate(cand, truth_mechs)
                            if shuffled:
                                case, label = "shuffled_pair_control", 0
                            elif not exists:
                                case, label = "absent", 0
                            else:
                                case, label = truth["case"], correct
                            m = measure_path(
                                program=prog, task_name=b.task.name,
                                source_probe=src_probe, dest_probe=dest_probe,
                                settings=self.cfg.settings,
                                source_mech=src_mech, source_signature=src_sig,
                                candidate=cand,
                                reverse_independent=f_ind, reverse_coupled=f_cou,
                                bounds=bounds, translator=name + ("_shuffled" if shuffled else ""),
                                seed=tseed, case=case, label=label,
                                truth_blocks=truth_mechs,
                                repr_source=repr_src, repr_dest_acts=repr_fn,
                                family_key=f"{prog}|v{var}",
                                weight_scheme=self.cfg.weight_scheme, eta=self.cfg.eta,
                                sig_fn=self._sig_fn(),
                            )
                            m.diagnostics["candidate_similarity_to_truth"] = sim
                            m.diagnostics["counterpart_exists"] = float(exists)
                            m.diagnostics["var"] = float(var)
                            m.truth_subspace_similarity = sim
                            out.append(m)
        return out


# --------------------------------------------------------- negatives for eps
def positives_from(records: Sequence[PathMeasurement]) -> List[PositiveRecord]:
    """Calibration positives, used by the cycle-aware bound mode only."""
    out: List[PositiveRecord] = []
    for m in records:
        if m.label != 1 or m.abstained:
            continue
        out.append(
            PositiveRecord(
                program=m.program, model=m.dest_model, task=m.task,
                d_shape=m.d_shape_forward, d_mag=m.d_mag_forward,
                scalar_error=m.ret_independent.scalar_error,
                representational=m.representational,
                d_shape_return=m.ret_independent.d_shape,
                d_mag_return=m.ret_independent.d_mag,
            )
        )
    return out


def negatives_from(records: Sequence[PathMeasurement]) -> List[NegativeRecord]:
    """Calibration negatives, used to cap the equivalence bounds (Sec. 3.6)."""
    out: List[NegativeRecord] = []
    for m in records:
        if m.label != 0 or m.abstained:
            continue
        out.append(
            NegativeRecord(
                program=m.program, model=m.dest_model, task=m.task,
                d_shape=m.d_shape_forward, d_mag=m.d_mag_forward,
                scalar_error=m.ret_independent.scalar_error,
                representational=m.representational,
                d_shape_return=m.ret_independent.d_shape,
                d_mag_return=m.ret_independent.d_mag,
            )
        )
    return out
