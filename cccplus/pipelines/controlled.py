"""Controlled ground-truth stage (Sec. 4.1) -- the primary methodological test.

The stage runs in two parts, both over the same planted models and prompts.

Part A, rule discrimination.
  Candidates are supplied *by construction*: the planted counterpart (positive),
  and matched decoys (negative). Every validation rule then sees exactly the same
  candidate set, which is what Sec. 3.6 requires ("Validation rules are compared
  on the same candidates") and what makes false acceptance at matched positive
  retention an apples-to-apples comparison of rules rather than of translators.

Part B, translator evaluation.
  Candidates come from the translators themselves, so coverage and abstention are
  measured as the paper defines them, and prompt-level signature agreement can be
  reported per translator.

Negatives are deliberately hard:
  wrong_variable         another planted block in the same destination: same rank,
                         comparable effect size, different causal role;
  effect_matched_random  a random subspace at the *same site* as the true
                         counterpart, selected so that its mean effect matches the
                         true counterpart's in sign and magnitude -- exactly the
                         candidate a mean-effect test cannot reject;
  absent                 the destination does not represent the variable at all,
                         so any non-null acceptance is a false acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..benchmarks.programs import Program, program_output
from ..benchmarks.planted import PlantedBlock, mechanism_sites
from ..benchmarks.siit import SIITModel, make_batch
from ..calibration import Calibrator, ControlRecord, NegativeRecord
from ..config import stable_seed
from ..interventions import CausalProbe, Setting, null_direction_mechanisms
from ..mechanisms import Mechanism, Site, orthonormalize, subspace_similarity
from ..models.base import PromptBatch
from ..signature import CausalSignature, compare_signatures, signature_weights, weighted_norm
from ..tasks.base import LogitDiffTask, TaskData, partition_by_family
from ..translators.base import ActivationStore, TranslatorPair, collect_activations, split_documents

SPLITS = ("discovery", "hyperparameter", "calibration", "test")


# --------------------------------------------------------------------- data
@dataclass
class PromptSet:
    """Prompts for one intervened variable, shared by every model."""

    var: int
    clean: np.ndarray            # [N, T]
    cf: np.ndarray               # [N, T]
    values: np.ndarray           # [N, K]
    cf_values: np.ndarray        # [N, K]
    family: np.ndarray           # [N] template-family id

    def subset(self, idx: np.ndarray) -> "PromptSet":
        return PromptSet(self.var, self.clean[idx], self.cf[idx], self.values[idx],
                         self.cf_values[idx], self.family[idx])

    def __len__(self) -> int:
        return int(self.clean.shape[0])


def build_prompt_set(
    program: Program, var: int, n_families: int, per_family: int, seed: int
) -> PromptSet:
    """Template families: a fixed context, with only the target group varying.

    Each family fixes the tokens of every group except the intervened one, so
    partitioning by family gives context-disjoint splits -- the synthetic analogue
    of IOI's split-disjoint template/name families.
    """
    rng = np.random.default_rng(seed)
    K = program.n_vars
    clean_rows, cf_rows, val_rows, cfval_rows, fam_rows = [], [], [], [], []
    for f in range(n_families):
        ctx_values = rng.integers(0, 3, size=K)
        ctx_tokens = program.tokens_for_values(ctx_values[None, :], rng)[0]
        v_target = rng.integers(0, 3, size=per_family)
        shift = rng.integers(1, 3, size=per_family)
        v_cf = (v_target + shift) % 3
        clean = np.tile(ctx_tokens, (per_family, 1))
        program.fill_group(clean, var, v_target, rng)
        cf = clean.copy()
        program.fill_group(cf, var, v_cf, rng)
        values = np.tile(ctx_values, (per_family, 1))
        values[:, var] = v_target
        cf_values = values.copy()
        cf_values[:, var] = v_cf
        clean_rows.append(clean)
        cf_rows.append(cf)
        val_rows.append(values)
        cfval_rows.append(cf_values)
        fam_rows.append(np.full(per_family, f, dtype=int))
    return PromptSet(
        var,
        np.concatenate(clean_rows), np.concatenate(cf_rows),
        np.concatenate(val_rows), np.concatenate(cfval_rows), np.concatenate(fam_rows),
    )


def task_data_for(program: Program, ps: PromptSet, absent: Set[int]) -> TaskData:
    """Attach a model-specific answer key (the model's own program variant)."""
    correct = program_output(ps.values, absent)
    distractor = program_output(ps.cf_values, absent)
    n = len(ps)
    unrelated = np.zeros((n, program.n_out - 2), dtype=np.int64)
    for i in range(n):
        others = [c for c in range(program.n_out) if c != correct[i] and c != distractor[i]]
        unrelated[i] = np.asarray(others[: program.n_out - 2])
    return TaskData(
        clean=make_batch(program, ps.clean),
        counterfactual=make_batch(program, ps.cf),
        correct=torch.as_tensor(correct, dtype=torch.long),
        distractor=torch.as_tensor(distractor, dtype=torch.long),
        family=ps.family,
        unrelated=torch.as_tensor(unrelated, dtype=torch.long),
        meta={"var": ps.var},
    )


# ------------------------------------------------------------------- bundle
@dataclass
class ProgramBundle:
    """Everything needed to run the protocol for one program."""

    program: Program
    models: Dict[str, SIITModel]
    prompts: Dict[Tuple[int, str], PromptSet] = field(default_factory=dict)
    probes: Dict[Tuple[str, int, str], CausalProbe] = field(default_factory=dict)
    stores: Dict[Tuple[str, str], ActivationStore] = field(default_factory=dict)
    task: LogitDiffTask = field(default_factory=lambda: LogitDiffTask("program_logit_diff", -1.0))
    splits: Dict[Tuple[int], Dict[str, np.ndarray]] = field(default_factory=dict)

    # ------------------------------------------------------------- accessors
    def model(self, inst: str) -> SIITModel:
        return self.models[inst]

    def absent(self, inst: str) -> Set[int]:
        return set(self.models[inst].placement.absent_vars)

    def probe(self, inst: str, var: int, split: str) -> CausalProbe:
        key = (inst, var, split)
        if key not in self.probes:
            sm = self.models[inst]
            ps = self.prompts[(var, split)]
            data = task_data_for(self.program, ps, self.absent(inst))
            sites = eligible_sites(sm)
            self.probes[key] = CausalProbe(sm.model, self.task, data, sites=sites, collect_kl=False)
        return self.probes[key]

    def store(self, inst: str, half: str) -> ActivationStore:
        return self.stores[(inst, half)]


def eligible_sites(sm: SIITModel) -> List[Site]:
    """Sites a translator is allowed to consider (the preregistered site family)."""
    return mechanism_sites(sm.model.cfg.n_layers)


# ---------------------------------------------------------- bundle assembly
def build_bundle(
    program: Program,
    models: Dict[str, SIITModel],
    manifest,
    seed: int,
) -> ProgramBundle:
    bundle = ProgramBundle(program=program, models=models)
    n_fam = int(manifest.path_of("controlled.n_families", 20))
    per_fam = int(manifest.path_of("controlled.per_family", 40))
    fractions = manifest.path_of("partitions.fractions")

    for var in range(program.n_vars):
        full = build_prompt_set(program, var, n_fam, per_fam, stable_seed(seed, program.name, "prompts", var))
        parts = partition_by_family(full.family, fractions, stable_seed(seed, program.name, "split", var))
        for split, idx in parts.items():
            bundle.prompts[(var, split)] = full.subset(idx)
        bundle.splits[(var,)] = parts

    # Generic probe corpus for translator fitting, split by document.
    n_probe = int(manifest.path_of("controlled.n_prompts.probe"))
    rng = np.random.default_rng(stable_seed(seed, program.name, "probe"))
    probe_tokens = program.sample_plain(n_probe, rng)
    fwd_idx, rev_idx = split_documents(
        n_probe, float(manifest.path_of("partitions.translator_corpus.forward_share")),
        stable_seed(seed, program.name, "probe_split"),
    )
    probe_batch = make_batch(program, probe_tokens)
    for inst, sm in models.items():
        sites = eligible_sites(sm)
        store = collect_activations(sm.model, probe_batch, sites)
        bundle.stores[(inst, "forward")] = store.subset(fwd_idx)
        bundle.stores[(inst, "reverse")] = store.subset(rev_idx)
        bundle.stores[(inst, "all")] = store
    return bundle


# ------------------------------------------------------- source mechanisms
def source_mechanisms(sm: SIITModel) -> Dict[int, Mechanism]:
    """The planted block of each variable in the source model."""
    out: Dict[int, Mechanism] = {}
    for var in range(sm.placement.n_vars):
        blocks = [b for b in sm.placement.blocks if b.vars == frozenset({var})]
        if len(blocks) == 1:
            out[var] = blocks[0].mechanism(sm.model.name)
    return out


# --------------------------------------------------------------- candidates
@dataclass
class CandidateItem:
    """One labelled candidate on a translation path."""

    var: int
    dest_inst: str
    case: str
    label: int
    mechanism: Optional[Mechanism]
    truth: List[Mechanism] = field(default_factory=list)
    set_valued: bool = False
    note: str = ""


def planted_truth(sm: SIITModel, var: int) -> Dict[str, object]:
    return sm.placement.counterpart(var)


def build_oracle_candidates(
    bundle: ProgramBundle,
    src_inst: str,
    dest_inst: str,
    var: int,
    probe_dest: CausalProbe,
    seed: int,
    n_random_tries: int = 24,
) -> List[CandidateItem]:
    """Positives from the planted structure, plus matched negatives."""
    dst = bundle.model(dest_inst)
    truth = planted_truth(dst, var)
    case = str(truth["case"])
    items: List[CandidateItem] = []
    truth_mechs = [b.mechanism(dst.model.name) for b in truth["blocks"]]

    if case == "absent":
        # The only correct answer is abstention. The candidate a similarity-driven
        # translator would nonetheless propose: the destination block whose
        # activations align best with the source mechanism's.
        best = _most_similar_block(bundle, src_inst, dest_inst, var)
        items.append(CandidateItem(var, dest_inst, "absent", 0, best, [], note="no counterpart exists"))
        return items

    if case in ("shared", "permuted"):
        for t in truth_mechs:
            items.append(CandidateItem(var, dest_inst, case, 1, t, truth_mechs))
    elif case == "redundant":
        # Either copy alone transfers the variable, so each is a valid counterpart.
        for t in truth_mechs:
            items.append(CandidateItem(var, dest_inst, "redundant", 1, t, truth_mechs))
    elif case == "split":
        # A variable spread over two *sites* is not a component mechanism at all; it
        # belongs to the multi-site extension (Sec. 3.7). At component level only the
        # halves exist, and a half is a partial correspondence, not the counterpart.
        joint = _join_blocks(truth["blocks"], dst.model.name)
        if joint is not None:
            items.append(CandidateItem(var, dest_inst, "split", 1, joint, truth_mechs, set_valued=True))
        for t in truth_mechs:
            items.append(CandidateItem(var, dest_inst, "split_half", 0, t, truth_mechs,
                                       set_valued=True, note="one half of a split variable"))
    elif case == "merged":
        # The merged block does carry the variable, but it carries another one too.
        for t in truth_mechs:
            items.append(CandidateItem(var, dest_inst, "merged", 1, t, truth_mechs, set_valued=True))

    # ---- negatives: other planted variables in the same destination
    for b in dst.placement.decoy_blocks(var):
        items.append(
            CandidateItem(var, dest_inst, "wrong_variable", 0, b.mechanism(dst.model.name), truth_mechs,
                          note=f"carries v{sorted(b.vars)}")
        )

    # ---- negative: effect-matched random subspace at the true counterpart's site
    if truth_mechs:
        target = truth_mechs[0]
        ref = float(probe_dest.effect(target, Setting()).mean())
        rnd = _effect_matched_random(probe_dest, target, ref, seed, n_random_tries)
        if rnd is not None:
            items.append(
                CandidateItem(var, dest_inst, "effect_matched_random", 0, rnd, truth_mechs,
                              note="random subspace with matched mean effect")
            )
    return items


def _join_blocks(blocks: Sequence[PlantedBlock], model_name: str) -> Optional[Mechanism]:
    """Stack blocks that share a site; split variables spanning sites cannot be one mechanism."""
    sites = {str(b.site) for b in blocks}
    if len(sites) != 1:
        return None
    V = torch.cat([b.V for b in blocks], dim=1)
    return Mechanism(site=blocks[0].site, V=V, model=model_name, label="joint")


def _most_similar_block(bundle: ProgramBundle, src_inst: str, dest_inst: str, var: int) -> Optional[Mechanism]:
    """Destination block whose in-subspace activations align best with the source's."""
    from ..validation import subspace_activation_similarity

    src = bundle.model(src_inst)
    dst = bundle.model(dest_inst)
    src_mech = source_mechanisms(src).get(var)
    if src_mech is None:
        return None
    xs = bundle.store(src_inst, "all").get(src_mech.site) @ src_mech.V
    best, best_s = None, -1.0
    for b in dst.placement.blocks:
        m = b.mechanism(dst.model.name)
        xb = bundle.store(dest_inst, "all").get(m.site) @ m.V
        s = subspace_activation_similarity(xs, xb)
        if s > best_s:
            best, best_s = m, s
    return best


def _effect_matched_random(
    probe: CausalProbe, target: Mechanism, ref_effect: float, seed: int, tries: int
) -> Optional[Mechanism]:
    """A random subspace at the same site whose mean effect matches ``ref_effect``.

    This negative exists to defeat the mean-effect test specifically: it has the
    right sign and (approximately) the right magnitude, and a different
    prompt-level signature.
    """
    if not np.isfinite(ref_effect) or abs(ref_effect) < 1e-9:
        return None
    gen = torch.Generator().manual_seed(int(seed))
    d, r = target.d, target.rank
    best, best_gap = None, float("inf")
    for _ in range(tries):
        V = torch.randn(d, r, generator=gen)
        cand = Mechanism(site=target.site, V=V, model=target.model, label="random")
        eff = float(probe.effect(cand, Setting()).mean())
        if np.sign(eff) != np.sign(ref_effect):
            continue
        gap = abs(abs(eff) - abs(ref_effect)) / max(abs(ref_effect), 1e-9)
        if gap < best_gap:
            best, best_gap = cand, gap
        if gap < 0.15:
            break
    if best is None:
        return None
    best.meta["effect_gap"] = best_gap
    best.label = "effect_matched_random"
    return best


# ------------------------------------------------------- calibration records
def add_calibration_records(
    cal: Calibrator,
    bundle: ProgramBundle,
    inst: str,
    settings: Sequence[Setting],
    weight_scheme: str,
    eta: float,
    seed: int,
) -> None:
    """Controls for tau_min, c_{M,T} and the equivalence bounds (Sec. 3.6)."""
    program = bundle.program
    sm = bundle.model(inst)
    sites = eligible_sites(sm)
    task_name = bundle.task.name
    mechs = source_mechanisms(sm)

    for var in range(program.n_vars):
        probe = bundle.probe(inst, var, "calibration")

        # --- null-direction controls: random subspaces with no planted role
        for nm in null_direction_mechanisms(sm.model, sites, rank=4, n=6,
                                            seed=stable_seed(seed, program.name, inst, "null", var)):
            eff = float(probe.effect(nm, settings[0]).mean())
            cal.add_control(ControlRecord("null_direction", program.name, sm.model.name, task_name,
                                          abs_mean_effect=abs(eff)))

        # --- unrelated-task controls: a planted mechanism measured on prompts that
        #     do not move its variable
        for other_var, mech in mechs.items():
            if other_var == var:
                continue
            eff = float(probe.effect(mech, settings[0]).mean())
            cal.add_control(ControlRecord("unrelated_task", program.name, sm.model.name, task_name,
                                          abs_mean_effect=abs(eff)))

        # --- source-gated norms feeding c_{M,T}
        mech = mechs.get(var)
        if mech is not None:
            sig = CausalSignature.measure(probe, mech, settings, scale=1.0)
            w = signature_weights(sig.n_settings, sig.n_prompts, weight_scheme)
            cal.add_gated_norm(sm.model.name, task_name, weighted_norm(sig.vector(), w))

    # --- equivalence-bound controls
    for var, mech in mechs.items():
        probe = bundle.probe(inst, var, "calibration")
        sig = CausalSignature.measure(probe, mech, settings, scale=1.0)
        n = sig.n_prompts
        half = n // 2
        if half >= 8:
            # identity: two matched halves of the same calibration prompts, so the
            # distance reflects prompt-level estimation noise alone
            a = CausalSignature(sig.setting_keys, sig.effects[:, :half], 1.0, sm.model.name, mech.key())
            b = CausalSignature(sig.setting_keys, sig.effects[:, half : 2 * half], 1.0, sm.model.name, mech.key())
            c_a = max(weighted_norm(a.vector(), signature_weights(a.n_settings, a.n_prompts, weight_scheme)), 1e-9)
            c_b = max(weighted_norm(b.vector(), signature_weights(b.n_settings, b.n_prompts, weight_scheme)), 1e-9)
            cmp = compare_signatures(a.rescaled(c_a), b.rescaled(c_a), 1.0, 1.0, eta, weight_scheme)
            cal.add_control(ControlRecord("identity", program.name, sm.model.name, task_name,
                                          d_shape=cmp.shape, d_mag=cmp.magnitude,
                                          scalar_error=abs(a.mean_effect() - b.mean_effect()),
                                          representational=1.0))
            # scale reparameterisation: the same signature under an independently
            # estimated response scale
            cmp_s = compare_signatures(a.rescaled(c_a), a.rescaled(c_b), 1.0, 1.0, eta, weight_scheme)
            cal.add_control(ControlRecord("scale_reparameterization", program.name, sm.model.name, task_name,
                                          d_shape=cmp_s.shape, d_mag=cmp_s.magnitude,
                                          scalar_error=0.0, representational=1.0))
        # orthogonal reparameterisation of the basis: same subspace, same projector
        g = torch.Generator().manual_seed(stable_seed(seed, "ortho", inst, var) % (2**31))
        O, _ = torch.linalg.qr(torch.randn(mech.rank, mech.rank, generator=g))
        rot = Mechanism(site=mech.site, V=mech.V @ O, model=mech.model, label="ortho")
        sig_rot = CausalSignature.measure(probe, rot, settings, scale=1.0)
        w = signature_weights(sig.n_settings, sig.n_prompts, weight_scheme)
        c = max(weighted_norm(sig.vector(), w), 1e-9)
        cmp_o = compare_signatures(sig.rescaled(c), sig_rot.rescaled(c), 1.0, 1.0, eta, weight_scheme)
        cal.add_control(ControlRecord("orthogonal_reparameterization", program.name, sm.model.name, task_name,
                                      d_shape=cmp_o.shape, d_mag=cmp_o.magnitude,
                                      scalar_error=abs(sig.mean_effect() - sig_rot.mean_effect()),
                                      representational=1.0))
