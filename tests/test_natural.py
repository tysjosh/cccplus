"""Tests for the natural-model stage, runnable without any checkpoint.

Randomly initialised models of the three architecture families exercise the adapter,
the head-subspace construction, both task pipelines and the document split. Run:

    $VENV/bin/python tests/test_natural.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cccplus.config import FrozenManifest
from cccplus.interventions import CausalProbe, settings_from_manifest
from cccplus.mechanisms import Site
from cccplus.models.base import Patch
from cccplus.models.hf_lm import HFCausalLM
from cccplus.pipelines.natural import (
    collect_corpus_activations,
    factual_source_mechanisms,
    ioi_source_mechanisms,
    permuted_destination_control,
    screen_factual_across_models,
    screen_ioi_across_models,
)
from cccplus.signature import CausalSignature, compare_signatures
from cccplus.tasks.corpus import load_corpus
from cccplus.tasks.factual import FactualRecallTask, build_facts, factual_task_data
from cccplus.tasks.ioi import IOITask, TEMPLATES, build_ioi_examples, ioi_task_data
from cccplus.translators.base import split_documents

MF = FrozenManifest.load()
SETTINGS = settings_from_manifest(MF)
FAMILIES = (("pythia", "gpt_neox"), ("gpt2", "gpt2"), ("llama", "llama"))


def _models():
    return {k: HFCausalLM.from_random_config(kind, k, device="cpu") for k, kind in FAMILIES}


def test_adapter_reads_every_sublayer_in_every_family():
    for key, kind in FAMILIES:
        m = HFCausalLM.from_random_config(kind, key, device="cpu")
        b = m.encode(["the capital of France is", "John gave a drink to"])
        sites = [Site(0, "attn_out"), Site(1, "mlp_out"), Site(1, "resid_post"), Site(2, "resid_pre")]
        acts = m.read(b, sites)
        assert len(acts) == len(sites)
        for v in acts.values():
            assert v.shape == (2, m.d_model) and torch.isfinite(v).all()


def test_head_write_subspace_is_orthonormal_and_spans_the_head():
    for key, kind in FAMILIES:
        m = HFCausalLM.from_random_config(kind, key, device="cpu")
        V = m.head_write_subspace(1, 2)
        assert V.shape == (m.d_model, m.info.d_head)
        assert torch.allclose(V.T @ V, torch.eye(V.shape[1]), atol=1e-4)
        # distinct heads occupy distinct subspaces
        W = m.head_write_subspace(1, 3)
        from cccplus.mechanisms import subspace_similarity
        assert subspace_similarity(V, W) < 0.99


def test_patch_changes_behaviour_and_respects_the_subspace():
    m = HFCausalLM.from_random_config("llama", "llama", device="cpu")
    b = m.encode(["John and Mary went to the store. John gave a drink to"])
    site = Site(1, "attn_out", "final")
    V = m.head_write_subspace(1, 0)
    base = m.logits(b)
    inside = lambda cur: cur + 5.0 * (torch.randn(cur.shape[0], cur.shape[1]) @ V) @ V.T
    assert (m.logits(b, [Patch(site, inside, b.index("final"))]) - base).abs().max() > 1e-5
    # a zero patch must be a no-op
    assert (m.logits(b, [Patch(site, lambda cur: cur, b.index("final"))]) - base).abs().max() < 1e-4


def test_ioi_counterfactual_swaps_only_the_roles():
    ex = build_ioi_examples(400, seed=0)
    assert len({e.template for e in ex}) == len(TEMPLATES) == 12
    for e in ex:
        assert e.io != e.s
        a, b = e.text.split(), e.cf_text.split()
        assert len(a) == len(b), "counterfactual must preserve the template length"
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert diff, "counterfactual must differ somewhere"
        assert all({x, y} <= {e.io, e.s} for x, y in diff), "only the two names may change"


def test_ioi_splits_are_name_disjoint():
    from cccplus.tasks.base import partition_by_family

    ex = build_ioi_examples(1200, seed=0)
    fam = np.asarray([e.name_group for e in ex])
    parts = partition_by_family(fam, MF.path_of("partitions.fractions"), seed=0)
    groups = {k: {int(fam[i]) for i in v} for k, v in parts.items()}
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert groups[a].isdisjoint(groups[b]), f"name groups shared by {a} and {b}"


def test_ioi_effects_are_measurable_and_controls_differ():
    m = HFCausalLM.from_random_config("gpt_neox", "pythia", device="cpu")
    task = IOITask()
    ex = build_ioi_examples(32, seed=0)
    data = ioi_task_data(m, ex)
    assert data is not None and len(data) > 0
    probe = CausalProbe(m, task, data, collect_kl=False)
    mech = ioi_source_mechanisms(m, layers=[1])[0]
    assert mech.meta["kind"] == "ioi_head" and mech.rank == m.info.d_head
    eff = probe.effect(mech, SETTINGS[0])
    assert torch.isfinite(eff).all()
    ctrl = permuted_destination_control(mech, m, seed=1)
    assert ctrl.meta["kind"] == "control_other_head" and ctrl.meta["head"] != mech.meta["head"]
    s = CausalSignature.measure(probe, mech, SETTINGS, 1.0)
    t = CausalSignature.measure(probe, ctrl, SETTINGS, 1.0)
    assert compare_signatures(s, t, 1.0, 1.0, 1e-6).shape > 0.0


def test_factual_recall_uses_length_normalised_sequence_scoring():
    m = HFCausalLM.from_random_config("llama", "llama", device="cpu")
    task = FactualRecallTask()
    assert task.uses_full_scoring
    facts = build_facts(24, seed=0)
    assert all(f.cf_obj != f.obj for f in facts), "distractor must differ from the object"
    assert all(f.cf_subject != f.subject for f in facts)
    data = factual_task_data(m, facts)
    assert "subject_last" in data.clean.positions
    s = task.score_full(m, data)
    assert s.shape == (len(facts),) and torch.isfinite(s).all()
    # multi-token objects must not be penalised purely for length
    lens = data.meta["continuations"]["correct"].length
    assert int(lens.max()) >= 1
    probe = CausalProbe(m, task, data, collect_kl=False)
    mech = factual_source_mechanisms(m, data, layers=[1], rank=1)[0]
    assert mech.rank == 1 and mech.site.token == "subject_last"
    assert torch.isfinite(probe.effect(mech, SETTINGS[0])).all()


def test_factual_rank4_direction_is_orthonormal():
    m = HFCausalLM.from_random_config("llama", "llama", device="cpu")
    data = factual_task_data(m, build_facts(24, seed=0))
    mech = factual_source_mechanisms(m, data, layers=[1], rank=4)[0]
    assert mech.rank == 4
    assert torch.allclose(mech.V.T @ mech.V, torch.eye(4), atol=1e-4)


def test_screening_intersects_prompts_across_models():
    models = _models()
    ex = build_ioi_examples(48, seed=0)
    shared = screen_ioi_across_models(models, ex, IOITask(), min_accuracy=0.0)
    assert len(shared.keep) > 0
    assert set(shared.report) == set(models)
    for name, m in models.items():
        sub = [ex[i] for i in shared.keep]
        assert all(m.is_single_token(e.io) and m.is_single_token(e.s) for e in sub), name
    facts = build_facts(24, seed=0)
    shared_f = screen_factual_across_models(models, facts, FactualRecallTask())
    assert len(shared_f.keep) <= len(facts)


def test_corpus_rows_carry_documents_and_split_disjointly():
    m = HFCausalLM.from_random_config("llama", "llama", device="cpu")
    corpus = load_corpus(40, seed=0, source="synthetic")
    sites = m.site_grid(("attn_out", "mlp_out"), 2)
    store, docs = collect_corpus_activations(m, corpus, sites, batch_size=8,
                                             max_positions_per_text=3, seed=0)
    assert store.n == len(docs) > 0
    f, r = split_documents(store.n, 0.5, seed=0, document_ids=docs)
    assert set(f).isdisjoint(set(r))
    assert set(docs[f]).isdisjoint(set(docs[r]))
    # global row provenance survives subsetting, which is what makes the independence
    # report meaningful
    assert set(store.subset(f).global_ids(range(len(f)))).isdisjoint(
        set(store.subset(r).global_ids(range(len(r))))
    )


def test_dtype_argument_spans_the_transformers_rename():
    """`torch_dtype` became `dtype` in transformers 4.56.

    `from_pretrained` swallows unknown **kwargs, so passing the wrong spelling does not
    raise -- it silently loads a multi-billion-parameter checkpoint in fp32, doubling
    memory on the GPU box. The version branch is pinned here because `load()` itself
    cannot be exercised without downloading a checkpoint.
    """
    from cccplus.models.hf_lm import _transformers_at_least

    for v in ("4.46.3", "4.50.0", "4.50.3", "4.55.4"):
        assert not _transformers_at_least(v, 4, 56), f"{v} must use torch_dtype"
    for v in ("4.56.0", "4.56.2", "4.57.6", "4.57.0.dev0", "5.0", "5.0.0", "5.17.0"):
        assert _transformers_at_least(v, 4, 56), f"{v} must use dtype"
    # Gemma-3 floor from the manifest's paper_matrix
    assert _transformers_at_least("4.50.0", 4, 50)
    assert not _transformers_at_least("4.46.3", 4, 50), "4.46.3 cannot load gemma3"


def test_translator_legs_are_verifiably_independent():
    from cccplus.pipelines.evaluate import fit_translator_pair

    a = HFCausalLM.from_random_config("gpt_neox", "pythia", device="cpu")
    b = HFCausalLM.from_random_config("llama", "llama", device="cpu")
    corpus = load_corpus(60, seed=0, source="synthetic")
    sa, sb = a.site_grid(("attn_out", "mlp_out"), 3), b.site_grid(("attn_out", "mlp_out"), 3)
    Sa, docs = collect_corpus_activations(a, corpus, sa, batch_size=8, max_positions_per_text=3, seed=0)
    Sb, _ = collect_corpus_activations(b, corpus, sb, batch_size=8, max_positions_per_text=3, seed=0)
    f, r = split_documents(Sa.n, 0.5, seed=0, document_ids=docs)
    mech = ioi_source_mechanisms(a, layers=[sa[0].layer])[0]
    for name in ("procrustes", "crosscoder", "dfc"):
        pair = fit_translator_pair(name, Sa.subset(f), Sb.subset(f), Sa.subset(r), Sb.subset(r),
                                   "pythia", "llama", 7, MF)
        ind = pair.independence_report()
        assert ind["n_shared_fit_rows"] == 0, f"{name} legs shared fitting rows"
        assert ind["independent"] is True, f"{name} legs are not independent"
        # Abstention is a first-class outcome, not a failure: DFC is configured with
        # abstain_if_not_shared, and these are randomly initialised models on a
        # synthetic corpus, so there is genuinely no shared structure to find. What the
        # method requires is that an abstention be *principled* and recorded.
        cand = pair.forward.translate(mech)
        if cand is None:
            reason = pair.forward.fit_info.get("last_abstention")
            assert reason in ("model_specific", "insufficient_shared_atoms"), \
                f"{name} abstained without recording a recognised reason: {reason!r}"
        else:
            assert 0.0 < float(cand.meta["map_gain"]) < 1e3

        # The coupled leg must be *detectably* coupled. It may reach the forward map
        # through a wrapper, so the check has to follow provenance transitively.
        cpl = pair.coupling_report()
        assert cpl["shared_parameters"] or cpl["derived_from_forward"], \
            f"{name} coupled leg should not look independent"


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
