"""Tests for the CCC+ method primitives and for each fixed correctness defect.

Run:  $VENV/bin/python -m pytest tests -q
      $VENV/bin/python tests/test_method.py     (no pytest needed)

The regression tests below are named after the defect they pin; each one fails on
the pre-fix behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cccplus.calibration import ControlRecord, NegativeRecord, fit_bounds
from cccplus.config import FrozenManifest, stable_seed
from cccplus.mechanisms import Mechanism, Site, orthonormalize, subspace_similarity
from cccplus.signature import (
    CausalSignature,
    assert_comparable,
    compare_signatures,
    d_mag,
    d_shape,
    signature_weights,
    weighted_norm,
)
from cccplus.stats import auroc, fa_at_matched_retention, holm, retention
from cccplus.translators.base import split_documents
from cccplus.validation import RULES, PathMeasurement, ReturnLeg

MF = FrozenManifest.load()


# ----------------------------------------------------------------- primitives
def test_orthonormalize_and_rank():
    V = torch.randn(16, 3)
    Q = orthonormalize(V)
    assert Q.shape == (16, 3)
    assert torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5)
    # rank-deficient input is reduced to its numerical rank
    W = torch.cat([V[:, :1], V[:, :1] * 2.0], dim=1)
    assert orthonormalize(W).shape[1] == 1


def test_projector_is_basis_invariant():
    """Eq. 6 depends on V V^T, so a change of basis must not change the projector."""
    V = torch.randn(12, 4)
    m1 = Mechanism(Site(0, "mlp_out"), V)
    O, _ = torch.linalg.qr(torch.randn(4, 4))
    m2 = Mechanism(Site(0, "mlp_out"), orthonormalize(V) @ O)
    assert torch.allclose(m1.projector(), m2.projector(), atol=1e-5)
    assert m1.fingerprint() == m2.fingerprint()


def test_d_shape_is_scale_invariant_and_d_mag_is_not():
    w = signature_weights(2, 20)
    # Seeded: an unseeded draw made this test fail on ~20% of runs, which reads as a
    # method defect rather than a test defect.
    g = torch.Generator().manual_seed(0)
    s = torch.randn(40, dtype=torch.float64, generator=g)
    t = s * 3.0
    assert d_shape(s, t, w, 1e-9) < 1e-8          # same direction
    assert abs(d_mag(s, t, w, 1e-9) - np.log(3.0)) < 1e-6
    # Symmetric up to floating point: d_mag is a log ratio, so the two argument orders
    # round differently. Asserting exact equality over-specifies the contract.
    assert abs(d_mag(s, t, w, 1e-9) - d_mag(t, s, w, 1e-9)) < 1e-12


def test_d_shape_detects_heterogeneity_a_mean_cannot():
    """The motivating case: equal means, different prompt-level pattern."""
    a = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    b = torch.tensor([2.0, 0.0, 2.0, 0.0], dtype=torch.float64)
    assert abs(float(a.mean()) - float(b.mean())) < 1e-12      # means agree exactly
    w = signature_weights(1, 4)
    assert d_shape(a, b, w, 1e-9) > 0.3                        # shapes do not


def test_weighted_norm_weights_sum_to_one():
    for scheme in ("uniform", "setting_balanced"):
        w = signature_weights(3, 7, scheme)
        assert abs(float(w.sum()) - 1.0) < 1e-12


# ------------------------------------------------------- regression: defects
def test_regression_source_gate_cannot_be_bypassed():
    """Defect 1: every rule must enforce source eligibility (Sec. 3.1)."""
    m = PathMeasurement(
        program="p", task="t", source_model="A", dest_model="B", translator="procrustes",
        seed=0, source_mechanism="m", case="shared", label=1, abstained=False,
        mean_effect_source=-2.4, mean_effect_dest=-2.3, sign_agrees=True, tau_dest=0.5,
        d_shape_forward=0.0, d_mag_forward=0.0, representational=1.0,
        bounds={"eps_shape": 1.0, "eps_mag": 1.0, "eps_scalar": 1.0, "eps_representational": 0.1},
    )
    m.ret_independent = ReturnLeg(available=True, scalar_error=0.0, d_shape=0.0, d_mag=0.0)
    m.ret_coupled = ReturnLeg(available=True, scalar_error=0.0, d_shape=0.0, d_mag=0.0)
    m.source_gate_passed = True
    assert all(RULES[r].accepts(m) for r in
               ("forward_only", "scalar_ccc", "ccc_sig", "pairwise_ccc_plus")), "should accept"
    m.source_gate_passed = False
    for r in RULES:
        assert not RULES[r].accepts(m), f"{r} accepted a path whose source was never established"


def test_regression_representational_threshold_respects_fa_budget():
    """Defect 3: for a similarity rule, the FA budget is a *lower* bound."""
    rng = np.random.default_rng(0)
    negs = [
        NegativeRecord("p", "B", "t", d_shape=1.0, d_mag=1.0, scalar_error=1.0,
                       representational=float(v))
        for v in rng.uniform(0.2, 0.95, size=200)
    ]
    # controls claim equivalent pairs sit low, which must not drag the threshold down
    ctrls = [ControlRecord("identity", "p", "B", "t", d_shape=0.05, d_mag=0.02,
                           scalar_error=0.05, representational=0.30) for _ in range(40)]
    b = fit_bounds(ctrls, negs, {"B|t": [1.0]}, MF)
    fa = float(np.mean([n.representational >= b.eps_representational for n in negs]))
    budget = float(MF.path_of("equivalence_bounds.max_calibration_false_acceptance"))
    assert fa <= budget + 0.02, f"calibration FA {fa:.3f} exceeds budget {budget}"


def test_regression_document_split_is_disjoint_and_respects_documents():
    """Defect 4: whole documents go to one side; rows never overlap."""
    f, r = split_documents(200, 0.5, seed=1)
    assert set(f).isdisjoint(set(r)) and len(f) + len(r) == 200
    docs = np.repeat(np.arange(20), 10)          # 20 real documents of 10 rows
    f, r = split_documents(200, 0.5, seed=1, document_ids=docs)
    assert set(f).isdisjoint(set(r))
    assert set(docs[f]).isdisjoint(set(docs[r])), "a document was split across both legs"


def test_regression_signature_comparison_validates_coordinates():
    """Defect 7: mismatched settings or prompt sets must raise, not return a number."""
    e = torch.randn(2, 10, dtype=torch.float64)
    a = CausalSignature(["k1", "k2"], e, 1.0, "A", "m", prompt_key="aaa")
    b = CausalSignature(["k1", "k2"], e.clone(), 1.0, "B", "m", prompt_key="aaa")
    assert_comparable(a, b)                                   # fine
    for bad in (
        CausalSignature(["k1", "kX"], e.clone(), 1.0, "B", "m", prompt_key="aaa"),
        CausalSignature(["k1", "k2"], e.clone(), 1.0, "B", "m", prompt_key="bbb"),
        CausalSignature(["k1", "k2"], torch.randn(2, 9, dtype=torch.float64), 1.0, "B", "m", "aaa"),
    ):
        try:
            assert_comparable(a, bad)
        except ValueError:
            continue
        raise AssertionError("incomparable signatures were accepted")


def test_regression_stable_seed_is_process_independent():
    """Defect 6: baseline seeding must not use Python's randomised hash()."""
    assert stable_seed("mas", "L1.mlp_out@final") == stable_seed("mas", "L1.mlp_out@final")
    assert stable_seed(0, "a") != stable_seed(0, "b")
    # value is pinned, so a different process cannot produce a different stream
    assert stable_seed("mas", "L1.mlp_out@final") == 3012460746


def test_regression_leave_one_program_out_excludes_held_out_norms():
    from cccplus.calibration import Calibrator

    cal = Calibrator(MF)
    cal.add_gated_norm("progA/src/M1", "t", 1.0)
    cal.add_gated_norm("progB/src/M1", "t", 2.0)
    b = cal.leave_one_program_out("progA")
    assert "progA/src/M1|t" not in b.scale
    assert "progB/src/M1|t" in b.scale


# ------------------------------------------------------------------ statistics
def test_auroc_matches_known_values():
    assert abs(auroc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) - 1.0) < 1e-9
    assert abs(auroc(np.array([4.0, 3.0, 2.0, 1.0]), np.array([0, 0, 1, 1])) - 0.0) < 1e-9
    assert abs(auroc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 1])) - 0.5) < 1e-9


def test_matched_retention_is_monotone_and_hits_target():
    rng = np.random.default_rng(0)
    err = np.concatenate([rng.uniform(0, 1, 100), rng.uniform(0.5, 2, 300)])
    lab = np.concatenate([np.ones(100, int), np.zeros(300, int)])
    prev = -1.0
    for target in (0.5, 0.7, 0.9, 0.99):
        fa = fa_at_matched_retention(err, lab, target)
        assert fa >= prev - 1e-9, "FA must not fall as retention rises"
        prev = fa
    from cccplus.stats import threshold_for_retention
    thr = threshold_for_retention(err, lab, 0.9)
    assert retention(err, lab, thr) >= 0.9 - 1e-9


def test_holm_is_monotone_and_bounded():
    adj = holm([0.001, 0.02, 0.5])
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[0] <= adj[1] <= adj[2]
    assert adj[0] >= 0.001


def _measurement(dest: str, case: str, label: int, mech: str = "m0") -> PathMeasurement:
    m = PathMeasurement(
        program="p", task="t", source_model="A", dest_model=dest, translator="procrustes",
        seed=0, source_mechanism=mech, case=case, label=label, abstained=False,
        mean_effect_source=-2.0, mean_effect_dest=-2.0, sign_agrees=True, tau_dest=0.5,
        d_shape_forward=0.0, d_mag_forward=0.0, representational=1.0,
        bounds={"eps_shape": 1.0, "eps_mag": 1.0, "eps_scalar": 1.0, "eps_representational": 0.1},
    )
    m.ret_independent = ReturnLeg(available=True, scalar_error=0.0, d_shape=0.0, d_mag=0.0)
    m.ret_coupled = ReturnLeg(available=True, scalar_error=0.0, d_shape=0.0, d_mag=0.0)
    m.source_gate_passed = True
    return m


def test_regression_multi_model_pairs_are_coherent():
    """Defect: conjunction items cross-paired unrelated candidate kinds.

    The planted case of a true counterpart differs by destination (``permuted`` in one
    model, ``merged`` in another), so pairing on ``case`` destroys every positive.
    Pairing without any constraint is worse: it mixes a true counterpart in one
    destination with a decoy in the other, giving an item whose label describes
    neither leg, and reuses one positive leg across many negative items.
    """
    from cccplus.analysis import build_multi_items

    records = [
        _measurement("B", "permuted", 1),
        _measurement("C", "merged", 1),
        _measurement("B", "wrong_variable", 0),
        _measurement("C", "wrong_variable", 0),
        _measurement("B", "effect_matched_random", 0),
    ]
    items, pairs = build_multi_items(records, ["pairwise_ccc_plus", "multi_model_ccc_plus"])

    # positives survive across differing planted cases
    assert int((items.labels == 1).sum()) == 1, "a true counterpart in each destination"
    # negatives pair only like with like
    assert int((items.labels == 0).sum()) == 1, "only wrong_variable+wrong_variable"
    for p in pairs:
        assert p.left.dest_model != p.right.dest_model, "a conjunction needs two destinations"
        if p.label == 1:
            assert p.left.label == 1 and p.right.label == 1
        else:
            assert p.left.case == p.right.case, "mixed-kind pair leaked into the item set"
    # no positive leg may be reused as part of a negative item
    neg_legs = [id(x) for p in pairs if p.label == 0 for x in (p.left, p.right)]
    pos_legs = {id(x) for p in pairs if p.label == 1 for x in (p.left, p.right)}
    assert not (set(neg_legs) & pos_legs), "a positive leg was reused inside a negative item"


def test_regression_set_valued_recall_is_not_a_per_path_mean():
    """Defect: recall and precision were both the mean per-path similarity.

    Being identical by construction, they could not express the thing they exist to
    measure: a variable split over two planted blocks is only half recovered when only
    one block is accepted.
    """
    from cccplus.analysis import set_valued_stats

    def leg(case, label, idx, sim, n_blocks=2, dest="B"):
        m = _measurement(dest, case, label)
        m.truth_subspace_similarity = sim
        m.diagnostics["n_truth_blocks"] = float(n_blocks)
        m.diagnostics["truth_block_index"] = float(idx)
        return m

    # one context, two planted blocks, only the first accepted
    half = set_valued_stats([leg("split", 1, 0, 1.0)])["split"]
    assert abs(half["subspace_recall"] - 0.5) < 1e-9, "one of two blocks is half recall"
    assert abs(half["subspace_precision"] - 1.0) < 1e-9, "the accepted block was correct"
    assert half["subspace_recall"] != half["subspace_precision"], \
        "recall and precision must be able to differ"
    assert half["set_exact"] == 0.0, "an uncovered block is not exact recovery"

    # both blocks accepted -> full recovery
    full = set_valued_stats([leg("split", 1, 0, 1.0), leg("split", 1, 1, 1.0)])["split"]
    assert abs(full["subspace_recall"] - 1.0) < 1e-9
    assert full["set_exact"] == 1.0

    # a spurious acceptance covering no planted block costs precision, not recall
    spurious = set_valued_stats([
        leg("split", 1, 0, 1.0), leg("split", 1, 1, 1.0), leg("wrong_variable", 0, -1, 0.0),
    ])["split"]
    assert abs(spurious["subspace_recall"] - 1.0) < 1e-9, "recall unaffected by a false positive"
    assert spurious["subspace_precision"] < 1.0, "precision must fall"
    assert spurious["set_exact"] == 0.0


def test_regression_both_bound_modes_are_reportable():
    """Defect: only the primary bound mode was ever written out.

    The bounds are normalisers of already-measured distances, so the second mode must
    be obtainable by rescoring the same measurements -- otherwise the two modes are not
    comparable. Rescoring must change acceptance, leave the raw measurement alone, and
    leave tau_min (not a function of the mode) alone.
    """
    from cccplus.analysis import rescore_with_bounds

    class _B:
        eps_shape = 0.10
        eps_mag = 0.10
        eps_scalar = 0.10
        eps_representational = 0.10
        provenance: dict = {}

    m = _measurement("B", "shared", 1)
    m.d_shape_forward = 0.25          # inside a loose bound, outside a tight one
    m.d_mag_forward = 0.25
    m.bounds = {"eps_shape": 1.0, "eps_mag": 1.0, "eps_scalar": 1.0,
                "eps_representational": 0.1}
    assert RULES["ccc_sig"].accepts(m), "loose bounds should accept"

    tight = rescore_with_bounds([m], {"p": _B()})[0]
    assert not RULES["ccc_sig"].accepts(tight), "tight bounds should reject"
    assert tight.d_shape_forward == m.d_shape_forward, "rescoring must not re-measure"
    assert tight.tau_dest == m.tau_dest, "tau_min is not a function of the bound mode"
    assert RULES["ccc_sig"].accepts(m), "the original record must be left untouched"


def test_regression_cluster_levels_come_from_the_manifest():
    """Defect: the manifest advertised four cluster levels; the code hardcoded three.

    The manifest must drive the clustering, and a level the path-level bootstrap cannot
    form must fail loudly rather than be dropped in silence -- otherwise the recorded
    configuration overstates the resampling actually performed.
    """
    from cccplus.analysis import CLUSTER_EXTRACTORS, build_item_set, resolve_cluster_levels

    levels = resolve_cluster_levels(MF)
    assert levels, "manifest must yield at least one level"
    assert all(lv in CLUSTER_EXTRACTORS for lv in levels)

    records = [_measurement("B", "shared", 1), _measurement("C", "wrong_variable", 0)]
    items = build_item_set(records, ["ccc_sig"], levels)
    assert len(items.clusters) == len(levels), "one cluster array per declared level"

    class _Fake:
        @staticmethod
        def path_of(key, default=None):
            return ["stratum", "family"] if key == "statistics.cluster_levels" else default

    try:
        resolve_cluster_levels(_Fake())
    except ValueError as exc:
        assert "family" in str(exc) and "prompt" in str(exc).lower()
    else:
        raise AssertionError("an unsupported cluster level must raise, not be ignored")


def test_regression_behavior_supervised_baselines_are_wired():
    """Defect: MAS / Latent Stitch / Stitch were implemented but unreachable.

    Part B filtered on a hardcoded tuple, ``fit_causal_translator`` was called from
    nowhere, and the ``causal_contexts`` parameter that would feed it was accepted and
    ignored -- so three of the paper's six translators could never run.
    """
    from cccplus.pipelines.evaluate import ControlledEvaluator
    from cccplus.translators import BEHAVIOR_SUPERVISED, REGISTRY, UNSUPERVISED

    # the registry is partitioned, and the hardcoded tuple is gone
    assert set(UNSUPERVISED) | set(BEHAVIOR_SUPERVISED) == set(REGISTRY)
    assert not (set(UNSUPERVISED) & set(BEHAVIOR_SUPERVISED))
    for name in BEHAVIOR_SUPERVISED:
        assert getattr(REGISTRY[name], "is_behavior_supervised", False), name

    # every supervised translator has a seed budget declared in the manifest
    seeds = MF.path_of("controlled.translator_seeds")
    for name in BEHAVIOR_SUPERVISED:
        assert seeds.get(name), f"{name} has no declared seeds"

    # the supervised path is reachable and fits on the hyperparameter split only
    for attr in ("causal_context", "supervised_pair", "measure_supervised_paths"):
        assert hasattr(ControlledEvaluator, attr), f"missing {attr}"
    # Inspect the code, not the prose: the docstring legitimately names the splits it
    # avoids, so only string constants in the executable body are checked.
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(ControlledEvaluator.causal_context)))
    fn = tree.body[0]
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    literals = {n.value for stmt in body for n in ast.walk(stmt)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    splits = literals & {"discovery", "hyperparameter", "calibration", "test"}
    assert splits == {"hyperparameter"}, \
        f"supervised fits must use the hyperparameter split alone, found {sorted(splits)}"


def test_ladder_is_cumulative():
    """Each rung must add conditions, never drop one."""
    order = ["forward_only", "scalar_ccc", "ccc_ind", "ccc_shape", "ccc_sig", "pairwise_ccc_plus"]
    for a, b in zip(order, order[1:]):
        na, nb = len(RULES[a].conditions), len(RULES[b].conditions)
        assert nb >= na, f"{b} has fewer conditions than {a}"
    assert RULES["multi_model_ccc_plus"].needs_two_destinations


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
