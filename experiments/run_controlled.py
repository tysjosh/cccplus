"""Controlled ground-truth stage: the primary methodological test (Sec. 4.1, 5.1, 5.3).

Order of operations is fixed so that no test label can influence a threshold:

  1. build the planted models, prompts and family-disjoint splits;
  2. collect calibration controls -> tau_min and c_{M,T};
  3. measure calibration-split paths -> labelled calibration negatives -> eps bounds
     (both pooled and leave-one-program-out);
  4. only then measure test-split paths and evaluate every rule on them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cccplus.analysis import (
    analyse,
    coupling_stats,
    rescore_with_bounds,
    set_valued_stats,
    signature_agreement,
)
from cccplus.benchmarks.programs import get_program
from cccplus.benchmarks.siit import load_siit
from cccplus.calibration import Bounds, Calibrator
from cccplus.config import FrozenManifest, set_all_seeds, stable_seed, write_json
from cccplus.interventions import Setting, settings_from_manifest
from cccplus.pipelines.controlled import add_calibration_records, build_bundle
from cccplus.pipelines.evaluate import (
    ControlledEvaluator, EvalConfig, negatives_from, positives_from,
)
from cccplus.reporting import (
    Table,
    benchmark_table,
    contrast_table,
    coupling_table,
    rule_table,
    set_valued_table,
    signature_table,
    translator_table,
    write_tables,
)
from cccplus.validation import LADDER, RULES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--programs", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--boot", type=int, default=None, help="override bootstrap replicates")
    ap.add_argument("--calibration-mode", default="leave_one_program_out",
                    choices=["leave_one_program_out", "pooled"],
                    help="whether eps bounds are fitted holding out the evaluated program")
    ap.add_argument("--bound-mode", default=None,
                    choices=["controls_only", "cycle_aware"],
                    help="eps estimator; defaults to the manifest's primary_mode")
    ap.add_argument("--skip-part-b", action="store_true")
    ap.add_argument("--part-c", action="store_true",
                    help="also run the behaviour-supervised baselines (MAS, Latent Stitch, "
                         "Stitch). Off by default: each candidate is a per-site gradient "
                         "optimisation through the destination model, so this costs far more "
                         "than the unsupervised translators. Reported separately because "
                         "these baselines have no coupled return leg.")
    ap.add_argument("--intervention", default="counterfactual_patch",
                    choices=["counterfactual_patch", "mean_ablation", "norm_matched_steering"],
                    help="Sec. 3.2: the counterfactual patch is the primary signature; the "
                         "other arms are robustness checks and are never pooled with it")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    mf = FrozenManifest.load(args.manifest)
    master = int(mf.path_of("seeds.master"))
    set_all_seeds(master)
    cache = mf.out_dir("cache_dir") / "controlled"
    programs = args.programs or list(mf.path_of("controlled.programs"))
    instances = list(mf.path_of("controlled.instances").keys())
    settings = settings_from_manifest(mf, kinds=(args.intervention,))
    if args.bound_mode is None:
        args.bound_mode = str(mf.path_of("equivalence_bounds.primary_mode"))
    ws = str(mf.path_of("signature.weights"))
    eta = float(mf.path_of("signature.eta"))
    seeds_by_name: Dict[str, List[int]] = mf.path_of("controlled.translator_seeds")
    part_a_translators = list(mf.path_of("controlled.rule_discrimination_translators"))
    cfg = EvalConfig(settings=settings, weight_scheme=ws, eta=eta,
                     reverse_anchor=str(mf.path_of("controlled.reverse_site_policy")) == "anchor")

    t0 = time.time()
    print(f"CCC+ controlled stage | manifest {mf.hash} | {len(programs)} programs", flush=True)
    print(f"settings A = {[s.key for s in settings]}", flush=True)

    # ------------------------------------------------ 1-2: models + calibration
    cal = Calibrator(mf)
    bundles = {}
    evaluators = {}
    benchmark_summary: Dict[str, Dict[str, dict]] = {}
    for prog_name in programs:
        program = get_program(prog_name)
        models = {}
        for inst in instances:
            p = cache / f"{prog_name}__{inst}.pt"
            if not p.exists():
                print(f"  ! missing {p.name}; run experiments/train_controlled.py first")
                return 2
            sm = load_siit(p)
            # Model names must be unique per trained model: c_{M,T} and tau_min are
            # per (model, task), and three programs reusing the name 'M1' would pool
            # scales across unrelated models.
            sm.model.name = f"{prog_name}/{inst}/{sm.name}"
            models[inst] = sm
        benchmark_summary[prog_name] = {
            inst: {
                "arch": sm.name, "structure": sm.placement.structure,
                **{k: sm.metrics.to_dict()[k] for k in
                   ("clean_accuracy", "min_iit_accuracy", "mean_strict_accuracy")},
                "meets_gate": bool(sm.metrics.to_dict()["clean_accuracy"] >= 0.95
                                   and sm.metrics.to_dict()["min_iit_accuracy"] >= 0.90),
            }
            for inst, sm in models.items()
        }
        bundle = build_bundle(program, models, mf, master)
        bundles[prog_name] = bundle
        for inst in instances:
            add_calibration_records(cal, bundle, inst, settings, ws, eta, master)
        evaluators[prog_name] = ControlledEvaluator(
            bundle, mf, cfg, master, cache_dir=mf.out_dir("cache_dir") / "translators"
        )
        print(f"  [{prog_name}] models + calibration controls ready "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"calibration controls: {cal.summary()}", flush=True)

    # ------------------------------------- 3: calibration-split paths -> eps
    partial = cal.pooled()   # tau_min and c_{M,T} are final here; eps is provisional
    cal_records = []
    for prog_name in programs:
        recs = evaluators[prog_name].measure_rule_discrimination(
            "calibration", partial, part_a_translators, seeds_by_name
        )
        cal_records.extend(recs)
        print(f"  [{prog_name}] calibration paths: {len(recs)}", flush=True)
    for nr in negatives_from(cal_records):
        cal.add_negative(nr)
    for pr in positives_from(cal_records):
        cal.add_positive(pr)
    cal.mode = args.bound_mode
    for mode in list(mf.path_of("equivalence_bounds.modes")):
        bb = cal.pooled(mode)
        print(f"bounds ({mode}): eps_shape={bb.eps_shape:.4f} eps_mag={bb.eps_mag:.4f} "
              f"eps_scalar={bb.eps_scalar:.4f} eps_repr={bb.eps_representational:.4f}"
              + ("  [negative cap conflicts with positive floor]"
                 if bb.provenance.get("shape", {}).get("cap_conflicts_with_floor") else ""),
              flush=True)
    pooled_bounds = cal.pooled(args.bound_mode)

    bounds_by_program = {
        p: cal.bounds_for(p, split_mode=args.calibration_mode, mode=args.bound_mode)
        for p in programs
    }

    # ------------------------------------------------- 4: test-split evaluation
    part_a: List = []
    part_b: List = []
    part_c: List = []
    for prog_name in programs:
        b = bounds_by_program[prog_name]
        recs = evaluators[prog_name].measure_rule_discrimination(
            "test", b, part_a_translators, seeds_by_name
        )
        part_a.extend(recs)
        print(f"  [{prog_name}] test paths (part A): {len(recs)}  "
              f"eps_shape={b.eps_shape:.4f} eps_mag={b.eps_mag:.4f}", flush=True)
        if not args.skip_part_b:
            recs_b = evaluators[prog_name].measure_translator_paths(
                "test", b, seeds_by_name, include_shuffled=True
            )
            part_b.extend(recs_b)
            print(f"  [{prog_name}] test paths (part B): {len(recs_b)}", flush=True)
        if args.part_c:
            recs_c = evaluators[prog_name].measure_supervised_paths("test", b, seeds_by_name)
            part_c.extend(recs_c)
            print(f"  [{prog_name}] test paths (part C, supervised): {len(recs_c)}", flush=True)

    # ------------------------------------------------------------- 5: analysis
    print(f"\nanalysing {len(part_a)} part-A paths ...", flush=True)
    res_a = analyse(part_a, mf, rules=LADDER, n_boot=args.boot, seed=int(mf.path_of("seeds.bootstrap")))
    couple = coupling_stats(part_a)
    setval = set_valued_stats(part_a)
    sigagree = signature_agreement(part_b if part_b else part_a)

    res_b = None
    if part_b:
        print(f"analysing {len(part_b)} part-B paths ...", flush=True)
        res_b = analyse(part_b, mf, rules=[r for r in LADDER if r != "multi_model_ccc_plus"],
                        n_boot=args.boot, seed=int(mf.path_of("seeds.bootstrap")) + 3)

    # Part C is reported on its own. These baselines have no coupled return
    # construction, so rules requiring one are excluded rather than scored as failures:
    # "the construction does not exist" and "the construction did not hold" are
    # different findings and must not share a cell.
    res_c = None
    COUPLED_RULES = {"round_trip_only", "scalar_ccc"}
    part_c_rules = [r for r in LADDER
                    if r != "multi_model_ccc_plus" and r not in COUPLED_RULES]
    if part_c:
        print(f"analysing {len(part_c)} part-C paths (behaviour-supervised) ...", flush=True)
        res_c = analyse(part_c, mf, rules=part_c_rules, n_boot=args.boot,
                        seed=int(mf.path_of("seeds.bootstrap")) + 11)

    # ------------------------------------------- 5b: the other bound mode
    # Sec. 3.6 is reported under *both* estimators. The bounds are only normalisers of
    # already-measured distances, so the alternate mode rescores the same paths: the
    # candidates, signatures and tau_min are identical and only the thresholds move.
    alt_results: Dict[str, dict] = {}
    alt_bounds_by_program: Dict[str, Dict[str, Bounds]] = {}
    for mode in [m for m in mf.path_of("equivalence_bounds.modes") if m != args.bound_mode]:
        ab = {p: cal.bounds_for(p, split_mode=args.calibration_mode, mode=mode) for p in programs}
        alt_bounds_by_program[mode] = ab
        print(f"analysing part A under bound mode '{mode}' ...", flush=True)
        alt_results[mode] = analyse(
            rescore_with_bounds(part_a, ab), mf, rules=LADDER, n_boot=args.boot,
            seed=int(mf.path_of("seeds.bootstrap")) + 7,
        )

    # ------------------------------------------------------------- 6: outputs
    out = mf.out_dir()
    tag = args.tag or ("" if args.intervention == "counterfactual_patch" else f"_{args.intervention}")
    write_json(out / f"controlled_part_a_paths{tag}.json",
               {"records": [m.to_dict() for m in part_a]}, mf)
    if part_b:
        write_json(out / f"controlled_part_b_paths{tag}.json",
                   {"records": [m.to_dict() for m in part_b]}, mf)
    if part_c:
        write_json(out / f"controlled_part_c_paths{tag}.json",
                   {"records": [m.to_dict() for m in part_c]}, mf)
    for mode, res in alt_results.items():
        write_json(out / f"controlled_analysis{tag}__{mode}.json", {
            "part_a": _jsonable(res),
            "bound_mode": mode,
            "secondary_to": args.bound_mode,
            "bounds_by_program": {k: v.to_dict() for k, v in alt_bounds_by_program[mode].items()},
            "calibration_mode": args.calibration_mode,
            "intervention": args.intervention,
            "note": "rescored from the same measurements as the primary mode; "
                    "candidates, signatures and tau_min are identical",
        }, mf)

    write_json(out / f"controlled_analysis{tag}.json", {
        "part_a": _jsonable(res_a),
        "part_b": _jsonable(res_b) if res_b else None,
        "part_c": _jsonable(res_c) if res_c else None,
        "part_c_rules": part_c_rules if res_c else None,
        "part_a_by_bound_mode": {m: _jsonable(r) for m, r in alt_results.items()},
        "coupling": couple,
        "set_valued": setval,
        "signature_agreement": sigagree,
        "bounds_pooled": pooled_bounds.to_dict(),
        "bounds_by_program": {k: v.to_dict() for k, v in bounds_by_program.items()},
        "calibration_summary": cal.summary(),
        "calibration_mode": args.calibration_mode,
        "bound_mode": args.bound_mode,
        "matched_retention_target": res_a.get("matched_retention_target"),
        "settings": [s.key for s in settings],
        "intervention": args.intervention,
    }, mf)

    tables = [
        benchmark_table("benchmark_validity", benchmark_summary,
                        "Planted benchmark validity. Clean accuracy, worst-case interchange "
                        "accuracy over planted groups, and strict-localisation accuracy. The "
                        "planted causal graph is only usable as ground truth where all three "
                        "are high."),
        rule_table("table_5_1_controlled", res_a["per_rule"], LADDER,
                   "Controlled ground-truth results. Ret. is positive retention, Cov. is "
                   "candidate coverage, FA is false acceptance; 95% hierarchical-bootstrap "
                   "intervals in brackets. Every rule is evaluated on the same candidates.",
                   notes=f"n = {res_a['n_items']} paths "
                         f"({res_a['n_positives']} positive, {res_a['n_negatives']} negative); "
                         f"multi-model row uses {res_a['n_multi_items']} two-destination items "
                         f"({res_a['n_multi_positives']} positive). "
                         f"Calibration: {args.calibration_mode}/{args.bound_mode}; "
                         f"FA matched at retention {res_a.get('matched_retention_target')}. "
                         + _conflict_note(bounds_by_program, args.bound_mode)),
        contrast_table("table_5_1b_contrasts", res_a["contrasts"],
                       "Planned paired contrasts on false acceptance at matched positive "
                       "retention. Negative estimates favour the first rule. Holm correction "
                       "covers the seven planned contrasts."),
        coupling_table("coupled_vs_independent", couple,
                       "Coupled versus independently estimated return legs."),
        set_valued_table("set_valued_cases", setval,
                         "Planted case types. Split and merged variables are set-valued: a "
                         "single component cannot be the whole counterpart."),
        signature_table("table_5_4_signatures", sigagree,
                        list(sigagree.keys()),
                        "Prompt-level causal agreement for accepted correspondences. Corr. is "
                        "effect correlation, Sign is prompt-level sign agreement."),
    ]
    if res_b:
        tables.append(
            rule_table("part_b_translator_rules", res_b["per_rule"],
                       [r for r in LADDER if r != "multi_model_ccc_plus"],
                       "Translator-proposed candidates. Coverage and abstention are measured "
                       "here, since candidates come from the translators themselves.",
                       notes=f"n = {res_b['n_items']} paths "
                             f"({res_b['n_positives']} positive, {res_b['n_negatives']} negative).")
        )
    if res_c:
        tables.append(
            rule_table("part_c_behavior_supervised", res_c["per_rule"], part_c_rules,
                       "Behaviour-supervised baselines (MAS, Latent Stitch, Stitch). These "
                       "learn an alignment from task labels rather than from representational "
                       "structure, and are reported apart from the unsupervised translators "
                       "for that reason.",
                       notes=f"n = {res_c['n_items']} paths "
                             f"({res_c['n_positives']} positive, {res_c['n_negatives']} "
                             f"negative). Fitted on the hyperparameter split only. Rules "
                             f"requiring a coupled return leg "
                             f"({', '.join(sorted(COUPLED_RULES))}) are omitted: these "
                             f"baselines have no coupled construction, so scoring them there "
                             f"would report an absent construction as a failed one.")
        )

    for mode, res in alt_results.items():
        tables.append(
            rule_table(f"table_5_1_controlled__{mode}", res["per_rule"], LADDER,
                       f"Secondary bound mode '{mode}'. Same candidates, same signatures, "
                       f"same tau_min as the primary table; only the equivalence bounds "
                       f"differ, so the two are directly comparable.",
                       notes=f"n = {res['n_items']} paths. "
                             f"Calibration: {args.calibration_mode}/{mode}. "
                             + _conflict_note(alt_bounds_by_program[mode], mode))
        )

    fa_rows = [[k] + [f"{v.get(c, float('nan')):.3f}" for c in
                      ("wrong_variable", "effect_matched_random", "absent", "split_half")]
               for k, v in res_a["fa_by_case"].items()]
    tables.append(Table(
        "fa_by_negative_case", ["Validation rule", "wrong variable", "effect-matched random",
                                "absent", "split half"], fa_rows,
        "False acceptance by kind of negative. The negative classes are not equally hard: an "
        "absent mechanism has no causal effect at all, whereas an effect-matched random "
        "subspace is built to survive a mean-effect test."))

    title = "Controlled ground truth" + (f" ({args.intervention})" if tag else "")
    path = write_tables(tables, mf.out_dir("tables_dir"), mf.hash, title)
    print(f"\nwrote {path}")
    print(f"total {time.time()-t0:.0f}s")
    return 0


def _conflict_note(bounds_by_program: Dict[str, Bounds], mode: str) -> str:
    """State plainly when the positive floor overrode the false-acceptance cap.

    Under ``cycle_aware`` the bound is floored so that calibration positives survive.
    Where that floor exceeds the 5% false-acceptance cap, the floor wins and the
    calibrated thresholds are therefore *not* capped. Saying so in the table keeps the
    deviation visible next to the numbers it produced.
    """
    hits = []
    for prog, b in bounds_by_program.items():
        prov = getattr(b, "provenance", {}) or {}
        comps = [c for c in ("shape", "magnitude", "scalar")
                 if isinstance(prov.get(c), dict) and prov[c].get("cap_conflicts_with_floor")]
        if comps:
            hits.append(f"{prog} ({'/'.join(comps)})")
    if not hits:
        return "The false-acceptance cap held for every program."
    return (
        f"NOTE: under '{mode}' the positive-retention floor exceeded the "
        f"false-acceptance cap for {len(hits)} of {len(bounds_by_program)} programs, so "
        f"these thresholds are floored, not capped, and the realised FA may exceed the "
        f"5% calibration budget: " + "; ".join(hits) + ". The controls_only mode is "
        "reported alongside, and the matched-retention column is threshold-free."
    )


def _jsonable(res):
    if res is None:
        return None
    out = dict(res)
    out["per_rule"] = {r: {k: v.to_dict() for k, v in d.items()} for r, d in res["per_rule"].items()}
    out["contrasts"] = [
        {"a": c["a"], "b": c["b"], "unit": c.get("unit"), **c["estimate"].to_dict()}
        for c in res["contrasts"]
    ]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
