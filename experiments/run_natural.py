"""Natural-model stage: IOI and factual recall on pretrained LMs (Sec. 4.2, 5.2, 5.4).

Two phases, so that only one checkpoint needs to be resident at a time:

  phase 1  --phase corpus    load each model once, cache corpus activations to disk
  phase 2  --phase evaluate  fit translators from the cached activations, then run the
                             CCC+ protocol per source model
  --phase all                both, in order

Every checkpoint serves as source, giving six directed mappings per stratum with
three models, and each source's two destinations are what multi-model CCC+ needs.

Intended target: a single A100 (bf16). Example:

  python experiments/run_natural.py --stratum larger --phase all \
      --device cuda --dtype bfloat16 --threads 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cccplus.analysis import (
    analyse,
    coupling_stats,
    rescore_with_bounds,
    signature_agreement,
)
from cccplus.calibration import Calibrator, ControlRecord
from cccplus.config import FrozenManifest, set_all_seeds, stable_seed, write_json
from cccplus.interventions import CausalProbe, Setting, null_direction_mechanisms, settings_from_manifest
from cccplus.mechanisms import Mechanism, Site
from cccplus.models.hf_lm import HFCausalLM
from cccplus.pipelines.natural import (
    collect_corpus_activations,
    factual_source_mechanisms,
    ioi_source_mechanisms,
    load_store,
    permuted_destination_control,
    save_store,
    screen_factual_across_models,
    screen_ioi_across_models,
    SharedPrompts,
)
from cccplus.pipelines.evaluate import (
    SignatureCache,
    fit_translator_pair,
    negatives_from,
    positives_from,
)
from cccplus.reporting import (
    Table,
    coupling_table,
    rule_table,
    signature_table,
    translator_table,
    write_tables,
)
from cccplus.signature import CausalSignature, signature_weights, weighted_norm
from cccplus.stats import retention
from cccplus.tasks.base import partition_by_family
from cccplus.tasks.corpus import load_corpus
from cccplus.tasks.factual import FactualRecallTask, build_fact_set, factual_task_data
from cccplus.tasks.ioi import IOITask, build_ioi_examples, ioi_task_data
from cccplus.translators.base import split_documents
from cccplus.validation import LADDER, RULES, measure_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--stratum", default="tiny",
                    help="key under natural.executed_matrix, or 'smaller'/'larger' for paper_matrix")
    ap.add_argument("--phase", default="all", choices=["corpus", "evaluate", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=["ioi", "factual_recall"])
    ap.add_argument("--translators", nargs="*", default=["procrustes", "crosscoder", "dfc"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--corpus-passages", type=int, default=2000)
    ap.add_argument("--corpus-source", default="auto")
    ap.add_argument("--corpus-path", default=None)
    ap.add_argument("--site-depths", type=int, default=6,
                    help="number of relative depths in the preregistered site grid")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--max-source-mechanisms", type=int, default=24)
    ap.add_argument("--facts-source", default="auto", choices=["auto", "counterfact", "builtin"])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--free-between-models", action="store_true",
                    help="drop weights after each model in the corpus phase")
    ap.add_argument("--skip-self-translation", action="store_true",
                    help="omit the identity positive control (Sec. 4.2 has no ground truth, "
                         "so without it a zero retention cannot be attributed)")
    ap.add_argument("--fresh", action="store_true",
                    help="discard cached screening/gated mechanisms and any partially "
                         "measured mappings, and recompute phase 2 from scratch")
    ap.add_argument("--self-test", action="store_true",
                    help="substitute randomly initialised models of the same families, so the "
                         "whole runner can be exercised offline with no checkpoints")
    return ap.parse_args()


# Architecture family per matrix key, used only by --self-test.
_SELF_TEST_KINDS = {"pythia": "gpt_neox", "gpt2": "gpt2", "llama": "llama", "gemma": "llama"}


def open_model(args, key: str, repo: str) -> HFCausalLM:
    if args.self_test:
        kind = _SELF_TEST_KINDS.get(key, "llama")
        return HFCausalLM.from_random_config(kind, key, device="cpu")
    return HFCausalLM.load(repo, name=key, device=args.device, dtype=args.dtype,
                           trust_remote_code=args.trust_remote_code)


def _prefix_key(mf: FrozenManifest, args, task_name: str) -> str:
    """Identity of the deterministic, expensive prefix of phase 2 for one task.

    Screening and source-mechanism gating depend only on the manifest, the stratum, the
    cached activation stores and these CLI knobs -- not on translators or seeds. Anything
    that would change their result has to appear here, or a stale cache would be reused.
    """
    return str(stable_seed(
        mf.hash, args.stratum, task_name, args.site_depths, args.max_source_mechanisms,
        args.facts_source, bool(args.self_test), args.dtype, args.device.split(":")[0],
    ) % (2 ** 31))


def _load_prefix(cache: Path, key: str, task_name: str):
    p = cache / f"prefix_{task_name}_{key}.pt"
    if not p.exists():
        return None
    try:
        got = torch.load(p, map_location="cpu", weights_only=False)
        return got if got.get("version") == 1 else None
    except Exception as exc:  # noqa: BLE001
        print(f"  [prefix] ignoring unreadable cache ({exc})", flush=True)
        return None


def _save_prefix(cache: Path, key: str, task_name: str, payload: dict) -> None:
    p = cache / f"prefix_{task_name}_{key}.pt"
    try:
        torch.save({"version": 1, **payload}, p)
    except Exception as exc:  # noqa: BLE001
        print(f"  [prefix] could not cache prefix ({exc})", flush=True)


def _partial_path(cache: Path, args) -> Path:
    return cache / f"partial_{'_'.join(sorted(args.tasks))}.pt"


# Bumped from 1: a partial file now carries calibration-split records as well as test
# ones. A v1 file has no calibration paths, so reusing it would silently reproduce the
# controls-only bounds this change exists to fix; it is discarded instead.
_PARTIAL_VERSION = 2


def _load_partial(cache: Path, args):
    """Records from mappings already measured, so a late failure is not total."""
    p = _partial_path(cache, args)
    if not p.exists():
        return [], set(), {}, []
    try:
        got = torch.load(p, map_location="cpu", weights_only=False)
        if got.get("version") != _PARTIAL_VERSION:
            print(f"  [resume] discarding v{got.get('version')} partial file: it predates "
                  f"calibration-split measurement", flush=True)
            return [], set(), {}, []
        cal = list(got.get("cal_records", []))
        print(f"  [resume] {len(got['records'])} test and {len(cal)} calibration records "
              f"from {len(got['done'])} completed mappings", flush=True)
        return list(got["records"]), set(got["done"]), dict(got.get("task_reports", {})), cal
    except Exception as exc:  # noqa: BLE001
        print(f"  [resume] ignoring unreadable partial file ({exc})", flush=True)
        return [], set(), {}, []


def _save_partial(cache: Path, args, records, done, task_reports, cal_records) -> None:
    try:
        torch.save({"version": _PARTIAL_VERSION, "records": records, "done": sorted(done),
                    "task_reports": task_reports, "cal_records": cal_records},
                   _partial_path(cache, args))
    except Exception as exc:  # noqa: BLE001
        print(f"  [resume] could not write partial file ({exc})", flush=True)


def _fit_quality(pair) -> Dict[str, float]:
    """Translator fit diagnostics, recorded on every path the pair produces.

    Needed to tell two very different readings apart when the causal tests reject
    everything: a translator that genuinely found no correspondence, versus one that was
    never fitted well enough to find one. Reconstruction quality and held-out alignment
    are properties of the fit alone, so they are available without any ground truth.
    """
    # Key names differ by family: the dictionary methods report fraction-of-variance-
    # unexplained per side, Procrustes reports held-out alignment across its site maps.
    wanted = ("fvu_a", "fvu_b", "mean_heldout_cosine", "n_fit")
    out: Dict[str, float] = {}
    for leg, tr in (("fwd", pair.forward), ("rev", pair.reverse_independent)):
        info = getattr(tr, "fit_info", {}) or {}
        for k in wanted:
            v = info.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                out[f"{leg}_{k}"] = float(v)
    return out


def _fit_summary(records) -> Dict[str, Dict[str, float]]:
    """Mean fit diagnostics per translator, for the results file."""
    keys = ("fwd_fvu_a", "fwd_fvu_b", "rev_fvu_a", "rev_fvu_b",
            "fwd_mean_heldout_cosine", "rev_mean_heldout_cosine", "fwd_n_fit")
    out: Dict[str, Dict[str, float]] = {}
    by: Dict[str, list] = {}
    for r in records:
        by.setdefault(r.translator, []).append(r)
    for tname, rs in by.items():
        d: Dict[str, float] = {"n": float(len(rs))}
        for k in keys:
            vals = [r.diagnostics[k] for r in rs if k in r.diagnostics]
            if vals:
                d[k] = float(np.mean(vals))
        # the round trip's realised subspace recovery, which needs no labels
        cos = [r.ret_independent.subspace_cosine for r in rs
               if np.isfinite(getattr(r.ret_independent, "subspace_cosine", float("nan")))]
        if cos:
            d["return_subspace_cosine"] = float(np.mean(cos))
        out[tname] = d
    return out


def _permuted_control(dest_model, dest_mechs, candidate, seed: int):
    """Build the permuted-destination control inside the destination model.

    Preference order, all of which keep the control in the destination's own index space:

    1. one of the destination's own source-gated mechanisms, permuted to a different head
       at that model's layer. This is the Sec. 4.2 control as specified: task-active in
       the destination, different identity;
    2. failing that, a permutation of the translated candidate, which also lives in the
       destination;
    3. failing both, ``None``, which the protocol records as an abstention rather than
       inventing a subspace of the wrong dimension.
    """
    if dest_mechs:
        basis = dest_mechs[seed % len(dest_mechs)]
    elif candidate is not None:
        basis = candidate
    else:
        return None
    return permuted_destination_control(basis, dest_model, seed)


def resolve_matrix(mf: FrozenManifest, stratum: str) -> Dict[str, str]:
    ex = mf.path_of("natural.executed_matrix", {}) or {}
    if stratum in ex:
        return dict(ex[stratum])
    paper = mf.path_of("natural.paper_matrix", {}) or {}
    if stratum in paper:
        return dict(paper[stratum])
    raise KeyError(f"unknown stratum '{stratum}'; have {sorted(set(ex) | set(paper))}")


# ------------------------------------------------------------------ phase 1
def phase_corpus(args, mf, matrix, cache: Path) -> Dict[str, object]:
    corpus = load_corpus(args.corpus_passages, stable_seed(int(mf.path_of("seeds.master")), "corpus"),
                         source=args.corpus_source, path=args.corpus_path)
    print(f"corpus: {len(corpus)} passages, provenance={corpus.provenance}", flush=True)
    prov = {"corpus": corpus.provenance, "stores": {}}
    for key, repo in matrix.items():
        path = cache / f"store_{key}.pt"
        if path.exists():
            print(f"  [cached] activations for {key}", flush=True)
            continue
        t0 = time.time()
        model = open_model(args, key, repo)
        sites = model.site_grid(("attn_out", "mlp_out"), args.site_depths, token="final")
        store, docs = collect_corpus_activations(
            model, corpus, sites, batch_size=16,
            seed=stable_seed(int(mf.path_of("seeds.master")), "acts", key),
        )
        save_store(path, store, docs)
        prov["stores"][key] = {"repo": repo, "n_rows": store.n, "n_sites": len(sites),
                               "d_model": model.d_model, "n_layers": model.n_layers,
                               "seconds": round(time.time() - t0, 1)}
        print(f"  [{key}] {store.n} rows x {len(sites)} sites in {time.time()-t0:.0f}s", flush=True)
        if args.free_between_models:
            model.free()
            del model
    return prov


# ------------------------------------------------------------------ phase 2
def phase_evaluate(args, mf, matrix, cache: Path) -> int:
    master = int(mf.path_of("seeds.master"))
    settings = settings_from_manifest(mf)
    ws = str(mf.path_of("signature.weights"))
    eta = float(mf.path_of("signature.eta"))
    bound_mode = str(mf.path_of("equivalence_bounds.primary_mode"))
    fractions = mf.path_of("partitions.fractions")
    stores: Dict[str, object] = {}
    docs: Dict[str, np.ndarray] = {}
    for key in matrix:
        p = cache / f"store_{key}.pt"
        if not p.exists():
            print(f"  ! missing {p.name}; run --phase corpus first")
            return 2
        stores[key], docs[key] = load_store(p)

    if args.fresh:
        for stale in list(cache.glob("prefix_*.pt")) + list(cache.glob("partial_*.pt")):
            stale.unlink()
        print("  [resume] --fresh: cleared cached prefixes and partial records", flush=True)
    all_records, done_groups, task_reports, cal_records = _load_partial(cache, args)

    for task_name in args.tasks:
        print(f"\n=== {task_name} ===", flush=True)
        models: Dict[str, HFCausalLM] = {}
        for key, repo in matrix.items():
            models[key] = open_model(args, key, repo)

        prefix_key = _prefix_key(mf, args, task_name)
        cached_prefix = _load_prefix(cache, prefix_key, task_name)

        if task_name == "ioi":
            task = IOITask()
            cfg = mf.path_of("natural.ioi")
            n_pairs = 64 if args.self_test else int(cfg["analysis_pairs"])
            examples = build_ioi_examples(n_pairs, stable_seed(master, "ioi"))
            if cached_prefix is not None:
                shared = SharedPrompts(keep=list(cached_prefix["keep"]),
                                       report=dict(cached_prefix["report"]))
                print("  [prefix] reusing cached screening", flush=True)
            else:
                shared = screen_ioi_across_models(models, examples, task,
                                                 float(cfg["min_clean_accuracy"]))
            print(f"  screening: {shared.report}", flush=True)
            kept = [examples[i] for i in shared.keep]
            fam = np.asarray([e.name_group for e in kept])
            parts = partition_by_family(fam, fractions, stable_seed(master, "ioi_split"))
            data_for = lambda m, idx: ioi_task_data(m, [kept[i] for i in idx])
            provenance = {"screening": shared.report, "n_shared": len(kept)}
        else:
            task = FactualRecallTask()
            cfg = mf.path_of("natural.factual_recall")
            n_facts = 24 if args.self_test else int(cfg["facts"])
            facts, fprov = build_fact_set(n_facts, stable_seed(master, "facts"),
                                          source="builtin" if args.self_test else args.facts_source,
                                          cache_dir=str(cache))
            if cached_prefix is not None:
                shared = SharedPrompts(keep=list(cached_prefix["keep"]),
                                       report=dict(cached_prefix["report"]))
                print("  [prefix] reusing cached screening", flush=True)
            else:
                shared = screen_factual_across_models(models, facts, task)
            print(f"  facts: {fprov}\n  screening: {shared.report}", flush=True)
            kept = [facts[i] for i in shared.keep]
            fam = np.asarray([f"{f.relation}|{f.subject}" for f in kept])
            parts = partition_by_family(fam, fractions, stable_seed(master, "fr_split"))
            data_for = lambda m, idx: factual_task_data(m, [kept[i] for i in idx])
            provenance = {"facts": fprov, "screening": shared.report, "n_shared": len(kept)}

        if len(kept) < (4 if args.self_test else 16):
            print("  ! too few shared prompts survived screening; skipping task")
            task_reports[task_name] = {**provenance, "skipped": True}
            continue
        task_reports[task_name] = provenance

        cal = Calibrator(mf)
        probes: Dict[str, Dict[str, CausalProbe]] = {}
        mechs: Dict[str, List[Mechanism]] = {}

        # Probes are always rebuilt: they hold live models and task data. The cache covers
        # the part that costs real time -- one intervention per candidate mechanism, per
        # model, to apply the source gate.
        for key, m in models.items():
            probes[key] = {}
            for split in ("calibration", "test"):
                d = data_for(m, parts[split])
                probes[key][split] = CausalProbe(m, task, d, collect_kl=False)

        if cached_prefix is not None:
            mechs = {k: list(v) for k, v in cached_prefix["mechs"].items()}
            for rec in cached_prefix["controls"]:
                cal.add_control(rec)
            for gkey, norms in cached_prefix["gated_norms"].items():
                cal.gated_norms.setdefault(gkey, []).extend(list(norms))
            for key in matrix:
                print(f"  [prefix] [{key}] {len(mechs.get(key, []))} source-gated "
                      f"mechanisms (cached)", flush=True)

        for key, m in ({} if cached_prefix is not None else models).items():
            layers = sorted({s.layer for s in m.site_grid(("attn_out",), args.site_depths)})
            if task_name == "ioi":
                cand = ioi_source_mechanisms(m, layers=layers)
            else:
                cand = factual_source_mechanisms(m, probes[key]["calibration"].data, layers=layers)
            # source gate: keep the strongest mechanisms of the predicted sign
            scored = []
            for mech in cand:
                eff = float(probes[key]["calibration"].effect(mech, settings[0]).mean())
                if np.sign(eff) == np.sign(task.predicted_sign):
                    scored.append((abs(eff), mech))
            scored.sort(key=lambda t: -t[0])
            mechs[key] = [mm for _, mm in scored[: args.max_source_mechanisms]]
            print(f"  [{key}] {len(mechs[key])} source-gated mechanisms "
                  f"(of {len(cand)} candidates)", flush=True)

            for nm in null_direction_mechanisms(
                m, [mm.site for mm in cand[:4]] or [Site(0, "mlp_out", "final")],
                rank=1, n=8, seed=stable_seed(master, "null", key, task_name)
            ):
                e = float(probes[key]["calibration"].effect(nm, settings[0]).mean())
                cal.add_control(ControlRecord("null_direction", task_name, m.name, task.name,
                                              abs_mean_effect=abs(e)))
            for mech in mechs[key]:
                sig = CausalSignature.measure(probes[key]["calibration"], mech, settings, scale=1.0)
                w = signature_weights(sig.n_settings, sig.n_prompts, ws)
                cal.add_gated_norm(m.name, task.name, weighted_norm(sig.vector(), w))

        if cached_prefix is None:
            _save_prefix(cache, prefix_key, task_name, {
                "keep": list(shared.keep),
                "report": dict(shared.report),
                "provenance": provenance,
                "mechs": mechs,
                "controls": list(cal.controls),
                "gated_norms": {k: list(v) for k, v in cal.gated_norms.items()},
            })
            print(f"  [prefix] cached screening + gated mechanisms for {task_name}", flush=True)

        # Provisional bounds. tau_min and the response scale c_{M,T} are fitted from the
        # controls and are final here; only the eps thresholds are provisional, because
        # setting those needs labelled calibration-split paths that do not exist yet.
        provisional = cal.pooled(bound_mode)
        final_provisional = provisional      # alias used by the self-translation control
        sigcache = SignatureCache(settings)

        # ---- six directed mappings: every checkpoint is a source
        for src_key in matrix:
            dests = [k for k in matrix if k != src_key]
            for dst_key in dests:
                fwd_i, rev_i = split_documents(
                    stores[src_key].n, 0.5, stable_seed(master, "corpus_split", src_key, dst_key),
                    document_ids=docs[src_key],
                )
                for tname in args.translators:
                    for seed in args.seeds:
                        group = f"{task_name}|{src_key}->{dst_key}|{tname}|s{seed}"
                        if group in done_groups:
                            print(f"  [resume] skipping {group} (already measured)", flush=True)
                            continue
                        base_seed = stable_seed(master, src_key, dst_key, tname, seed) % (2**31)
                        pair = fit_translator_pair(
                            tname,
                            stores[src_key].subset(fwd_i), stores[dst_key].subset(fwd_i),
                            stores[src_key].subset(rev_i), stores[dst_key].subset(rev_i),
                            models[src_key].name, models[dst_key].name, base_seed, mf,
                        )
                        ind = pair.independence_report()
                        fit = _fit_quality(pair)
                        # Shuffled-pair control (Sec. 4.2): the destination activations are
                        # row-shuffled before fitting, so the pair cannot encode a real
                        # correspondence and *any* acceptance is a false acceptance. This is
                        # what gives false acceptance an absolute scale rather than only a
                        # rate against one plausible alternative.
                        shuf = fit_translator_pair(
                            tname,
                            stores[src_key].subset(fwd_i), stores[dst_key].subset(fwd_i),
                            stores[src_key].subset(rev_i), stores[dst_key].subset(rev_i),
                            models[src_key].name, models[dst_key].name, base_seed, mf,
                            shuffle_destination=True,
                        )

                        # Both splits are measured from the same fitted translators. The
                        # calibration split sets the eps thresholds; the test split carries
                        # the result and is re-scored once those thresholds are final.
                        for split in ("calibration", "test"):
                            src_probe = probes[src_key][split]
                            dst_probe = probes[dst_key][split]
                            sink = cal_records if split == "calibration" else all_records
                            for mech in mechs[src_key]:
                                src_sig = sigcache.measure(
                                    src_probe, mech,
                                    provisional.c(models[src_key].name, task.name),
                                    tag=f"{src_key}|{split}")
                                cand = pair.translate(mech)
                                shuf_cand = shuf.translate(mech)
                                for tr in (pair.reverse_independent, pair.reverse_coupled,
                                           shuf.reverse_independent, shuf.reverse_coupled):
                                    if tr is not None and hasattr(tr, "anchor_site"):
                                        tr.anchor_site = mech.site
                                # The permuted-destination control must be defined *in the
                                # destination model*: a different head at one of that
                                # model's own layers. Deriving it from the source mechanism
                                # would index the destination's layers with a source layer
                                # number.
                                ctrl_seed = stable_seed(master, "ctrl", src_key, dst_key,
                                                        mech.label)
                                ctrl = _permuted_control(
                                    models[dst_key], mechs[dst_key], cand, ctrl_seed
                                )
                                for case, candidate, label, back in (
                                    ("translated", cand, 1, pair),
                                    ("permuted_destination", ctrl, 0, pair),
                                    ("shuffled_pair_translator", shuf_cand, 0, shuf),
                                ):
                                    rec = measure_path(
                                        program=task_name, task_name=task.name,
                                        source_probe=src_probe, dest_probe=dst_probe,
                                        settings=settings, source_mech=mech,
                                        source_signature=src_sig,
                                        candidate=candidate,
                                        reverse_independent=(
                                            lambda mm, b=back: b.back(mm, coupled=False)),
                                        reverse_coupled=(
                                            lambda mm, b=back: b.back(mm, coupled=True)),
                                        bounds=provisional, translator=tname, seed=seed,
                                        case=case, label=label,
                                        family_key=f"{task_name}|{src_key}->{dst_key}",
                                        weight_scheme=ws, eta=eta,
                                        sig_fn=lambda p, mm, sc: sigcache.measure(
                                            p, mm, sc, tag=f"{p.model.name}|{split}"),
                                    )
                                    rec.diagnostics["independent_legs"] = float(
                                        ind.get("independent", False))
                                    rec.diagnostics.update(fit)
                                    sink.append(rec)
                        # Flushed per mapping so a failure late in the six costs one
                        # mapping, not the whole phase.
                        done_groups.add(group)
                        task_reports[task_name] = provenance
                        _save_partial(cache, args, all_records, done_groups, task_reports,
                                      cal_records)
                    print(f"  [{src_key} -> {dst_key}] {tname}: "
                          f"{len(all_records)} test records so far", flush=True)
        # ---- self-translation: the positive control the stage otherwise lacks
        # Without ground truth, "the criterion correctly rejected bad proposals" and "the
        # criterion rejects everything" produce identical tables. Translating a mechanism
        # from a model into *itself* is a correspondence that must hold: the identity map
        # is the right answer and the candidate is the mechanism unchanged. If these are
        # not retained, the thresholds are broken rather than the translators.
        if not args.skip_self_translation:
            self_recs = []
            for key in matrix:
                probe = probes[key]["test"]
                scale = final_provisional.c(models[key].name, task.name)
                for mech in mechs[key]:
                    sig = sigcache.measure(probe, mech, scale, tag=f"{key}|test")
                    rec = measure_path(
                        program=task_name, task_name=task.name,
                        source_probe=probe, dest_probe=probe,
                        settings=settings, source_mech=mech, source_signature=sig,
                        candidate=mech,
                        reverse_independent=lambda mm: mm,
                        reverse_coupled=None,
                        bounds=final_provisional, translator="identity", seed=0,
                        case="self_translation", label=1,
                        family_key=f"{task_name}|{key}->{key}",
                        weight_scheme=ws, eta=eta,
                        sig_fn=lambda p, mm, sc: sigcache.measure(
                            p, mm, sc, tag=f"{p.model.name}|test"),
                    )
                    rec.diagnostics["self_translation"] = 1.0
                    self_recs.append(rec)
            all_records.extend(self_recs)
            n_ok = sum(1 for r in self_recs if not r.abstained
                       and RULES["pairwise_ccc_plus"].accepts(r))
            print(f"  [self-translation] {n_ok}/{len(self_recs)} identity correspondences "
                  f"retained by pairwise CCC+", flush=True)
            _save_partial(cache, args, all_records, done_groups, task_reports, cal_records)

        for m in models.values():
            m.free()
        models.clear()

    # ----------------------------------------------------- finalise the bounds
    # Until now every path was scored against provisional eps thresholds. The labelled
    # calibration-split paths set the real ones, exactly as Sec. 3.6 requires and as the
    # controlled stage does. Without this the cycle-aware positive floor never engages
    # (it needs at least five calibration positives), so the run silently used
    # controls-only bounds while reporting cycle_aware -- which over-rejects to the point
    # that every signature rule retains nothing.
    if not all_records:
        print("no records produced")
        return 1

    final_bounds = cal.pooled(bound_mode)
    if cal_records:
        for nr in negatives_from(cal_records):
            cal.add_negative(nr)
        for pr in positives_from(cal_records):
            cal.add_positive(pr)
        cal.mode = bound_mode
        final_bounds = cal.pooled(bound_mode)
        print(f"\ncalibration: {cal.summary()}", flush=True)
        for mode in list(mf.path_of("equivalence_bounds.modes")):
            b = cal.pooled(mode)
            flag = ("  [negative cap conflicts with positive floor]"
                    if any(isinstance(b.provenance.get(c), dict)
                           and b.provenance[c].get("cap_conflicts_with_floor")
                           for c in ("shape", "magnitude", "scalar")) else "")
            print(f"bounds ({mode}): eps_shape={b.eps_shape:.4f} eps_mag={b.eps_mag:.4f} "
                  f"eps_scalar={b.eps_scalar:.4f} eps_repr={b.eps_representational:.4f}{flag}",
                  flush=True)
        # Re-scoring rather than re-measuring: eps only normalises distances that are
        # already recorded, so the candidates and signatures are untouched.
        all_records = rescore_with_bounds(
            all_records, {r.program: final_bounds for r in all_records}
        )
    else:
        print("\n! no calibration-split paths; eps bounds are controls-only", flush=True)

    # --------------------------------------------------------------- analysis
    out = mf.out_dir()
    write_json(out / f"natural_paths_{args.stratum}.json",
               {"records": [r.to_dict() for r in all_records],
                "tasks": task_reports, "matrix": matrix}, mf)
    rules = [r for r in LADDER if r != "multi_model_ccc_plus"]
    res = analyse(all_records, mf, rules=rules, n_boot=args.boot,
                  seed=int(mf.path_of("seeds.bootstrap")))
    couple = coupling_stats(all_records)
    agree = signature_agreement(all_records)
    write_json(out / f"natural_analysis_{args.stratum}.json",
               {"per_rule": {k: {m: v.to_dict() for m, v in d.items()}
                             for k, d in res["per_rule"].items()},
                "coupling": couple, "signature_agreement": agree,
                "tasks": task_reports, "matrix": matrix,
                # The realised thresholds and their provenance, not just the mode name:
                # positive_floor / negative_cap / cap_conflicts_with_floor are what make
                # a retention of zero interpretable.
                "bounds": {"mode": bound_mode, **final_bounds.to_dict()},
                "bounds_by_mode": {m: cal.pooled(m).to_dict()
                                   for m in list(mf.path_of("equivalence_bounds.modes"))},
                "calibration_summary": cal.summary(),
                "n_calibration_paths": len(cal_records),
                "translator_fit": _fit_summary(all_records)}, mf)

    per_tr: Dict[str, Dict[str, object]] = {}
    for tname in args.translators:
        sub = [r for r in all_records if r.translator == tname]
        if not sub:
            continue
        errs = np.array([RULES["pairwise_ccc_plus"].error(r) for r in sub])
        labs = np.array([r.label for r in sub])
        absts = np.array([r.abstained for r in sub])
        from cccplus.stats import Estimate, coverage, false_acceptance
        per_tr[tname] = {
            "pairwise": Estimate(retention(errs, labs)),
            "false_acceptance": Estimate(false_acceptance(errs, labs)),
            "coverage": Estimate(coverage(absts)),
            "abstention": Estimate(float(absts.mean())),
        }
    tables = [
        rule_table(f"table_5_1_natural_{args.stratum}", res["per_rule"], rules,
                   f"Natural-model results ({args.stratum} stratum): validation rules on "
                   f"pretrained checkpoints, pooled over the six directed mappings.",
                   notes=f"n = {res['n_items']} paths "
                         f"({res['n_positives']} translated, {res['n_negatives']} control)."),
        translator_table(f"table_5_2_natural_{args.stratum}", per_tr, args.translators,
                         "Held-out correspondence and coverage by translator. Coverage and "
                         "abstention are reported together.",
                         columns=("pairwise", "false_acceptance", "coverage", "abstention"),
                         header=["Translator", "Retention", "FA", "Cov.", "Abstention"]),
        signature_table(f"table_5_4_natural_{args.stratum}", agree, list(agree.keys()),
                        "Prompt-level causal agreement on pretrained models."),
        coupling_table(f"coupling_natural_{args.stratum}", couple,
                       "Coupled versus independent return legs on pretrained models."),
    ]
    path = write_tables(tables, mf.out_dir("tables_dir"), mf.hash, f"Natural models {args.stratum}")
    print(f"\nwrote {path}")
    return 0


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    mf = FrozenManifest.load(args.manifest)
    set_all_seeds(int(mf.path_of("seeds.master")))
    matrix = resolve_matrix(mf, args.stratum)
    cache = mf.out_dir("cache_dir") / f"natural_{args.stratum}"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"CCC+ natural stage | manifest {mf.hash} | stratum {args.stratum} | {matrix}", flush=True)
    print(f"device={args.device} dtype={args.dtype}", flush=True)

    if args.self_test:
        print("SELF-TEST: randomly initialised models, no checkpoints loaded", flush=True)
    if args.phase in ("corpus", "all"):
        prov = phase_corpus(args, mf, matrix, cache)
        write_json(mf.out_dir() / f"natural_corpus_{args.stratum}.json", prov, mf)
    if args.phase in ("evaluate", "all"):
        return phase_evaluate(args, mf, matrix, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
