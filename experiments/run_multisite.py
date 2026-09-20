"""Multi-site circuit extension experiment (Sec. 3.7, 4.4).

Reported separately from the component-level claim, as the paper requires: "we
treat this as a separately labeled extension experiment rather than evidence
required for the primary component-level claim."

Circuit under test: the planted blocks of two variables at two different sites in
the source model, with a hypothesised dependency between them. Because the
program's output function is non-additive in its variables, the interaction
contrast Gamma is genuinely non-zero, so the circuit has an edge to recover.

Compared methods
  nodewise    translate each node independently; judged on node coordinates only
  multisite   translate nodes, then require the *concatenated* signature -- node
              effects, joint effect and Gamma -- to match

Negative: the node-permuted circuit. It occupies the correct sites with the
correct subspaces but pairs them with the wrong source nodes. Gamma is symmetric
under the swap, so only the per-node coordinates expose it.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cccplus.benchmarks.programs import get_program, program_output
from cccplus.benchmarks.siit import load_siit, make_batch
from cccplus.circuits import (
    CircuitHypothesis,
    CircuitResult,
    circuit_signature,
    edge_recovery,
    interaction_contrast,
    node_recovery,
    nodewise_signature,
    permute_nodes,
    translate_circuit,
)
from cccplus.config import FrozenManifest, set_all_seeds, stable_seed, write_json
from cccplus.interventions import CausalProbe, Setting, settings_from_manifest
from cccplus.mechanisms import Mechanism
from cccplus.pipelines.controlled import eligible_sites, source_mechanisms, task_data_for
from cccplus.pipelines.evaluate import fit_translator_pair
from cccplus.reporting import Table, multisite_table, write_tables
from cccplus.signature import compare_signatures, signature_weights, weighted_norm
from cccplus.tasks.base import LogitDiffTask, partition_by_family
from cccplus.translators.base import collect_activations, split_documents


def build_joint_prompts(program, vars_, n_families, per_family, seed):
    """Prompts whose counterfactual changes *both* circuit variables.

    Needed so that single-node, joint and interaction interventions are all measured
    on one prompt set and therefore share coordinates.
    """
    rng = np.random.default_rng(seed)
    K = program.n_vars
    clean, cf, vals, cfvals, fam = [], [], [], [], []
    for f in range(n_families):
        ctx = rng.integers(0, 3, size=K)
        ctx_tokens = program.tokens_for_values(ctx[None, :], rng)[0]
        base = np.tile(ctx_tokens, (per_family, 1))
        v = np.tile(ctx, (per_family, 1))
        vp = v.copy()
        for k in vars_:
            tv = rng.integers(0, 3, size=per_family)
            shift = rng.integers(1, 3, size=per_family)
            v[:, k] = tv
            vp[:, k] = (tv + shift) % 3
            program.fill_group(base, k, tv, rng)
        x = base
        xp = x.copy()
        for k in vars_:
            program.fill_group(xp, k, vp[:, k], rng)
        clean.append(x); cf.append(xp); vals.append(v); cfvals.append(vp)
        fam.append(np.full(per_family, f, dtype=int))
    return (np.concatenate(clean), np.concatenate(cf), np.concatenate(vals),
            np.concatenate(cfvals), np.concatenate(fam))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--programs", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    mf = FrozenManifest.load(args.manifest)
    master = int(mf.path_of("seeds.master"))
    set_all_seeds(master)
    cache = mf.out_dir("cache_dir") / "controlled"
    programs = args.programs or list(mf.path_of("controlled.programs"))
    settings = settings_from_manifest(mf)
    ws = str(mf.path_of("signature.weights"))
    eta = float(mf.path_of("signature.eta"))
    n_fam = int(mf.path_of("controlled.n_families"))
    per_fam = int(mf.path_of("controlled.per_family"))
    fractions = mf.path_of("partitions.fractions")
    tr_names = list(mf.path_of("controlled.rule_discrimination_translators"))

    task = LogitDiffTask("circuit_logit_diff", -1.0)
    results: List[CircuitResult] = []
    t0 = time.time()

    for prog_name in programs:
        program = get_program(prog_name)
        # Circuit variables: the two whose planted blocks sit at distinct sites.
        vars_ = (0, 1)
        clean, cf, vals, cfvals, fam = build_joint_prompts(
            program, vars_, n_fam, per_fam, stable_seed(master, prog_name, "circuit")
        )
        parts = partition_by_family(fam, fractions, stable_seed(master, prog_name, "circuit_split"))
        test_idx = parts["test"]

        src = load_siit(cache / f"{prog_name}__src.pt")
        src.model.name = f"{prog_name}/src/{src.name}"
        dst = load_siit(cache / f"{prog_name}__dst_shared.pt")
        dst.model.name = f"{prog_name}/dst_shared/{dst.name}"

        def probe_for(sm):
            absent = set(sm.placement.absent_vars)
            data = type("D", (), {})  # placeholder replaced below
            from cccplus.tasks.base import TaskData
            td = TaskData(
                clean=make_batch(program, clean[test_idx]),
                counterfactual=make_batch(program, cf[test_idx]),
                correct=torch.as_tensor(program_output(vals[test_idx], absent), dtype=torch.long),
                distractor=torch.as_tensor(program_output(cfvals[test_idx], absent), dtype=torch.long),
                family=fam[test_idx],
            )
            return CausalProbe(sm.model, task, td, sites=eligible_sites(sm), collect_kl=False)

        p_src, p_dst = probe_for(src), probe_for(dst)

        src_mechs = source_mechanisms(src)
        if not all(v in src_mechs for v in vars_):
            print(f"  [{prog_name}] source lacks a planted block for {vars_}; skipping")
            continue
        source_circuit = CircuitHypothesis(
            nodes=[src_mechs[v] for v in vars_], edges=[(0, 1)],
            model=src.model.name, label=f"v{vars_[0]}+v{vars_[1]}",
        )
        truth_nodes = []
        ok = True
        for v in vars_:
            blocks = [b for b in dst.placement.blocks if b.vars == frozenset({v})]
            if not blocks:
                ok = False
                break
            truth_nodes.append(blocks[0].mechanism(dst.model.name))
        if not ok:
            print(f"  [{prog_name}] destination lacks planted counterparts; skipping")
            continue
        truth_circuit = CircuitHypothesis(nodes=truth_nodes, edges=[(0, 1)],
                                          model=dst.model.name, label="truth")

        g_src = float(interaction_contrast(p_src, source_circuit.nodes[0], source_circuit.nodes[1],
                                           settings[0]).mean())
        print(f"  [{prog_name}] source Gamma = {g_src:+.3f} "
              f"(non-additive joint effect required by Sec. 4.4)", flush=True)

        # translator (one seed: this is a reported extension, not the primary claim)
        probe_batch = make_batch(program, program.sample_plain(
            int(mf.path_of("controlled.n_prompts.probe")), np.random.default_rng(master)))
        fwd_i, rev_i = split_documents(len(probe_batch), 0.5, stable_seed(master, prog_name, "cps"))
        s_all = collect_activations(src.model, probe_batch, eligible_sites(src))
        d_all = collect_activations(dst.model, probe_batch, eligible_sites(dst))
        for name in tr_names:
            pair = fit_translator_pair(
                name, s_all.subset(fwd_i), d_all.subset(fwd_i), s_all.subset(rev_i), d_all.subset(rev_i),
                src.model.name, dst.model.name, stable_seed(master, prog_name, name, "circuit") % (2**31), mf,
            )
            proposed = translate_circuit(pair.forward.translate, source_circuit, dst.model.name)

            c_src = max(weighted_norm(
                circuit_signature(p_src, source_circuit, settings).vector(),
                signature_weights(1, 1, "uniform")), 1e-9)
            for case, candidate, label in (
                ("truth", truth_circuit, 1),
                ("node_permuted", permute_nodes(truth_circuit), 0),
                ("translated", proposed, 1 if proposed is not None else 0),
            ):
                r = CircuitResult(prog_name, src.model.name, dst.model.name, name,
                                  "multisite", case, label)
                if candidate is None:
                    results.append(r)
                    continue
                r.abstained = False
                r.node_recovery = node_recovery(candidate, truth_circuit)
                er, ee = edge_recovery(p_dst, candidate, p_src, source_circuit, settings[0])
                r.edge_recovery, r.interaction_error = er, ee

                s_full = circuit_signature(p_src, source_circuit, settings)
                t_full = circuit_signature(p_dst, candidate, settings)
                n_src = max(weighted_norm(s_full.vector(), signature_weights(s_full.n_settings, s_full.n_prompts, ws)), 1e-9)
                n_dst = max(weighted_norm(t_full.vector(), signature_weights(t_full.n_settings, t_full.n_prompts, ws)), 1e-9)
                cmp = compare_signatures(s_full.rescaled(n_src), t_full.rescaled(n_dst), 1.0, 1.0, eta, ws)
                r.d_shape, r.d_mag = cmp.shape, cmp.magnitude
                results.append(r)

                # nodewise counterpart: same candidate, node coordinates only
                rn = CircuitResult(prog_name, src.model.name, dst.model.name, name,
                                   "nodewise", case, label, abstained=False,
                                   node_recovery=r.node_recovery)
                s_n = nodewise_signature(p_src, source_circuit, settings)
                t_n = nodewise_signature(p_dst, candidate, settings)
                nn_s = max(weighted_norm(s_n.vector(), signature_weights(s_n.n_settings, s_n.n_prompts, ws)), 1e-9)
                nn_t = max(weighted_norm(t_n.vector(), signature_weights(t_n.n_settings, t_n.n_prompts, ws)), 1e-9)
                cmp_n = compare_signatures(s_n.rescaled(nn_s), t_n.rescaled(nn_t), 1.0, 1.0, eta, ws)
                rn.d_shape, rn.d_mag = cmp_n.shape, cmp_n.magnitude
                results.append(rn)
        print(f"  [{prog_name}] done ({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ report
    out = mf.out_dir()
    write_json(out / "multisite_results.json", {"records": [r.to_dict() for r in results]}, mf)

    def agg(method: str) -> Dict[str, float]:
        rs = [r for r in results if r.method == method and not r.abstained]
        pos = [r for r in rs if r.case == "truth"]
        neg = [r for r in rs if r.case == "node_permuted"]
        f = lambda xs, k: float(np.nanmean([getattr(x, k) for x in xs])) if xs else float("nan")
        d = {
            "node": f(pos, "node_recovery"),
            "edge": f(pos, "edge_recovery") if method == "multisite" else float("nan"),
            "interaction_error": f(pos, "interaction_error") if method == "multisite" else float("nan"),
        }
        # separation between the true circuit and the node-permuted one
        d["d_shape_truth"] = f(pos, "d_shape")
        d["d_shape_permuted"] = f(neg, "d_shape")
        return d

    rows = {"Nodewise translation": agg("nodewise"), "Multi-site CCC+": agg("multisite")}
    t1 = multisite_table(
        "multisite_extension", rows,
        "Multi-site circuit extension. Node and edge recovery are evaluated against "
        "planted circuits; the interaction error is the relative discrepancy in Gamma.")
    sep_rows = [
        [k, f"{v['d_shape_truth']:.3f}" if np.isfinite(v["d_shape_truth"]) else "n/a",
         f"{v['d_shape_permuted']:.3f}" if np.isfinite(v["d_shape_permuted"]) else "n/a",
         f"{v['d_shape_permuted'] - v['d_shape_truth']:.3f}"
         if np.isfinite(v["d_shape_permuted"]) and np.isfinite(v["d_shape_truth"]) else "n/a"]
        for k, v in rows.items()
    ]
    t2 = Table("multisite_separation",
               ["Signature used", "d_shape (true circuit)", "d_shape (node-permuted)", "separation"],
               sep_rows,
               "Separation between the true circuit and a node-permuted circuit occupying the "
               "same sites. A larger gap means the signature distinguishes wiring, not just "
               "location.")
    path = write_tables([t1, t2], mf.out_dir("tables_dir"), mf.hash, "Multisite extension")
    print(f"\nwrote {path}\ntotal {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
