"""Train the planted ground-truth benchmark (Sec. 4.1).

For each of the six programs, train five independently initialised SIIT models:

  src                  source model, plain placement
  dst_shared           permuted placement (planted channel permutation)
  dst_split_redundant  one variable split over two sites, one duplicated
  dst_merged           two variables sharing one block
  dst_absent           trained on the restricted program, so one variable is
                       genuinely absent and abstention is the only correct answer

Results are cached per model so the script is resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cccplus.benchmarks.programs import get_program
from cccplus.benchmarks.siit import load_siit, save_siit, train_siit
from cccplus.config import FrozenManifest, stable_seed, write_json

# Steps scale with the number of planted groups each structure has to satisfy.
STEPS = {"plain": 900, "permuted": 900, "split_redundant": 1600, "merged": 800, "absent": 800}

# Fixed escalation ladder for models that miss the SIIT quality gate.
ESCALATION = [
    {"step_scale": 1.0, "lr_scale": 1.0, "warmup_frac": 0.00, "ramp_frac": 0.00},
    {"step_scale": 1.5, "lr_scale": 1.0, "warmup_frac": 0.15, "ramp_frac": 0.30},
    {"step_scale": 2.0, "lr_scale": 0.7, "warmup_frac": 0.20, "ramp_frac": 0.35},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--programs", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="retries allowed for a model that misses the SIIT quality gate")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    mf = FrozenManifest.load(args.manifest)
    cache = mf.out_dir("cache_dir") / "controlled"
    cache.mkdir(parents=True, exist_ok=True)

    programs = args.programs or list(mf.path_of("controlled.programs"))
    archs = mf.path_of("controlled.architectures")
    instances = mf.path_of("controlled.instances")
    siit_cfg = mf.path_of("controlled.siit")
    rank = int(mf.path_of("controlled.mechanism_rank"))
    master = int(mf.path_of("seeds.master"))

    summary = {}
    t_start = time.time()
    for prog_name in programs:
        program = get_program(prog_name)
        summary[prog_name] = {}
        for inst_name, inst in instances.items():
            path = cache / f"{prog_name}__{inst_name}.pt"
            attempts: list = []
            if path.exists() and not args.force:
                sm = load_siit(path)
                print(f"[cached] {prog_name}/{inst_name}", flush=True)
            else:
                structure = inst["structure"]
                base_steps = STEPS.get(structure, int(siit_cfg["steps"]))
                sm = None
                # A model that misses the preregistered SIIT gate does not carry the
                # planted causal graph, so it must not enter the analysis. The retry
                # policy is a fixed escalation ladder, applied identically to every
                # model, rather than per-model hand tuning: attempt 0 trains all
                # objectives together; later attempts add an interchange curriculum,
                # more steps, and a lower learning rate. Every attempt is recorded.
                for attempt in range(args.max_attempts):
                    seed = stable_seed(master, prog_name, inst_name, inst["arch"], attempt) % (2**31)
                    steps = int(base_steps * ESCALATION[attempt]["step_scale"])
                    esc = ESCALATION[min(attempt, len(ESCALATION) - 1)]
                    t0 = time.time()
                    cand = train_siit(
                        program=program,
                        arch=archs[inst["arch"]],
                        structure=structure,
                        seed=seed,
                        steps=steps,
                        lr=float(siit_cfg["lr"]) * esc["lr_scale"],
                        batch=int(siit_cfg["batch"]),
                        behavior_weight=float(siit_cfg["behavior_weight"]),
                        iit_weight=float(siit_cfg["iit_weight"]),
                        strict_weight=float(siit_cfg["strict_weight"]),
                        rank=rank,
                        name=inst["arch"],
                        warmup_frac=esc["warmup_frac"],
                        ramp_frac=esc["ramp_frac"],
                    )
                    md = cand.metrics.to_dict()
                    ok = md["clean_accuracy"] >= float(siit_cfg["min_clean_accuracy"]) and \
                        md["min_iit_accuracy"] >= float(siit_cfg["min_iit_accuracy"])
                    attempts.append({"attempt": attempt, "seed": seed, "steps": steps,
                                     "warmup_frac": esc["warmup_frac"], "lr_scale": esc["lr_scale"],
                                     "clean": md["clean_accuracy"], "min_iit": md["min_iit_accuracy"],
                                     "strict": md["mean_strict_accuracy"], "passed": bool(ok)})
                    print(
                        f"[{'trained' if ok else 'RETRY  '}] {prog_name}/{inst_name} "
                        f"({inst['arch']}, {structure}) attempt {attempt} steps={steps} "
                        f"{time.time()-t0:.0f}s clean={md['clean_accuracy']:.3f} "
                        f"minIIT={md['min_iit_accuracy']:.3f} strict={md['mean_strict_accuracy']:.3f}",
                        flush=True,
                    )
                    if sm is None or md["min_iit_accuracy"] > sm.metrics.to_dict()["min_iit_accuracy"]:
                        sm = cand
                    if ok:
                        break
                save_siit(path, sm)
            md = sm.metrics.to_dict()
            summary[prog_name][inst_name] = {
                "arch": inst["arch"],
                "structure": inst["structure"],
                "clean_accuracy": md["clean_accuracy"],
                "min_iit_accuracy": md["min_iit_accuracy"],
                "mean_strict_accuracy": md["mean_strict_accuracy"],
                "iit_accuracy": md["iit_accuracy"],
                "placement": sm.placement.summary(),
                "attempts": attempts,
                "meets_gate": bool(
                    md["clean_accuracy"] >= float(siit_cfg["min_clean_accuracy"])
                    and md["min_iit_accuracy"] >= float(siit_cfg["min_iit_accuracy"])
                ),
            }

    out = write_json(mf.out_dir() / "controlled_benchmark_models.json", summary, mf)
    n_ok = sum(1 for p in summary.values() for v in p.values() if v["meets_gate"])
    n_tot = sum(len(p) for p in summary.values())
    print(f"\n{n_ok}/{n_tot} models meet the preregistered SIIT quality gate")
    print(f"total {time.time()-t_start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
