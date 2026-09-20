# CCC+ reproduction

Implementation and empirical evaluation of *"CCC+: Independent Causal Signatures for
Cross-Model Mechanism Translation"*. The paper specifies a method and an experimental
protocol but leaves every result cell as TBD; this repository implements the method,
builds the benchmark, and fills the tables.

## What CCC+ is

A **validation criterion**, not a translator. Given a mechanism `m = (s, V)` whose causal
role is established in a source model, a translation operator proposes a candidate in a
destination model. CCC+ accepts the correspondence only when

1. the candidate has a non-negligible effect of the predicted sign in the destination
   (Eq. 8, the forward gate);
2. its **standardised causal signature** matches the source's in *shape* and *magnitude*
   (Eq. 9) — a per-prompt, multi-setting vector, not a mean;
3. an **independently fitted** reverse translator returns a mechanism whose signature also
   matches (Eq. 11);
4. optionally, both of two independently trained destinations pass (multi-model CCC+).

The two ideas being tested are that *signature* equivalence beats mean-effect equivalence,
and that *independent* reverse estimation beats a coupled one whose round trip can succeed
by construction.

## Layout

```
manifest/manifest.yaml     frozen configuration: seeds, splits, budgets, threshold estimators
cccplus/
  mechanisms.py            m = (s, V), orthonormalisation, subspace similarity
  interventions.py         Eq. 6 directional counterfactual patch, robustness arms, collateral metrics
  signature.py             Eqs. 1-5, 13-14: effects, standardised signatures, d_shape / d_mag, scores
  calibration.py           Sec. 3.6: tau_min, c_{M,T}, equivalence bounds
  validation.py            Eqs. 7-12 and the nine-rung validation ladder
  circuits.py              Sec. 3.7: multi-site circuits, Eq. 16 interaction contrast
  analysis.py              retention / coverage / FA / AUROC, matched-retention contrasts
  stats.py                 hierarchical bootstrap, Holm, AUROC
  reporting.py             Markdown + LaTeX table emission
  models/                  hookable transformer interface; tiny SIIT models
  benchmarks/              six programs, planted structures, SIIT training
  translators/             Procrustes, crosscoder, DFC, MAS, Stitch, Latent Stitch
  pipelines/               candidate construction and the measurement engine
  tasks/                   IOI, factual recall, translator corpus
experiments/
  train_controlled.py      trains the 30 planted benchmark models
  run_controlled.py        primary methodological test (Tables 5.1, 5.3, 5.4)
  run_multisite.py         Sec. 3.7 extension
  run_natural.py           pretrained-LM stage (Tables 5.2, 5.4)
tests/
  test_method.py           method primitives + one regression test per audit defect
  test_natural.py          natural stage, no checkpoints required
```

## Natural-model stage (Sec. 4.2)

**Status: implemented and wired end to end, but never run on real weights.** Intended for a
remote A100; see **`RUNBOOK_A100.md`**.

The files `results/natural_*_tiny.json` and `results/tables/*_natural_tiny.*` exist, and it
would be easy to mistake them for results. They are `--self-test` output: randomly
initialised 3-layer, 64-dimensional models on a synthetic corpus. The `matrix` field names
real repositories (`pythia-160m-deduped`, `gpt2`, `SmolLM2-135M`) but no checkpoint was ever
downloaded, every model fails the screening gate, and prompt-level effect correlations sit
near zero. They demonstrate that the wiring runs; they say nothing about any pretrained
model. `from_random_config` records its seed and dimensions on the model so self-test
provenance is checkable rather than inferred.

Summary of what is implemented:

- **Adapter** (`models/hf_lm.py`) presenting any HF causal LM through the same
  `HookedModel` interface, so no method module changes. Hook points are submodule
  boundaries (`resid_pre`, `attn_out`, `mlp_out`, `resid_post`) that exist in GPT-NeoX,
  Llama/Gemma and GPT-2 alike.
- **Attention heads without per-head plumbing.** A head's causal footprint is the subspace
  it writes into, i.e. the column space of its slice of the output projection. The paper's
  IOI mechanism — "V equal to the residual-stream column space of the head output
  projection" — is therefore obtained by applying Eq. 6 at `attn_out` restricted to
  `head_write_subspace(layer, head)`.
- **IOI**: 12 template families, name-disjoint splits, counterfactual that swaps only the
  two names' roles, scored by the IO-minus-S logit difference.
- **Factual recall**: length-normalised *sequence* log-probability difference, which needs
  two teacher-forced passes per prompt, so the task interface gained `uses_full_scoring`.
  Mechanisms are final-subject-token MLP outputs with V the normalised mean
  clean-counterfactual difference. Facts come from CounterFact when reachable (it supplies
  the paraphrases and an already relation-matched distractor), else an inline table.
- **Six directed mappings per stratum**: every checkpoint serves as source, so each source
  has exactly the two destinations multi-model CCC+ needs.
- **One checkpoint resident at a time**: corpus activations are cached to disk in a separate
  phase, and all causal measurement is within a single model.
- **Screening gates** enforced before anything is reported: single-token names in every
  tokenizer, ≥90% clean IOI accuracy per model, and facts whose object outranks its
  distractor in *every* model.

Verified offline with `--self-test`, which substitutes randomly initialised models of the
same families. That checks wiring, not science: with random weights there are no mechanisms
to find, so retention and false acceptance are zero by construction.

## Running

The controlled stage is CPU-only. The committed results were produced with torch 2.4.1 on
Python 3.11; `requirements.txt` pins that environment.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
V=.venv/bin/python        # or any interpreter with requirements.txt installed

$V experiments/train_controlled.py --threads 4      # ~1 h, resumable, caches per model
$V experiments/run_controlled.py  --threads 4       # primary result
$V experiments/run_multisite.py   --threads 4       # extension
$V experiments/run_controlled.py --intervention mean_ablation          # robustness arm
$V experiments/run_controlled.py --intervention norm_matched_steering  # robustness arm
$V experiments/run_controlled.py --part-c                              # supervised baselines, slow
```

`run_controlled.py` refits the translator cache from scratch on a clean checkout
(`results/cache/` is not versioned), which is the bulk of its runtime. The natural-model
stage is separate and needs a GPU; see `RUNBOOK_A100.md`.

## The planted benchmark

The paper's controlled stage uses InterpBench/Tracr programs with SIIT-trained models. Those
checkpoints are not downloaded here; the benchmark is **rebuilt from scratch** to the same
specification: six programs with three labelled causal variables each, and five
independently initialised models per program that differ in depth (2/3/4 layers), width
(64/80/96) and head count (4/5/6).

Each program maps three disjoint token groups to three ternary variables and combines them
through a deliberately **non-additive** output function

```
G(v0, v1, v2) = 3 * ((v0 + v2) mod 3) + ((v1 + 2*v2) mod 3)
```

so the three variables have comparable mean effects but *different prompt-level
signatures*. That is the configuration in which a mean-effect test cannot separate them and
a signature test can, which is what the paper's central claim is about.

Training has three objectives:

| objective | what it forces |
|---|---|
| behaviour | the model computes the program |
| IIT | importing a planted subspace from a counterfactual prompt produces the interchange output, so the subspace really carries its variable |
| strict | importing the orthogonal complement of a site's planted blocks leaves the output unchanged, so the site carries nothing task-relevant outside them |

Planted structures give the six case types the paper asks for: `shared`, `permuted`
(planted channel permutation), `split` (one variable over two sites), `redundant`
(duplicated, either copy sufficient), `merged` (two variables in one block), and `absent`
(the model is trained on the restricted program, so abstention is the only correct answer).

**Quality gate.** A model that misses the preregistered SIIT gate (clean ≥ 0.95, worst-case
interchange ≥ 0.90) does not carry the planted graph and must not enter the analysis.
Failures trigger a fixed escalation ladder — more steps, then an interchange curriculum,
then a lower learning rate — applied identically to every model, with all attempts recorded
in `results/controlled_benchmark_models.json`.

## Three-part evaluation

**Part A — rule discrimination (primary).** Candidates are supplied by construction, so
every validation rule is scored on an *identical* candidate set, as Sec. 3.6 requires.
Positives are planted counterparts; negatives are deliberately hard:

- `wrong_variable` — another planted block in the same destination: same rank, comparable
  effect size, different causal role;
- `effect_matched_random` — a random subspace at the *same site*, selected to match the
  true counterpart's mean effect in sign and magnitude, i.e. built to survive a mean-effect
  test;
- `absent` — no counterpart exists;
- `split_half` — one half of a split variable: a partial correspondence.

**Part B — translator evaluation.** Candidates come from the translators themselves, so
coverage and abstention are measured as defined. Adds a `shuffled_pair_translator` control
(destination activations row-shuffled before fitting), where any acceptance is a false
acceptance. Uses the three unsupervised translators (Procrustes, crosscoder, DFC).

**Part C — behaviour-supervised baselines** (`--part-c`, off by default). MAS, Latent Stitch
and Stitch learn an alignment from task labels instead of representational structure. They
are fitted on the **hyperparameter** split only, never on calibration or test prompts, so
they get no more privileged access than the unsupervised translators get corpus access.

Two reasons they are reported separately rather than pooled into Part B. They consume task
labels, which the others do not. And they have no coupled return construction, so the rules
that need one (`round_trip_only`, `scalar_ccc`) are *omitted* from their table rather than
scored as failures — "the construction does not exist" and "the construction did not hold"
are different findings and must not share a cell.

Off by default because the cost is not comparable: each candidate is a gradient optimisation
over every eligible destination site, run through the destination model. At the manifest's
600 steps this does not finish within half an hour for a *single* program, against roughly 17
minutes for all six in Part A/B. **No Part C numbers are currently reported**; the path is
wired and tested, but has only been exercised at a reduced step budget.

## Deviations from the paper, and why

| # | Paper | Here | Reason |
|---|---|---|---|
| 1 | InterpBench/Tracr checkpoints | benchmark rebuilt to the same spec | checkpoints not downloaded; rebuilding also lets every planted case type be constructed exactly and verified |
| 2 | Pythia 1.4B/2.8B, Gemma-3 1B/4B, Llama-3.2 1B/3B | natural-model stage **implemented and verified offline; never executed on real weights** | the checkpoints do not fit this 8 GB host, and the Gemma/Llama repos are gated. The code is complete and tested against randomly initialised models of the same three architecture families; see `RUNBOOK_A100.md`. The only natural-stage artifacts in `results/` come from `--self-test`, i.e. random 64-dim weights on a synthetic corpus, and carry no scientific content |
| 3 | five translator seeds | Procrustes 1 (deterministic), crosscoder/DFC 3 | compute; recorded in the manifest |
| 4 | crosscoder 512 atoms / 3000 steps | 256 atoms / 800 steps | compute; reconstruction quality (FVU) is reported with the results |
| 5 | rank-1 primary, rank-4 as robustness | rank-4 primary | the benchmark was trained at rank 4 before this was reconciled. Rank 4 is the **harder** setting for a translator, so the reported alignment quality is conservative rather than flattering |
| 6 | clusters over stratum, source mechanism, **family** and translator seed, "prompts are resampled within these clusters" | hierarchical bootstrap over program/model stratum, source mechanism and translator seed; **family** clustering and prompt resampling absent | prompt-level resampling would require recomputing every signature per replicate. Family clustering is not merely skipped but *undefined* at this granularity: a path's signature is measured over every prompt in its split, so a path is not nested inside one template family. The manifest now declares only the three levels actually resampled, and `statistics.cluster_levels` is read by the code — asking for `family` or `prompts` raises instead of being silently dropped. Prompt-level variability is reported directly as per-family worst-case error |
| 7 | equivalence bounds from identity and reparameterisation controls | reported under **two** bound modes | see below |

### The bound-mode issue (deviation 7)

Sec. 3.6 fixes `eps_shape`, `eps_mag`, `eps_scalar` from *within-model* identity and
reparameterisation controls. Implemented literally, this over-rejects: a cross-model round
trip through an independently fitted reverse map incurs variation that a within-model
control cannot see, so nearly all true correspondences fail and positive retention collapses
towards zero. This is a property of the specification, not of the implementation, and it is
reported.

Both modes are therefore reported, as separate artifacts rather than as a stdout line:
`controlled_analysis.json` carries the primary mode and a `part_a_by_bound_mode` block, the
secondary mode is written to `controlled_analysis__<mode>.json`, and a
`table_5_1_controlled__<mode>` table is emitted alongside the primary one. The two are
directly comparable because the secondary mode is obtained by *rescoring the same
measurements*: the candidates, signatures and `tau_min` are identical and only the thresholds
move.

- **`controls_only`** — literal Sec. 3.6.
- **`cycle_aware`** — additionally floored so that labelled *calibration-split* positives are
  retained at a preregistered target (0.90).

**The cap does not survive in practice, and this must not be glossed.** Where the positive
floor exceeds the 5% false-acceptance cap, the code keeps the floor, so the calibrated
threshold is floored, *not* capped, and the realised false acceptance can exceed the
calibration budget. On this benchmark that happens for **shape, magnitude and scalar on all
six held-out programs** — it is the normal case here, not an edge case. Every affected
program and component is named in the note under the rule tables, and the per-component
`positive_floor` / `negative_cap` / `cap_conflicts_with_floor` values are recorded in
`bounds_by_program`. So `cycle_aware` is honestly a *new calibration variant*, not the
paper's procedure; `controls_only` is the literal one.

The paper's headline comparison — false acceptance at **matched positive retention** — is
threshold-free, so the rule ranking does not depend on this choice.

Matching is done at a preregistered retention level (0.90) rather than at the reference
rule's own realised retention, because matching to a rule whose calibrated retention is low
collapses every contrast to zero and measures nothing. The full FA-vs-retention curve is
reported too.

## Claims this repository can and cannot support

Supported by the controlled stage: the relative ordering of the validation rules, the
contribution of each added condition, the coupled-vs-independent return comparison, and
behaviour on planted split/merged/redundant/absent structures.

Not supported here: anything about pretrained language models, IOI, or factual recall
(deviation 2); and, as the paper itself states, passing CCC+ supports equivalence of the
*evaluated signature*, not global mechanistic identity.

## Verification

```bash
$V tests/test_method.py          # 20 tests, no pytest required
$V tests/test_natural.py         # 11 tests, no checkpoints or network required
$V -m pytest tests -q            # if pytest is available
```

`test_method.py` covers the method primitives (projector basis-invariance, scale-invariance
of `d_shape`, symmetry of `d_mag`, weight normalisation, AUROC against known values, Holm
monotonicity, monotonicity of FA in matched retention, and that the ladder is genuinely
cumulative) and carries one regression test per defect listed below. `test_natural.py`
covers the HF adapter across all three architecture families, head write-subspaces, patch
efficacy, IOI counterfactual construction and name-disjoint splits, factual sequence scoring,
cross-model screening, corpus document splits, and verifiable translator-leg independence.

Both suites are deterministic and order-independent. Two things that were not:
`d_mag` symmetry was asserted as exact float equality on an **unseeded** draw, which failed
on roughly a fifth of runs, and `from_random_config` drew its weights from the *global* torch
RNG, so model weights — and therefore which translators abstained — depended on what had
consumed that RNG earlier in the process.

### Correctness fixes from code audit

Each of these was a real defect that would have made reported numbers untrustworthy. Each
now has a test that fails on the pre-fix behaviour.

| defect | fix |
|---|---|
| `source_gate_passed` was computed but never enforced, so a rule could accept a path whose source mechanism was never causally established (Sec. 3.1) | the source gate is prepended as a condition to **every** rule |
| the representational threshold applied `min(...)` *after* the false-acceptance constraint; lowering a similarity threshold admits more negatives and can break the 5% calibration budget | for a similarity criterion the FA budget is a lower bound and the controls an upper bound; conflicts are recorded, not silently resolved |
| leave-one-program-out excluded the held-out program's controls and negatives but still passed all `gated_norms` | norms are filtered by program, so the invariant holds by construction rather than by a model-naming coincidence |
| `split_documents` invented contiguous pseudo-documents | takes real `document_ids`; for the i.i.d. synthetic probe corpus each row is its own document, which is the correct semantics there |
| `independence_report` looked for a `fit_index_hash` that translators never recorded, so it always reported "no shared examples" without proving it | translators record their fitting rows and a content fingerprint; the report verifies disjoint rows *and* that no parameter tensor is shared |
| causal baselines seeded from Python's randomised `hash()`, so runs differed between processes | deterministic `stable_seed`, with the value pinned in a test |
| the returned "norm error" differenced the norms of two orthonormal bases, which are `sqrt(rank)` by construction, so the diagnostic was identically zero | translators record the norm gain of the mapped subspace *before* re-orthonormalisation; the diagnostic is its absolute log deviation from unity |
| `compare_signatures` checked tensor shape only, so a mismatch in setting order or prompt pairing would yield a well-formed but meaningless distance | `assert_comparable` verifies setting keys, shape, and a prompt-set/pairing key, and raises. The key is based on family labels and pairing rather than token ids, because the same prompts tokenise differently across architectures while the coordinate correspondence must still hold |
| **multi-model conjunction items were cross-paired.** `build_multi_items` built a grouping keyed on `case`, discarded it, and paired on everything *except* `case`. 74% of the 7350 items paired two unrelated candidate kinds — a true counterpart in one destination against a decoy in the other — so the item's label described neither leg, and one positive leg was reused across thousands of negative items, breaking the independence the bootstrap assumes. This made the multi-model row and the `Multi − pairwise` contrast unreliable | a conjunction item is now one coherent decision: both legs the planted counterpart (positive) or both legs the same decoy species (negative); mixed pairs are dropped. `case` is deliberately *not* the key, because a true counterpart is `permuted` in one destination and `merged` in another — keying on `case` yields zero positives. 378 positives + 1890 negatives |
| **set-valued precision and recall were the same number**: both were the mean per-path subspace similarity, so they could not express the thing they exist to measure — a variable split over two blocks is only half recovered when one block is accepted | recall and precision are computed over the set of candidates a rule *accepts* per (source mechanism, destination), matched to planted blocks; an uncovered block costs recall, a spurious acceptance costs precision. Adds `set_exact` (every block covered, nothing spurious) and the context count |
| **only the primary bound mode was ever written out**, despite the README claiming both are reported, so the literal Sec. 3.6 result could not be compared against the variant actually used | the secondary mode is emitted as its own analysis JSON and table, obtained by rescoring the same measurements so the two differ only in thresholds. The floor-overrides-cap conflict is named per program and component in the table notes |
| **three of the six translators were unreachable.** MAS, Latent Stitch and Stitch were fully implemented but Part B filtered on a hardcoded tuple, `fit_causal_translator` was called from nowhere, and the `causal_contexts` parameter that would feed it was accepted and ignored | wired as Part C behind `--part-c`, fitted on the hyperparameter split only, with the coupled-return rules excluded rather than scored as failures. The filter now uses the `UNSUPERVISED` constant that already existed |
| **the coupling diagnostic had a blind spot.** `_is_derived_from` and `_shares_parameters` only inspected the `forward` attribute, but the crosscoder and DFC coupled legs reach the forward map through `same_atom`/`shared_dict`, so a genuinely coupled leg reported as independent. The same helpers certify the *independent* leg, so the gap could equally have certified a coupled leg as independent | provenance is followed transitively through a declared set of delegate attributes; both helpers walk it |
| **`from_random_config` drew weights from the global torch RNG**, so the natural self-test was not reproducible between runs and test outcomes changed with execution order | seeded from the model name via `stable_seed`, with the surrounding RNG state saved and restored so the caller's stream is untouched; the seed and dimensions are recorded on the model |
| `statistics.cluster_levels` advertised four bootstrap levels while the code hardcoded three, so the recorded configuration overstated the resampling performed | the manifest drives the clustering and declares only the three levels actually resampled; an unsupported level raises rather than being silently dropped |
