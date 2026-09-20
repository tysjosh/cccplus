# Natural-model stage on a remote A100

The controlled stage (Sec. 4.1) runs on CPU and is unaffected by this document. What
follows is the pretrained-LM stage (Sec. 4.2, Tables 5.2 and 5.4).

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "transformers==4.46.3" "tokenizers>=0.20,<0.21" "safetensors>=0.4" datasets
```

`datasets` is needed only for the wikitext translator corpus; without it the loader falls
back to a local text file or synthetic text, and records which in the provenance.

Gated repositories (Gemma-3, Llama-3.2) need an authenticated token:

```bash
huggingface-cli login      # or: export HF_TOKEN=...
```

## 2. Verify before spending GPU time

```bash
python tests/test_method.py
python tests/test_natural.py
python experiments/run_natural.py --self-test --phase all --device cpu --dtype float32 \
    --corpus-passages 60 --corpus-source synthetic --site-depths 3 --max-source-mechanisms 4
```

`--self-test` substitutes randomly initialised models of the same architecture families,
so the full runner — six directed mappings, both tasks, three translators, calibration,
the validation ladder, bootstrap and tables — executes without downloading a checkpoint.
It verifies wiring, not science: with random weights there are no mechanisms to find, so
retention and false acceptance are both zero by construction. That is the expected result.

## 3. Run the stage

Two phases, so only one checkpoint is ever resident.

```bash
# phase 1: cache corpus activations, one model at a time
python experiments/run_natural.py --stratum larger --phase corpus \
    --device cuda --dtype bfloat16 --corpus-passages 2000 --site-depths 6 \
    --free-between-models

# phase 2: fit translators from the cache and run the protocol
python experiments/run_natural.py --stratum larger --phase evaluate \
    --device cuda --dtype bfloat16 --boot 10000
```

`--stratum smaller` and `--stratum larger` select the paper's frozen matrix:

| stratum | Pythia | Gemma 3 PT | Llama 3.2 |
|---|---|---|---|
| smaller | 1.4B-deduped | 1B | 1B |
| larger | 2.8B-deduped | 4B | 3B |

Every checkpoint serves as source, giving the six directed mappings per stratum, and each
source's two destinations are exactly what multi-model CCC+ requires.

## 4. Memory and cost

Phase 2 holds all three models of a stratum at once (it interleaves source and destination
measurement). In bf16 the larger stratum is roughly 5.6 + 8 + 6.4 = 20 GB of weights, which
fits a 40 GB A100 with room for activations. If it does not fit:

- raise `--site-depths` down to 4, which shrinks both the activation cache and the number
  of fitted site pairs quadratically;
- lower `--corpus-passages`;
- lower `--max-source-mechanisms` (default 24 of the strongest source-gated mechanisms).

The activation cache is the other large object: `rows x n_sites x d_model x 4` bytes, with
`rows = passages x positions_per_passage`. At 2000 passages, 4 positions, 12 sites and
d_model 2560 that is about 1 GB per model, written to `results/cache/natural_<stratum>/`.

Rough wall-clock on one A100 for the larger stratum: phase 1 around 20-40 min, phase 2
a few hours, dominated by factual recall, which needs two teacher-forced passes per
prompt because its score is a length-normalised sequence log-probability difference.

## 5. What the stage checks before it trusts a model

- **IOI.** Names must be single tokens in *every* tokenizer, and the surviving prompt set
  is intersected across models so all three are scored on identical prompts. A stratum
  requires at least 90% clean accuracy in every model (`natural.ioi.min_clean_accuracy`);
  the realised accuracies are printed and stored.
- **Factual recall.** Only facts whose correct object outranks its relation-matched
  distractor in *every* model enter the analysis.
- **Source gate.** A mechanism is eligible only if its mean effect clears `tau_min` with
  the predicted sign; the strongest `--max-source-mechanisms` are kept and the count of
  rejected candidates is reported.

If a screening gate fails, the stage says so and skips the task rather than reporting
numbers from a model that cannot do the task.

## 6. Datasets

- **IOI** is generated in-process: 12 template families, name-disjoint splits.
- **Factual recall** prefers CounterFact (Meng et al. 2022), downloaded once to the cache
  directory, because it supplies enough facts to reach the specified 600, paraphrase
  prompts, and a `target_new` that is already relation-matched. `--facts-source builtin`
  forces the offline inline table instead, which has fewer distinct facts; whichever is
  used is recorded in `results/natural_paths_<stratum>.json`.
- **Translator corpus** prefers wikitext-103 with real article boundaries as document ids,
  since the forward and reverse legs must not share a document.

## 7. Outputs

```
results/natural_corpus_<stratum>.json     corpus + activation-cache provenance
results/natural_paths_<stratum>.json      every measured path
results/natural_analysis_<stratum>.json   per-rule metrics, coupling, agreement
results/tables/natural_models_<stratum>.md
results/tables/table_5_2_natural_<stratum>.tex
results/tables/table_5_4_natural_<stratum>.tex
```

## 8. Reading the result honestly

The controlled stage is the paper's primary methodological test because it has planted
ground truth. The natural stage has no ground truth: a translated candidate's correctness
is unknown, which is why its positives are "translator proposed a candidate" and its
negatives are the Sec. 4.2 controls (permuted destination, task-active but different
subspace, shuffled-pair translator). Sec. 5.4 puts it plainly — natural-model results
"report causal-signature agreement without assigning unknown correspondences a correctness
label". Treat Table 5.2 as coverage and agreement, not as accuracy.
