# Controlled ground truth (mean_ablation)

manifest `e5c891d9f85101e4`

**benchmark_validity**

Planted benchmark validity. Clean accuracy, worst-case interchange accuracy over planted groups, and strict-localisation accuracy. The planted causal graph is only usable as ground truth where all three are high.

| Program        | Instance            | Arch | Structure       | Clean acc. | min IIT acc. | Strict acc. | Gate |
|----------------|---------------------|------|-----------------|------------|--------------|-------------|------|
| frac_prevs     | src                 | M1   | plain           | 1.000      | 1.000        | 1.000       | yes  |
| frac_prevs     | dst_shared          | M2   | permuted        | 1.000      | 1.000        | 1.000       | yes  |
| frac_prevs     | dst_split_redundant | M3   | split_redundant | 1.000      | 1.000        | 1.000       | yes  |
| frac_prevs     | dst_merged          | M2   | merged          | 1.000      | 1.000        | 1.000       | yes  |
| frac_prevs     | dst_absent          | M3   | absent          | 1.000      | 1.000        | 1.000       | yes  |
| running_max    | src                 | M1   | plain           | 1.000      | 1.000        | 1.000       | yes  |
| running_max    | dst_shared          | M2   | permuted        | 1.000      | 1.000        | 1.000       | yes  |
| running_max    | dst_split_redundant | M3   | split_redundant | 1.000      | 1.000        | 1.000       | yes  |
| running_max    | dst_merged          | M2   | merged          | 1.000      | 1.000        | 1.000       | yes  |
| running_max    | dst_absent          | M3   | absent          | 1.000      | 1.000        | 1.000       | yes  |
| token_hist     | src                 | M1   | plain           | 1.000      | 1.000        | 1.000       | yes  |
| token_hist     | dst_shared          | M2   | permuted        | 1.000      | 1.000        | 1.000       | yes  |
| token_hist     | dst_split_redundant | M3   | split_redundant | 1.000      | 1.000        | 1.000       | yes  |
| token_hist     | dst_merged          | M2   | merged          | 1.000      | 1.000        | 1.000       | yes  |
| token_hist     | dst_absent          | M3   | absent          | 1.000      | 1.000        | 1.000       | yes  |
| dyck1_balance  | src                 | M1   | plain           | 1.000      | 1.000        | 1.000       | yes  |
| dyck1_balance  | dst_shared          | M2   | permuted        | 1.000      | 1.000        | 1.000       | yes  |
| dyck1_balance  | dst_split_redundant | M3   | split_redundant | 1.000      | 1.000        | 1.000       | yes  |
| dyck1_balance  | dst_merged          | M2   | merged          | 1.000      | 1.000        | 1.000       | yes  |
| dyck1_balance  | dst_absent          | M3   | absent          | 1.000      | 1.000        | 1.000       | yes  |
| sort_key       | src                 | M1   | plain           | 1.000      | 1.000        | 1.000       | yes  |
| sort_key       | dst_shared          | M2   | permuted        | 1.000      | 1.000        | 1.000       | yes  |
| sort_key       | dst_split_redundant | M3   | split_redundant | 1.000      | 1.000        | 1.000       | yes  |
| sort_key       | dst_merged          | M2   | merged          | 1.000      | 1.000        | 1.000       | yes  |
| sort_key       | dst_absent          | M3   | absent          | 1.000      | 1.000        | 1.000       | yes  |
| induction_copy | src                 | M1   | plain           | 1.000      | 1.000        | 0.999       | yes  |
| induction_copy | dst_shared          | M2   | permuted        | 1.000      | 0.969        | 0.993       | yes  |
| induction_copy | dst_split_redundant | M3   | split_redundant | 1.000      | 0.991        | 1.000       | yes  |
| induction_copy | dst_merged          | M2   | merged          | 1.000      | 0.989        | 0.998       | yes  |
| induction_copy | dst_absent          | M3   | absent          | 1.000      | 1.000        | 1.000       | yes  |


**table_5_1_controlled**

Controlled ground-truth results. Ret. is positive retention, Cov. is candidate coverage, FA is false acceptance; 95% hierarchical-bootstrap intervals in brackets. Every rule is evaluated on the same candidates.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.048 [0.007, 0.096] | 0.581 [0.483, 0.684] | 0.742 [0.603, 0.901] |
| Round trip only     | 0.918 [0.846, 0.971] | 1.000 [1.000, 1.000] | 0.749 [0.667, 0.826] | 0.658 [0.589, 0.722] | 0.694 [0.575, 0.841] |
| Forward only        | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.930 [0.875, 0.972] | 0.535 [0.514, 0.563] | 0.930 [0.875, 0.972] |
| Original scalar CCC | 0.918 [0.846, 0.971] | 1.000 [1.000, 1.000] | 0.697 [0.610, 0.776] | 0.685 [0.619, 0.748] | 0.647 [0.527, 0.786] |
| CCC-Ind             | 0.910 [0.829, 0.969] | 1.000 [1.000, 1.000] | 0.706 [0.621, 0.785] | 0.686 [0.617, 0.752] | 0.661 [0.519, 0.795] |
| CCC-Shape           | 0.810 [0.666, 0.924] | 1.000 [1.000, 1.000] | 0.204 [0.138, 0.271] | 0.897 [0.838, 0.948] | 0.343 [0.099, 0.507] |
| CCC-Sig             | 0.810 [0.666, 0.924] | 1.000 [1.000, 1.000] | 0.159 [0.095, 0.226] | 0.925 [0.876, 0.965] | 0.267 [0.070, 0.378] |
| Pairwise CCC+       | 0.714 [0.549, 0.859] | 1.000 [1.000, 1.000] | 0.130 [0.073, 0.191] | 0.878 [0.811, 0.936] | 0.459 [0.198, 0.590] |
| Multi-model CCC+    | 0.550 [0.271, 0.814] | 1.000 [1.000, 1.000] | 0.042 [0.016, 0.074] | 0.845 [0.706, 0.954] | 0.572 [0.117, 0.697] |

n = 1680 paths (378 positive, 1302 negative); multi-model row uses 2268 two-destination items (378 positive). Calibration: leave_one_program_out/cycle_aware; FA matched at retention 0.9. NOTE: under 'cycle_aware' the positive-retention floor exceeded the false-acceptance cap for 6 of 6 programs, so these thresholds are floored, not capped, and the realised FA may exceed the 5% calibration budget: frac_prevs (shape/magnitude/scalar); running_max (shape/magnitude/scalar); token_hist (shape/magnitude/scalar); dyck1_balance (shape/magnitude/scalar); sort_key (shape/magnitude/scalar); induction_copy (shape/magnitude/scalar). The controls_only mode is reported alongside, and the matched-retention column is threshold-free.


**table_5_1b_contrasts**

Planned paired contrasts on false acceptance at matched positive retention. Negative estimates favour the first rule. Holm correction covers the seven planned contrasts.

| Contrast (FA @ matched retention)  | Estimate 95% CI         | p      | p (Holm) |
|------------------------------------|-------------------------|--------|----------|
| Forward only - Representational    | 0.188 [0.027, 0.323]    | 0.0198 | 0.0792   |
| Original scalar CCC - Forward only | -0.283 [-0.405, -0.151] | 0.0000 | 0.0000   |
| CCC-Ind - Original scalar CCC      | 0.014 [-0.078, 0.097]   | 0.8738 | 1.0000   |
| CCC-Sig - CCC-Ind                  | -0.393 [-0.576, -0.238] | 0.0000 | 0.0000   |
| Pairwise CCC+ - CCC-Sig            | 0.191 [-0.032, 0.352]   | 0.1480 | 0.4440   |
| Pairwise CCC+ - Forward only       | -0.472 [-0.727, -0.348] | 0.0000 | 0.0000   |
| Multi-model CCC+ - Pairwise CCC+   | -0.028 [-0.221, 0.196]  | 0.6848 | 1.0000   |


**coupled_vs_independent**

Coupled versus independently estimated return legs.

| Translator | Coupled pass (pos) | Indep. pass (pos) | Coupled pass (neg) | Indep. pass (neg) | Coupled ret. cos | Indep. ret. cos |
|------------|--------------------|-------------------|--------------------|-------------------|------------------|-----------------|
| procrustes | 0.815              | 0.815             | 0.344              | 0.333             | 0.403            | 0.405           |
| crosscoder | 0.951              | 0.914             | 0.819              | 0.846             | 0.520            | 0.525           |
| dfc        | 0.920              | 0.938             | 0.814              | 0.815             | 0.545            | 0.546           |

A coupled return that passes on negatives as readily as on positives carries no evidence: recovery is automatic by construction.


**set_valued_cases**

Planted case types. Split and merged variables are set-valued: a single component cannot be the whole counterpart.

| Planted case | n   | Contexts | Subspace recall | Subspace precision | Set exact | Joint signature pass | Component-level pass |
|--------------|-----|----------|-----------------|--------------------|-----------|----------------------|----------------------|
| permuted     | 126 | 126      | 0.738           | 0.553              | 0.389     | 0.738                | 0.738                |
| split_half   | 84  | 42       | 0.119           | 0.155              | 0.000     | n/a                  | 0.119                |
| shared       | 84  | 84       | 0.810           | 0.663              | 0.536     | 0.810                | 0.810                |
| redundant    | 84  | 42       | 0.643           | 0.694              | 0.262     | 0.643                | 0.643                |
| merged       | 84  | 84       | 0.655           | 0.619              | 0.583     | 0.655                | 0.655                |


**table_5_4_signatures**

Prompt-level causal agreement for accepted correspondences. Corr. is effect correlation, Sign is prompt-level sign agreement.

| Translator | Corr. | Sign  | Effect ratio | Worst error | Calib. slope |
|------------|-------|-------|--------------|-------------|--------------|
| Procrustes | 0.383 | 0.751 | 0.316        | 1.504       | 0.158        |
| Crosscoder | 0.422 | 0.850 | 0.390        | 1.255       | 0.218        |
| DFC        | 0.442 | 0.810 | 0.421        | 1.359       | 0.205        |


**part_b_translator_rules**

Translator-proposed candidates. Coverage and abstention are measured here, since candidates come from the translators themselves.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.059 [0.000, 0.500] | 0.987 [0.969, 1.000] | 0.024 [0.007, 0.046] | 0.584 [0.235, 0.931] | 0.885 [0.089, 0.935] |
| Round trip only     | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.972 [0.946, 0.992] | 0.370 [0.175, 0.882] | 0.866 [0.150, 0.927] |
| Forward only        | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.801 [0.690, 0.883] | 0.600 [0.558, 0.655] | 0.801 [0.691, 0.883] |
| Original scalar CCC | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.789 [0.679, 0.871] | 0.501 [0.336, 0.903] | 0.700 [0.124, 0.790] |
| CCC-Ind             | 0.882 [0.667, 1.000] | 0.987 [0.969, 1.000] | 0.631 [0.526, 0.716] | 0.647 [0.507, 0.970] | 0.665 [0.039, 0.733] |
| CCC-Shape           | 0.588 [0.125, 1.000] | 0.987 [0.969, 1.000] | 0.175 [0.103, 0.252] | 0.808 [0.661, 0.954] | 0.339 [0.065, 0.525] |
| CCC-Sig             | 0.294 [0.000, 1.000] | 0.987 [0.969, 1.000] | 0.120 [0.063, 0.184] | 0.764 [0.596, 0.945] | 0.457 [0.077, 0.560] |
| Pairwise CCC+       | 0.235 [0.000, 0.714] | 0.987 [0.969, 1.000] | 0.109 [0.058, 0.168] | 0.795 [0.651, 0.929] | 0.401 [0.116, 0.503] |

n = 840 paths (17 positive, 823 negative).


**table_5_1_controlled__controls_only**

Secondary bound mode 'controls_only'. Same candidates, same signatures, same tau_min as the primary table; only the equivalence bounds differ, so the two are directly comparable.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.048 [0.007, 0.095] | 0.581 [0.482, 0.686] | 0.742 [0.604, 0.903] |
| Round trip only     | 0.056 [0.008, 0.127] | 1.000 [1.000, 1.000] | 0.048 [0.010, 0.105] | 0.641 [0.571, 0.709] | 0.746 [0.610, 0.870] |
| Forward only        | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.930 [0.872, 0.971] | 0.535 [0.514, 0.564] | 0.930 [0.872, 0.971] |
| Original scalar CCC | 0.056 [0.008, 0.127] | 1.000 [1.000, 1.000] | 0.038 [0.005, 0.090] | 0.670 [0.603, 0.737] | 0.694 [0.557, 0.816] |
| CCC-Ind             | 0.045 [0.003, 0.105] | 1.000 [1.000, 1.000] | 0.039 [0.005, 0.091] | 0.674 [0.604, 0.742] | 0.697 [0.547, 0.816] |
| CCC-Shape           | 0.021 [0.000, 0.067] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.678 [0.608, 0.746] | 0.697 [0.547, 0.816] |
| CCC-Sig             | 0.016 [0.000, 0.059] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.896 [0.841, 0.944] | 0.288 [0.155, 0.406] |
| Pairwise CCC+       | 0.019 [0.000, 0.065] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.864 [0.798, 0.921] | 0.402 [0.206, 0.556] |
| Multi-model CCC+    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.850 [0.715, 0.952] | 0.514 [0.124, 0.694] |

n = 1680 paths. Calibration: leave_one_program_out/controls_only. The false-acceptance cap held for every program.


**fa_by_negative_case**

False acceptance by kind of negative. The negative classes are not equally hard: an absent mechanism has no causal effect at all, whereas an effect-matched random subspace is built to survive a mean-effect test.

| Validation rule   | wrong variable | effect-matched random | absent | split half |
|-------------------|----------------|-----------------------|--------|------------|
| representational  | 0.044          | 0.019                 | 0.000  | 0.250      |
| round_trip_only   | 0.817          | 0.638                 | 0.929  | 0.512      |
| forward_only      | 0.947          | 1.000                 | 0.000  | 0.917      |
| scalar_ccc        | 0.788          | 0.638                 | 0.000  | 0.440      |
| ccc_ind           | 0.810          | 0.606                 | 0.000  | 0.524      |
| ccc_shape         | 0.281          | 0.071                 | 0.000  | 0.167      |
| ccc_sig           | 0.242          | 0.000                 | 0.000  | 0.167      |
| pairwise_ccc_plus | 0.199          | 0.000                 | 0.000  | 0.119      |

