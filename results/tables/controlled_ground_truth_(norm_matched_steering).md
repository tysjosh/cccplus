# Controlled ground truth (norm_matched_steering)

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
| Round trip only     | 0.892 [0.798, 0.962] | 1.000 [1.000, 1.000] | 0.750 [0.668, 0.828] | 0.638 [0.569, 0.702] | 0.770 [0.597, 0.880] |
| Forward only        | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.892 [0.819, 0.945] | 0.554 [0.528, 0.590] | 0.892 [0.819, 0.945] |
| Original scalar CCC | 0.892 [0.798, 0.962] | 1.000 [1.000, 1.000] | 0.667 [0.571, 0.754] | 0.678 [0.611, 0.741] | 0.685 [0.520, 0.799] |
| CCC-Ind             | 0.899 [0.807, 0.971] | 1.000 [1.000, 1.000] | 0.653 [0.546, 0.745] | 0.673 [0.599, 0.743] | 0.654 [0.518, 0.786] |
| CCC-Shape           | 0.741 [0.600, 0.866] | 1.000 [1.000, 1.000] | 0.267 [0.180, 0.357] | 0.787 [0.711, 0.858] | 0.498 [0.315, 0.660] |
| CCC-Sig             | 0.704 [0.562, 0.833] | 1.000 [1.000, 1.000] | 0.152 [0.082, 0.228] | 0.877 [0.811, 0.930] | 0.267 [0.136, 0.380] |
| Pairwise CCC+       | 0.582 [0.422, 0.732] | 1.000 [1.000, 1.000] | 0.121 [0.064, 0.184] | 0.860 [0.793, 0.915] | 0.334 [0.198, 0.460] |
| Multi-model CCC+    | 0.368 [0.121, 0.648] | 1.000 [1.000, 1.000] | 0.028 [0.003, 0.063] | 0.902 [0.797, 0.965] | 0.277 [0.076, 0.465] |

n = 1680 paths (378 positive, 1302 negative); multi-model row uses 2268 two-destination items (378 positive). Calibration: leave_one_program_out/cycle_aware; FA matched at retention 0.9. NOTE: under 'cycle_aware' the positive-retention floor exceeded the false-acceptance cap for 6 of 6 programs, so these thresholds are floored, not capped, and the realised FA may exceed the 5% calibration budget: frac_prevs (shape/magnitude/scalar); running_max (shape/magnitude/scalar); token_hist (shape/magnitude/scalar); dyck1_balance (shape/magnitude/scalar); sort_key (shape/magnitude/scalar); induction_copy (shape/magnitude/scalar). The controls_only mode is reported alongside, and the matched-retention column is threshold-free.


**table_5_1b_contrasts**

Planned paired contrasts on false acceptance at matched positive retention. Negative estimates favour the first rule. Holm correction covers the seven planned contrasts.

| Contrast (FA @ matched retention)  | Estimate 95% CI         | p      | p (Holm) |
|------------------------------------|-------------------------|--------|----------|
| Forward only - Representational    | 0.151 [-0.011, 0.292]   | 0.0652 | 0.1956   |
| Original scalar CCC - Forward only | -0.207 [-0.367, -0.102] | 0.0000 | 0.0000   |
| CCC-Ind - Original scalar CCC      | -0.031 [-0.111, 0.079]  | 0.7482 | 0.7482   |
| CCC-Sig - CCC-Ind                  | -0.387 [-0.512, -0.264] | 0.0000 | 0.0000   |
| Pairwise CCC+ - CCC-Sig            | 0.068 [-0.042, 0.212]   | 0.2914 | 0.5828   |
| Pairwise CCC+ - Forward only       | -0.558 [-0.694, -0.431] | 0.0000 | 0.0000   |
| Multi-model CCC+ - Pairwise CCC+   | -0.154 [-0.257, -0.004] | 0.0468 | 0.1872   |


**coupled_vs_independent**

Coupled versus independently estimated return legs.

| Translator | Coupled pass (pos) | Indep. pass (pos) | Coupled pass (neg) | Indep. pass (neg) | Coupled ret. cos | Indep. ret. cos |
|------------|--------------------|-------------------|--------------------|-------------------|------------------|-----------------|
| procrustes | 0.759              | 0.759             | 0.371              | 0.355             | 0.403            | 0.405           |
| crosscoder | 0.926              | 0.920             | 0.830              | 0.828             | 0.520            | 0.525           |
| dfc        | 0.901              | 0.926             | 0.796              | 0.762             | 0.545            | 0.546           |

A coupled return that passes on negatives as readily as on positives carries no evidence: recovery is automatic by construction.


**set_valued_cases**

Planted case types. Split and merged variables are set-valued: a single component cannot be the whole counterpart.

| Planted case | n   | Contexts | Subspace recall | Subspace precision | Set exact | Joint signature pass | Component-level pass |
|--------------|-----|----------|-----------------|--------------------|-----------|----------------------|----------------------|
| permuted     | 126 | 126      | 0.571           | 0.495              | 0.421     | 0.571                | 0.571                |
| split_half   | 84  | 42       | 0.155           | 0.240              | 0.000     | n/a                  | 0.155                |
| shared       | 84  | 84       | 0.405           | 0.351              | 0.298     | 0.405                | 0.405                |
| redundant    | 84  | 42       | 0.798           | 0.885              | 0.476     | 0.798                | 0.798                |
| merged       | 84  | 84       | 0.560           | 0.506              | 0.452     | 0.560                | 0.560                |


**table_5_4_signatures**

Prompt-level causal agreement for accepted correspondences. Corr. is effect correlation, Sign is prompt-level sign agreement.

| Translator | Corr. | Sign  | Effect ratio | Worst error | Calib. slope |
|------------|-------|-------|--------------|-------------|--------------|
| Procrustes | 0.356 | 0.772 | 0.476        | 1.177       | 0.177        |
| Crosscoder | 0.480 | 0.768 | 0.645        | 1.068       | 0.331        |
| DFC        | 0.355 | 0.762 | 0.478        | 1.056       | 0.208        |


**part_b_translator_rules**

Translator-proposed candidates. Coverage and abstention are measured here, since candidates come from the translators themselves.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.059 [0.000, 0.500] | 0.987 [0.969, 1.000] | 0.024 [0.007, 0.046] | 0.584 [0.235, 0.931] | 0.885 [0.089, 0.935] |
| Round trip only     | 0.941 [0.600, 1.000] | 0.987 [0.969, 1.000] | 0.962 [0.926, 0.990] | 0.424 [0.156, 0.882] | 0.927 [0.174, 0.991] |
| Forward only        | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.868 [0.763, 0.938] | 0.566 [0.531, 0.618] | 0.868 [0.764, 0.938] |
| Original scalar CCC | 0.941 [0.600, 1.000] | 0.987 [0.969, 1.000] | 0.844 [0.739, 0.917] | 0.498 [0.255, 0.901] | 0.814 [0.141, 0.911] |
| CCC-Ind             | 0.882 [0.572, 1.000] | 0.987 [0.969, 1.000] | 0.695 [0.593, 0.775] | 0.627 [0.409, 0.933] | 0.781 [0.112, 0.848] |
| CCC-Shape           | 0.588 [0.000, 1.000] | 0.987 [0.969, 1.000] | 0.311 [0.220, 0.400] | 0.708 [0.518, 0.863] | 0.667 [0.184, 0.799] |
| CCC-Sig             | 0.235 [0.000, 0.750] | 0.987 [0.969, 1.000] | 0.168 [0.102, 0.239] | 0.780 [0.667, 0.895] | 0.358 [0.170, 0.455] |
| Pairwise CCC+       | 0.235 [0.000, 0.750] | 0.987 [0.969, 1.000] | 0.142 [0.085, 0.203] | 0.812 [0.728, 0.910] | 0.267 [0.140, 0.475] |

n = 840 paths (17 positive, 823 negative).


**table_5_1_controlled__controls_only**

Secondary bound mode 'controls_only'. Same candidates, same signatures, same tau_min as the primary table; only the equivalence bounds differ, so the two are directly comparable.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.048 [0.007, 0.095] | 0.581 [0.482, 0.686] | 0.742 [0.604, 0.903] |
| Round trip only     | 0.085 [0.026, 0.160] | 1.000 [1.000, 1.000] | 0.062 [0.022, 0.121] | 0.636 [0.569, 0.702] | 0.779 [0.612, 0.878] |
| Forward only        | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.892 [0.817, 0.946] | 0.554 [0.527, 0.591] | 0.892 [0.817, 0.946] |
| Original scalar CCC | 0.085 [0.026, 0.160] | 1.000 [1.000, 1.000] | 0.053 [0.017, 0.105] | 0.676 [0.609, 0.742] | 0.697 [0.534, 0.799] |
| CCC-Ind             | 0.082 [0.021, 0.162] | 1.000 [1.000, 1.000] | 0.048 [0.013, 0.100] | 0.671 [0.596, 0.742] | 0.670 [0.519, 0.789] |
| CCC-Shape           | 0.037 [0.000, 0.104] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.675 [0.599, 0.747] | 0.670 [0.519, 0.789] |
| CCC-Sig             | 0.016 [0.000, 0.064] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.848 [0.777, 0.908] | 0.367 [0.223, 0.512] |
| Pairwise CCC+       | 0.021 [0.000, 0.079] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.870 [0.807, 0.923] | 0.331 [0.183, 0.452] |
| Multi-model CCC+    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.898 [0.790, 0.972] | 0.270 [0.076, 0.440] |

n = 1680 paths. Calibration: leave_one_program_out/controls_only. The false-acceptance cap held for every program.


**fa_by_negative_case**

False acceptance by kind of negative. The negative classes are not equally hard: an absent mechanism has no causal effect at all, whereas an effect-matched random subspace is built to survive a mean-effect test.

| Validation rule   | wrong variable | effect-matched random | absent | split half |
|-------------------|----------------|-----------------------|--------|------------|
| representational  | 0.044          | 0.019                 | 0.000  | 0.250      |
| round_trip_only   | 0.810          | 0.624                 | 0.857  | 0.690      |
| forward_only      | 0.904          | 0.963                 | 0.000  | 0.917      |
| scalar_ccc        | 0.737          | 0.606                 | 0.000  | 0.619      |
| ccc_ind           | 0.731          | 0.566                 | 0.000  | 0.631      |
| ccc_shape         | 0.350          | 0.130                 | 0.000  | 0.226      |
| ccc_sig           | 0.233          | 0.000                 | 0.000  | 0.143      |
| pairwise_ccc_plus | 0.180          | 0.000                 | 0.000  | 0.155      |

