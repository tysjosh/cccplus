# Controlled ground truth

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
| Round trip only     | 0.884 [0.790, 0.953] | 1.000 [1.000, 1.000] | 0.667 [0.577, 0.755] | 0.668 [0.599, 0.730] | 0.701 [0.533, 0.804] |
| Forward only        | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.925 [0.869, 0.963] | 0.538 [0.518, 0.565] | 0.925 [0.869, 0.963] |
| Original scalar CCC | 0.884 [0.790, 0.953] | 1.000 [1.000, 1.000] | 0.608 [0.515, 0.695] | 0.703 [0.638, 0.764] | 0.638 [0.475, 0.740] |
| CCC-Ind             | 0.870 [0.766, 0.950] | 1.000 [1.000, 1.000] | 0.599 [0.508, 0.686] | 0.703 [0.637, 0.766] | 0.661 [0.467, 0.772] |
| CCC-Shape           | 0.870 [0.766, 0.950] | 1.000 [1.000, 1.000] | 0.185 [0.122, 0.246] | 0.930 [0.882, 0.966] | 0.246 [0.068, 0.409] |
| CCC-Sig             | 0.870 [0.766, 0.950] | 1.000 [1.000, 1.000] | 0.146 [0.088, 0.208] | 0.951 [0.915, 0.978] | 0.190 [0.043, 0.289] |
| Pairwise CCC+       | 0.743 [0.607, 0.862] | 1.000 [1.000, 1.000] | 0.108 [0.061, 0.161] | 0.922 [0.874, 0.959] | 0.201 [0.119, 0.419] |
| Multi-model CCC+    | 0.608 [0.360, 0.808] | 1.000 [1.000, 1.000] | 0.030 [0.007, 0.063] | 0.941 [0.846, 0.986] | 0.248 [0.033, 0.433] |

n = 1680 paths (378 positive, 1302 negative); multi-model row uses 2268 two-destination items (378 positive). Calibration: leave_one_program_out/cycle_aware; FA matched at retention 0.9. NOTE: under 'cycle_aware' the positive-retention floor exceeded the false-acceptance cap for 6 of 6 programs, so these thresholds are floored, not capped, and the realised FA may exceed the 5% calibration budget: frac_prevs (shape/magnitude/scalar); running_max (shape/magnitude/scalar); token_hist (shape/magnitude/scalar); dyck1_balance (shape/magnitude/scalar); sort_key (shape/magnitude/scalar); induction_copy (shape/magnitude/scalar). The controls_only mode is reported alongside, and the matched-retention column is threshold-free.


**table_5_1b_contrasts**

Planned paired contrasts on false acceptance at matched positive retention. Negative estimates favour the first rule. Holm correction covers the seven planned contrasts.

| Contrast (FA @ matched retention)  | Estimate 95% CI         | p      | p (Holm) |
|------------------------------------|-------------------------|--------|----------|
| Forward only - Representational    | 0.183 [0.025, 0.318]    | 0.0242 | 0.0968   |
| Original scalar CCC - Forward only | -0.286 [-0.450, -0.186] | 0.0000 | 0.0000   |
| CCC-Ind - Original scalar CCC      | 0.022 [-0.100, 0.118]   | 0.7812 | 0.7812   |
| CCC-Sig - CCC-Ind                  | -0.470 [-0.560, -0.372] | 0.0000 | 0.0000   |
| Pairwise CCC+ - CCC-Sig            | 0.011 [-0.017, 0.163]   | 0.2072 | 0.6216   |
| Pairwise CCC+ - Forward only       | -0.724 [-0.805, -0.505] | 0.0000 | 0.0000   |
| Multi-model CCC+ - Pairwise CCC+   | -0.050 [-0.212, 0.075]  | 0.2558 | 0.6216   |


**coupled_vs_independent**

Coupled versus independently estimated return legs.

| Translator | Coupled pass (pos) | Indep. pass (pos) | Coupled pass (neg) | Indep. pass (neg) | Coupled ret. cos | Indep. ret. cos |
|------------|--------------------|-------------------|--------------------|-------------------|------------------|-----------------|
| procrustes | 0.778              | 0.759             | 0.237              | 0.220             | 0.403            | 0.405           |
| crosscoder | 0.901              | 0.864             | 0.740              | 0.731             | 0.520            | 0.525           |
| dfc        | 0.901              | 0.914             | 0.737              | 0.731             | 0.545            | 0.546           |

A coupled return that passes on negatives as readily as on positives carries no evidence: recovery is automatic by construction.


**set_valued_cases**

Planted case types. Split and merged variables are set-valued: a single component cannot be the whole counterpart.

| Planted case | n   | Contexts | Subspace recall | Subspace precision | Set exact | Joint signature pass | Component-level pass |
|--------------|-----|----------|-----------------|--------------------|-----------|----------------------|----------------------|
| permuted     | 126 | 126      | 0.730           | 0.598              | 0.468     | 0.730                | 0.730                |
| split_half   | 84  | 42       | 0.095           | 0.179              | 0.000     | n/a                  | 0.095                |
| shared       | 84  | 84       | 0.714           | 0.619              | 0.524     | 0.714                | 0.714                |
| redundant    | 84  | 42       | 0.810           | 0.786              | 0.452     | 0.810                | 0.810                |
| merged       | 84  | 84       | 0.726           | 0.625              | 0.524     | 0.726                | 0.726                |


**table_5_4_signatures**

Prompt-level causal agreement for accepted correspondences. Corr. is effect correlation, Sign is prompt-level sign agreement.

| Translator | Corr. | Sign  | Effect ratio | Worst error | Calib. slope |
|------------|-------|-------|--------------|-------------|--------------|
| Procrustes | 0.690 | 0.968 | 0.509        | 0.591       | 0.709        |
| Crosscoder | 0.500 | 0.946 | 0.493        | 0.662       | 0.468        |
| DFC        | 0.592 | 0.980 | 0.565        | 0.573       | 0.613        |


**part_b_translator_rules**

Translator-proposed candidates. Coverage and abstention are measured here, since candidates come from the translators themselves.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.059 [0.000, 0.500] | 0.987 [0.969, 1.000] | 0.024 [0.007, 0.046] | 0.584 [0.235, 0.931] | 0.885 [0.089, 0.935] |
| Round trip only     | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.955 [0.922, 0.982] | 0.428 [0.219, 0.886] | 0.849 [0.156, 0.899] |
| Forward only        | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.875 [0.768, 0.947] | 0.563 [0.527, 0.615] | 0.875 [0.769, 0.947] |
| Original scalar CCC | 1.000 [1.000, 1.000] | 0.987 [0.969, 1.000] | 0.849 [0.744, 0.923] | 0.501 [0.309, 0.905] | 0.750 [0.130, 0.821] |
| CCC-Ind             | 0.941 [0.778, 1.000] | 0.987 [0.969, 1.000] | 0.622 [0.523, 0.701] | 0.651 [0.506, 0.974] | 0.610 [0.032, 0.745] |
| CCC-Shape           | 0.412 [0.000, 1.000] | 0.987 [0.969, 1.000] | 0.236 [0.164, 0.311] | 0.738 [0.595, 0.967] | 0.512 [0.048, 0.737] |
| CCC-Sig             | 0.412 [0.000, 1.000] | 0.987 [0.969, 1.000] | 0.148 [0.085, 0.219] | 0.825 [0.707, 0.978] | 0.320 [0.030, 0.541] |
| Pairwise CCC+       | 0.353 [0.000, 1.000] | 0.987 [0.969, 1.000] | 0.131 [0.073, 0.197] | 0.824 [0.724, 0.978] | 0.324 [0.030, 0.498] |

n = 840 paths (17 positive, 823 negative).


**table_5_1_controlled__controls_only**

Secondary bound mode 'controls_only'. Same candidates, same signatures, same tau_min as the primary table; only the equivalence bounds differ, so the two are directly comparable.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.048 [0.007, 0.095] | 0.581 [0.482, 0.686] | 0.742 [0.604, 0.903] |
| Round trip only     | 0.061 [0.011, 0.133] | 1.000 [1.000, 1.000] | 0.047 [0.010, 0.103] | 0.669 [0.600, 0.731] | 0.710 [0.534, 0.801] |
| Forward only        | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.925 [0.869, 0.964] | 0.538 [0.518, 0.565] | 0.925 [0.869, 0.964] |
| Original scalar CCC | 0.061 [0.011, 0.133] | 1.000 [1.000, 1.000] | 0.038 [0.005, 0.087] | 0.705 [0.640, 0.766] | 0.648 [0.474, 0.740] |
| CCC-Ind             | 0.069 [0.012, 0.145] | 1.000 [1.000, 1.000] | 0.038 [0.004, 0.091] | 0.704 [0.637, 0.767] | 0.671 [0.459, 0.763] |
| CCC-Shape           | 0.058 [0.004, 0.135] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.714 [0.645, 0.778] | 0.671 [0.459, 0.763] |
| CCC-Sig             | 0.019 [0.000, 0.060] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.892 [0.841, 0.934] | 0.332 [0.159, 0.424] |
| Pairwise CCC+       | 0.024 [0.000, 0.069] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.918 [0.871, 0.955] | 0.233 [0.106, 0.410] |
| Multi-model CCC+    | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.941 [0.851, 0.986] | 0.237 [0.033, 0.427] |

n = 1680 paths. Calibration: leave_one_program_out/controls_only. The false-acceptance cap held for every program.


**fa_by_negative_case**

False acceptance by kind of negative. The negative classes are not equally hard: an absent mechanism has no causal effect at all, whereas an effect-matched random subspace is built to survive a mean-effect test.

| Validation rule   | wrong variable | effect-matched random | absent | split half |
|-------------------|----------------|-----------------------|--------|------------|
| representational  | 0.044          | 0.019                 | 0.000  | 0.250      |
| round_trip_only   | 0.733          | 0.566                 | 0.810  | 0.417      |
| forward_only      | 0.939          | 1.000                 | 0.000  | 0.917      |
| scalar_ccc        | 0.684          | 0.566                 | 0.000  | 0.369      |
| ccc_ind           | 0.688          | 0.534                 | 0.000  | 0.345      |
| ccc_shape         | 0.273          | 0.016                 | 0.000  | 0.202      |
| ccc_sig           | 0.217          | 0.000                 | 0.000  | 0.202      |
| pairwise_ccc_plus | 0.167          | 0.000                 | 0.000  | 0.095      |

