# Natural models tiny

manifest `e5c891d9f85101e4`

**table_5_1_natural_tiny**

Natural-model results (tiny stratum): validation rules on pretrained checkpoints, pooled over the six directed mappings.

| Validation rule     | Ret.                 | Cov.                 | FA                   | AUROC                | FA @ matched Ret.    |
|---------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Representational    | 0.006 [0.000, 0.038] | 0.897 [0.679, 0.965] | 0.000 [0.000, 0.000] | 0.628 [0.512, 0.703] | 1.000 [1.000, 1.000] |
| Round trip only     | 0.256 [0.023, 0.406] | 0.897 [0.679, 0.965] | 0.256 [0.023, 0.436] | 0.510 [0.456, 0.563] | 1.000 [1.000, 1.000] |
| Forward only        | 0.071 [0.000, 0.151] | 0.897 [0.679, 0.965] | 0.154 [0.000, 0.271] | 0.484 [0.402, 0.540] | 1.000 [1.000, 1.000] |
| Original scalar CCC | 0.071 [0.000, 0.151] | 0.897 [0.679, 0.965] | 0.141 [0.000, 0.259] | 0.500 [0.438, 0.552] | 1.000 [1.000, 1.000] |
| CCC-Ind             | 0.064 [0.000, 0.143] | 0.897 [0.679, 0.965] | 0.135 [0.000, 0.240] | 0.501 [0.448, 0.553] | 1.000 [1.000, 1.000] |
| CCC-Shape           | 0.000 [0.000, 0.000] | 0.897 [0.679, 0.965] | 0.000 [0.000, 0.000] | 0.509 [0.457, 0.562] | 1.000 [1.000, 1.000] |
| CCC-Sig             | 0.000 [0.000, 0.000] | 0.897 [0.679, 0.965] | 0.000 [0.000, 0.000] | 0.507 [0.455, 0.560] | 1.000 [1.000, 1.000] |
| Pairwise CCC+       | 0.000 [0.000, 0.000] | 0.897 [0.679, 0.965] | 0.000 [0.000, 0.000] | 0.506 [0.453, 0.558] | 1.000 [1.000, 1.000] |

n = 312 paths (156 translated, 156 control).


**table_5_2_natural_tiny**

Held-out correspondence and coverage by translator. Coverage and abstention are reported together.

| Translator | Retention | FA    | Cov.  | Abstention |
|------------|-----------|-------|-------|------------|
| Procrustes | 0.000     | 0.000 | 0.923 | 0.077      |
| Crosscoder | 0.000     | 0.000 | 0.923 | 0.077      |
| DFC        | 0.000     | 0.000 | 0.846 | 0.154      |


**table_5_4_natural_tiny**

Prompt-level causal agreement on pretrained models.

| Translator | Corr. | Sign  | Effect ratio | Worst error | Calib. slope |
|------------|-------|-------|--------------|-------------|--------------|
| Procrustes | 0.033 | 0.532 | 1.132        | 2.606       | 0.069        |
| Crosscoder | 0.050 | 0.493 | 0.502        | 1.670       | 0.019        |
| DFC        | 0.078 | 0.453 | 0.668        | 1.944       | 0.151        |


**coupling_natural_tiny**

Coupled versus independent return legs on pretrained models.

| Translator | Coupled pass (pos) | Indep. pass (pos) | Coupled pass (neg) | Indep. pass (neg) | Coupled ret. cos | Indep. ret. cos |
|------------|--------------------|-------------------|--------------------|-------------------|------------------|-----------------|
| procrustes | 1.000              | 1.000             | 0.846              | 0.846             | 0.746            | 0.672           |
| crosscoder | 1.000              | 1.000             | 0.846              | 0.846             | 0.484            | 0.431           |
| dfc        | 1.000              | 0.861             | 0.577              | 0.596             | 0.488            | 0.434           |

A coupled return that passes on negatives as readily as on positives carries no evidence: recovery is automatic by construction.

