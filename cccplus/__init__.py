"""CCC+ : causal cycle consistency with independent causal signatures.

Reference implementation of the method and experiments described in
"CCC+: Independent Causal Signatures for Cross-Model Mechanism Translation".

Module map (paper section in parentheses):

  mechanisms    component mechanism m = (s, V)                       (3.1)
  interventions directional counterfactual patch and robustness arms (3.2)
  signature     causal effects, standardised signatures, d_shape/d_mag (3.1)
  translators/  cosine alignment, crosscoder, DFC, causal baselines  (3.3, 4.5)
  calibration   tau_min, c_{M,T}, equivalence bounds                 (3.6)
  validation    forward gate, destination test, independent round trip (3.4, 3.5)
  circuits      multi-site extension                                 (3.7)
  benchmarks/   planted ground truth, IOI, factual recall            (4.1, 4.2)
  stats         hierarchical bootstrap, Holm, AUROC, matched retention (3.6, 4.1)
"""

__version__ = "1.0.0"
