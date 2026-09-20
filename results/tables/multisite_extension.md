# Multisite extension

manifest `e5c891d9f85101e4`

**multisite_extension**

Multi-site circuit extension. Node and edge recovery are evaluated against planted circuits; the interaction error is the relative discrepancy in Gamma.

| IOI extension        | Node  | Edge  | Interact. err. |
|----------------------|-------|-------|----------------|
| Nodewise translation | 1.000 | —     | —              |
| Multi-site CCC+      | 1.000 | 0.333 | 1.193          |


**multisite_separation**

Separation between the true circuit and a node-permuted circuit occupying the same sites. A larger gap means the signature distinguishes wiring, not just location.

| Signature used       | d_shape (true circuit) | d_shape (node-permuted) | separation |
|----------------------|------------------------|-------------------------|------------|
| Nodewise translation | 0.440                  | 0.505                   | 0.065      |
| Multi-site CCC+      | 0.510                  | 0.565                   | 0.055      |

