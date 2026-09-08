# Statistical and computational notes

## Classic projection correlation

For each anchor observation, the implementation forms the angular matrices,
sets the anchor row and column to zero, double-centres them, and accumulates
their Frobenius inner product. Anchors are streamed, so working memory is
quadratic rather than an `n x n x n` tensor. Time remains
`O(n^3 (p + q))`; each permutation repeats the calculation.

The test statistic is `n * Pcov_hat^2 / (pi^2 - S2_hat)`. Zhu et al. (2017)
used an uncorrected fraction from 2,000 permutation replicates. `projCorr`
uses `(1 + exceedances) / (B + 1)` and counts ties in the upper tail.

## Improved projection correlation

The pairwise kernel is

```
acos((sigma2 + Zi'Zj) /
     sqrt((sigma2 + ||Zi||^2) * (sigma2 + ||Zj||^2))).
```

Cosines are clamped to `[-1, 1]`. Squared scales default to median squared row
norms. The distinct-index U-statistic is reduced to pairwise matrix sums and
therefore takes `O(n^2 (p + q))` time and `O(n^2)` memory. The reduction is
tested against direct four-index enumeration. The publisher's supplementary
file containing its derivation was unavailable during development, so the
missing source was treated as optimization-critical rather than
definition-critical.

The unbiased squared estimator can be negative in finite samples. It is never
silently truncated. Its square-root IPC estimate is reported as `NA` when
materially negative, while the signed squared value and test statistic remain
available whenever the self-IPCov denominator is positive.

Improved IPC and IPC-RCV default to the paper's high-dimensional normal limits.
The package cannot verify Conditions 1--4 from the supplied matrices. For this
reason, `p_method = "auto"` conservatively selects permutation inference even
though the explicit default is `p_method = "asymptotic"`.

## Refitted cross-validation

The implementation follows Algorithm 1 of Zhang and Zhu (2024). It randomly
splits an even sample, screens all marginal coordinate pairs on one half,
evaluates the selected coordinates on the other half, reverses the roles, and
averages the two squared IPC values. Duplicate selected coordinates are
collapsed. Permutation inference fixes the observed split and repeats the full
screening and cross-fitting procedure for each permuted response sample.

## Benchmark

On macOS arm64, R 4.5.1 and Apple clang 17, elapsed seconds were:

| Calculation | Pure R | Compiled |
|---|---:|---:|
| Classic components, `n=60, p=5, q=4` | 0.019 | 0.005 |
| IPC quantities, `n=800, p=q=10` | 0.049 | 0.021 |
| Marginal screening, `n=100, p=q=40` | 0.129 | 0.029 |

These are representative measurements, not performance guarantees. Serial
execution is always available. OpenMP, where supported, is used only over
independent ordinary PC/IPC permutations; RCV permutations remain serial.

