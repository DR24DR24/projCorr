# projCorr

`projCorr` is an R package for measuring and testing unconditional dependence
between two random vectors. It implements the classic projection correlation
of Zhu et al. (2017), the improved projection correlation of Zhang and Zhu
(2024), and the latter paper's refitted cross-validation procedure for
high-dimensional data.

Projection-based methods can detect general nonlinear dependence. Their
angular kernels are bounded, so the procedures do not require finite moments
of the original random vectors. The package accepts paired numeric matrices or
numeric data frames and returns familiar `htest` objects.

## Authors and license

Authors: Wang and Chia <xzm1@ruc.edu.cn>

License: MIT

## Installation

### Install from GitHub (recommended)

The lightweight [`remotes`](https://cran.r-project.org/package=remotes)
package is sufficient for installing `projCorr` from GitHub:

```r
install.packages("remotes")
remotes::install_github("Chia202/projCorr")
```

The equivalent `devtools` command is:

```r
install.packages("devtools")
devtools::install_github("Chia202/projCorr")
```

`remotes` is preferred when installation is the only goal because it has fewer
dependencies. `devtools` is useful when the full package-development toolkit
is also needed.

To install a particular release tag, for example `v0.1.0`, use:

```r
remotes::install_github("Chia202/projCorr@v0.1.0")
```

### Install from a local source archive

```r
install.packages("projCorr_0.1.0.tar.gz", repos = NULL, type = "source")
```

Compilation requires R 4.2 or later, a C++17 compiler, `Rcpp`, and
`RcppArmadillo`. OpenMP is optional, and every method has a serial path.

## Implemented methods

### Classic projection correlation

`pcor_test()` implements Zhu et al. (2017). For each observation used as an
anchor, it constructs and double-centres the sample angular matrices. The
squared sample projection covariance is

```text
Pcov_hat^2 = n^(-3) sum_{k,l,r} A_klr B_klr.
```

The normalized projection correlation and test statistic are

```text
PC_hat = sqrt(Pcov_hat^2 / (Pcov_hat(X,X) Pcov_hat(Y,Y))),
T_n    = n Pcov_hat^2 / (pi^2 - S2_hat).
```

The null distribution contains unknown distribution-dependent weights, so
`pcor_test()` uses permutation calibration. Its direct computation takes
`O(n^3 (p + q))` time and `O(n^2)` working memory.

### Improved projection correlation

`ipcor_test()` implements Zhang and Zhu (2024). For observations `Z_i` and
`Z_j`, it uses the pairwise angular kernel

```text
A(Z_i, Z_j) = acos(
  (sigma_z^2 + Z_i' Z_j) /
  sqrt((sigma_z^2 + ||Z_i||^2) (sigma_z^2 + ||Z_j||^2))
).
```

By default, `sigma_z^2` is the median squared row norm. Users may instead
provide `sigma_x2` and `sigma_y2`. The squared improved projection covariance
is estimated by the paper's distinct-index U-statistic. An algebraically
verified pairwise reduction gives `O(n^2 (p + q))` time and `O(n^2)` memory.

The high-dimensional statistic is

```text
Z_n = n IPC_hat^2 / sqrt(2).
```

Under the null hypothesis and the paper's regularity conditions, `Z_n`
converges to a standard normal distribution when at least one dimension
diverges. The package therefore defaults to `p_method = "asymptotic"`.

These high-dimensional dependence and trace conditions cannot be verified
from the observed matrices alone. When they are doubtful, use
`p_method = "permutation"`. The option `p_method = "auto"` also chooses the
conservative permutation calibration instead of applying an arbitrary
dimension cutoff.

### Refitted cross-validation test

`ipcor_rcv_test()` implements Algorithm 1 of Zhang and Zhu (2024):

1. Randomly divide an even-sized sample into two halves.
2. On the first half, rank all `p * q` coordinate pairs by marginal squared
   IPC and retain the strongest pairs.
3. Evaluate the selected coordinates on the second half to obtain `T1`.
4. Reverse the screening and evaluation halves to obtain `T2`.
5. Return `T = (T1 + T2) / 2` and use `n * T / 2` for normal calibration.

The default number of retained pairs is `ceiling(log(n))`. The default normal
calibration relies on Conditions 3 and 4 and Theorem 6 of the paper. For
permutation calibration, every replicate repeats the complete screening and
cross-fitting procedure while keeping the observed split fixed.

## Quick start

```r
library(projCorr)

set.seed(2024)
n <- 100
x <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
y <- cbind(
  x[, 1]^2 + rnorm(n, sd = 0.3),
  sin(x[, 2]) + rnorm(n, sd = 0.3),
  matrix(rnorm(n * 3), nrow = n, ncol = 3)
)

# Classic projection-correlation permutation test
classic <- pcor_test(x, y, n_perm = 999, seed = 1)
classic

# Improved projection correlation: default normal approximation
improved <- ipcor_test(x, y)
improved

# Improved projection correlation: permutation calibration
improved_perm <- ipcor_test(
  x, y,
  p_method = "permutation",
  n_perm = 999,
  seed = 1
)
improved_perm

# High-dimensional screening and refitted cross-validation
rcv <- ipcor_rcv_test(
  x, y,
  n_pairs = ceiling(log(n)),
  seed = 1
)
rcv
```

An independent heavy-tailed example is:

```r
set.seed(2025)
x0 <- matrix(rnorm(600), nrow = 100, ncol = 6)
y0 <- matrix(rcauchy(400), nrow = 100, ncol = 4)

ipcor_test(x0, y0)
ipcor_test(x0, y0, p_method = "permutation", n_perm = 999, seed = 2)
```

## Function summary

| Function | Estimate | Test statistic | Default calibration |
|---|---|---|---|
| `pcor_test()` | classic `PC` and `Pcov^2` | `n * Pcov^2 / (pi^2 - S2)` | permutation |
| `ipcor_test()` | improved `IPC` and signed `IPCov^2` | `n * IPC^2 / sqrt(2)` | normal approximation |
| `ipcor_rcv_test()` | cross-fitted `T`, `T1`, and `T2` | `n * T / 2` | normal approximation |

Common arguments are:

- `x`, `y`: paired numeric matrices or data frames, with observations in rows;
- `p_method`: `"asymptotic"`, `"permutation"`, or `"auto"` where available;
- `n_perm`: number of sampled non-identity permutations;
- `seed`: optional local random seed;
- `n_threads`: requested thread count for ordinary PC or IPC permutations;
- `na.action`: `"fail"` by default or `"omit"` for paired row omission.

Use `?pcor_test`, `?ipcor_test`, and `?ipcor_rcv_test` for complete argument
documentation.

## Returned values

Each function returns an object compatible with class `htest`. Important fields
include:

- `estimate`: the normalized dependence estimate or RCV statistic;
- `raw_estimate`: the unnormalized covariance estimate or two RCV components;
- `statistic`: the quantity actually used for p-value calibration;
- `p.value`, `null.value`, `alternative`, `method`, and `data.name`;
- `details`: dimensions, scales, selected calibration, permutation count,
  exceedance count, seed, selected pairs, and thread information.

The unbiased squared IPC estimator can be negative in finite samples. The
signed value is retained in `details$ipc_squared` and is never silently
truncated. If its square root is undefined, `estimate["IPC"]` is `NA`, while
the signed test statistic remains available whenever the self-IPCov denominator
is positive.

## Permutation convention and reproducibility

All permutation tests are upper-tailed. Ties count as exceedances, and the
corrected Monte Carlo p-value is

```text
(1 + number of T_perm >= T_obs) / (n_perm + 1).
```

The identity arrangement is represented by the added `1` and is not inserted
into the sampled permutation matrix. Non-identity permutations are sampled
independently with replacement; exact enumeration is not automatic. Supplying
`seed` makes results reproducible without altering the caller's R random-number
state. Permutations are generated before parallel computation, so their values
do not depend on worker scheduling.

## Numerical and data conventions

- Inputs must have equal row counts and contain numeric, finite values.
- `na.action = "omit"` removes incomplete paired rows before computation.
- Classic projection correlation rejects repeated rows because they can create
  undefined non-anchor directions in the 2017 angular formula.
- Improved projection correlation permits repeated observations. A completely
  degenerate sample can still produce a zero self-IPCov denominator; the
  normalized statistic and p-value are then undefined.
- Cosines are clamped to `[-1, 1]` before `acos` solely to control floating-point
  roundoff.
- IPC requires at least four observations; IPC-RCV requires an even sample size
  of at least eight.

## Implementation and validation

Readable pure-R formulas are retained internally as correctness references.
RcppArmadillo kernels implement angular matrices, U-statistic reductions,
marginal screening, streamed classic projection covariance, and ordinary
PC/IPC permutation evaluation. OpenMP may parallelize independent permutations
when supported; serial execution remains the default. RCV permutations are
currently serial.

The tests compare optimized results against the pure-R formulas with relative
tolerance `1e-8`. They also verify the direct four-index IPC U-statistic,
transformation properties, reproducible permutation counts, heavy-tailed data,
repeated observations, degenerate denominators, and input validation. See
`inst/IMPLEMENTATION-NOTES.md` for further details and representative timings.

## References

Zhu, L., Xu, K., Li, R., and Zhong, W. (2017). Projection correlation between
two random vectors. *Biometrika*, **104**(4), 829-843.
[https://doi.org/10.1093/biomet/asx043](https://doi.org/10.1093/biomet/asx043)

Zhang, Y., and Zhu, L. (2024). Projective independence tests in high
dimensions: the curses and the cures. *Biometrika*, **111**(3), 1013-1027.
[https://doi.org/10.1093/biomet/asad070](https://doi.org/10.1093/biomet/asad070)
