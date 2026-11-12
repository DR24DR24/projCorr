## Answer Summary

This R package **projCorr** does include test data generation, but they're not standalone reusable functions - they use base R patterns:

**Test data generation example** (README.md, lines 45-52):
```r
set.seed(2024)
n <- 100
x <- matrix(rnorm(n * 5), nrow = n)
y <- cbind(
  x[, 1]^2 + rnorm(n, sd = 0.3),
  sin(x[, 2]) + rnorm(n, sd = 0.3),
  matrix(rnorm(n * 3), nrow = n)
)
```

**Base R functions used:**
- `set.seed()` - reproducible results
- `rnorm()` - normal random numbers  
- `matrix()` / `cbind()` - data structure creation
- `sin()` - nonlinear relationships

**Test framework:** Uses `testthat` package

**Actual test functions provided by the package:**
- `pcor_test()` - classic projection-correlation test
- `ipcor_test()` - improved projection-correlation test  
- `ipcor_rcv_test()` - refitted cross-validation test

`projCorr` provides projection-based tests of unconditional independence
between two random vectors. It implements the classic projection correlation
of Zhu et al. (2017) and the improved projection correlation methods of Zhang
and Zhu (2024).

## Installation

Install from GitHub with `remotes` (recommended):

```r
install.packages("remotes")
remotes::install_github("Chia202/projCorr")
```

Installation from a local source archive is also supported:

```r
install.packages("projCorr_0.1.0.tar.gz", repos = NULL, type = "source")
```

The package requires R 4.2 or later, `Rcpp`, `RcppArmadillo`, and a C++17
compiler. OpenMP is optional.

## Functions

| Function | Description | Default calibration |
| --- | --- | --- |
| `pcor_test()` | Classic projection-correlation independence test | permutation |
| `ipcor_test()` | Improved projection-correlation independence test | normal approximation |
| `ipcor_rcv_test()` | Improved test with feature screening and refitted cross-validation | normal approximation |

The normal approximation for improved IPC relies on the high-dimensional
conditions in Zhang and Zhu (2024). Use `p_method = "permutation"` when these
conditions are doubtful. `p_method = "auto"` conservatively selects
permutation because the theoretical conditions cannot be verified from the
input dimensions alone.

## Example

```r
library(projCorr)

set.seed(2024)
n <- 100
x <- matrix(rnorm(n * 5), nrow = n)
y <- cbind(
  x[, 1]^2 + rnorm(n, sd = 0.3),
  sin(x[, 2]) + rnorm(n, sd = 0.3),
  matrix(rnorm(n * 3), nrow = n)
)

# Classic projection correlation with permutation calibration
pcor_test(x, y, n_perm = 999, seed = 1)

# Improved projection correlation with the default normal approximation
ipcor_test(x, y)

# Improved projection correlation with permutation calibration
ipcor_test(x, y, p_method = "permutation", n_perm = 999, seed = 1)

# Refitted cross-validation test
ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)
```

All functions return objects compatible with `htest`. The main fields are
`estimate`, `raw_estimate`, `statistic`, `p.value`, and `details`. Inputs must
be paired numeric matrices or data frames with observations in rows. See
`?pcor_test`, `?ipcor_test`, and `?ipcor_rcv_test` for full documentation.

Permutation inference uses the corrected upper-tail Monte Carlo p-value

```text
(1 + number of permuted statistics >= observed statistic) / (n_perm + 1).
```

Supplying `seed` makes the randomized procedures reproducible without changing
the caller's R random-number state.

## Development

This package was created with the [`paper-to-r-independence-test`](https://github.com/Chia202/skill-hub/tree/main/skills/codex/paper-to-r-independence-test)
skill and GPT-5.6 Sol. The implementation retains pure-R reference formulas and
uses RcppArmadillo for computationally expensive kernels. Optimized results are
tested against the reference implementation.

## Python / GPU (PyTorch) Implementation

The repository also includes a fully vectorized Python implementation (`projcorr.py`) with support for CUDA GPU acceleration:

- [`pcor_test(x, y, n_perm=999, seed=1, device="cuda")`](file:///d:/googledisk6/user/goit/Diplom/var1_Di/code/projCorr/projcorr.py)
- [`ipcor_test(x, y, device="cuda")`](file:///d:/googledisk6/user/goit/Diplom/var1_Di/code/projCorr/projcorr.py)
- [`ipcor_rcv_test(x, y, n_pairs=ceil(log(n)), seed=1, device="cuda")`](file:///d:/googledisk6/user/goit/Diplom/var1_Di/code/projCorr/projcorr.py)

### Quick Run (Python)
```bash
python run_correlation_analysis.py
```

### Run in Google Colab (GPU)
```bash
python colab_correlation_analysis.py
```

## Authors and license

Authors: Wang and Chia <xzm1@ruc.edu.cn>

License: MIT

## References

Zhu, L., Xu, K., Li, R., and Zhong, W. (2017). Projection correlation between
two random vectors. *Biometrika*, **104**(4), 829-843.
[https://doi.org/10.1093/biomet/asx043](https://doi.org/10.1093/biomet/asx043)

Zhang, Y., and Zhu, L. (2024). Projective independence tests in high
dimensions: the curses and the cures. *Biometrika*, **111**(3), 1013-1027.
[https://doi.org/10.1093/biomet/asad070](https://doi.org/10.1093/biomet/asad070)
