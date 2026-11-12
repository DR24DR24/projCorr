# Additions to README

## New scripts

### `generate_data.R`
Generates test data based on the README.md example and saves it as timestamped CSV files:
- `data_YYYYMMDD_HHMMSS_x.csv` — matrix x (n × 5)
- `data_YYYYMMDD_HHMMSS_y.csv` — matrix y (n × 8)

CSV format makes the data readable in Excel, text editors, and other tools.

### `run_correlation_analysis.R`
Runs three projection correlation tests:
- `pcor_test(x, y, n_perm = 999, seed = 1)`
- `ipcor_test(x, y)`
- `ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)`

Auto-detects the most recent `data_*_x.csv` / `data_*_y.csv` pair, or accepts file paths as command-line arguments. Results are written to `results_<data_filename>_YYYYMMDD_HHMMSS.txt`.

## Python scripts


### `projcorr.py`
PyTorch implementation of Projection Correlation (PC) and Improved Projection Correlation (IPC / IPC-RCV) independence tests on GPU.
- `pcor_test(x, y, ...)` — classic PC independence test (Zhu et al. 2017) with p-value via asymptotic or batched permutation calibration.
- `ipcor_test(x, y, ...)` — Improved Projection Correlation test (Zhang & Zhu 2024) with angular kernel and median scale heuristic.
- `ipcor_rcv_test(x, y, ...)` — IPC with Refitted Cross-Validation (split-sample screening + cross-fitting).
- GPU-optimised: automatic anchor chunking (O(C · n²) memory), batched permutations, vectorised marginal screening.
- Returns `HTestResult` dataclass compatible with R's `htest` (statistic, p-value, estimate, method, details).

### `pcor_ipcor.py`
Lightweight pure-math module for computing only the correlation coefficients — no tests, no p-values, no result objects.
- `pcor(X, Y, chunk_size=64, device=None)` — scalar PC in [0, 1], chunked anchor computation matching Zhu et al. (2017).
- `ipcor(X, Y, sigma_x2=None, sigma_y2=None, device=None)` — scalar IPC in [0, 1] matching Zhang & Zhu (2024).
  - Uses exact U-statistic with bias correction terms (term1, term2, term3).
  - `sigma_x2`/`sigma_y2` estimated by median of squared row norms if not provided.
- Supports NumPy and PyTorch tensors, optional CUDA device.
- No `HTestResult`, no permutations, no pandas.

### `generate_data.py`
Generates synthetic test data matching the README example and `generate_data.R`:
- `n = 100`, `x` is 100 × 5 standard normal.
- `y[:, 0] = x[:, 0]² + ε`, `y[:, 1] = sin(x[:, 1]) + ε`, remaining columns are noise.
- Saves timestamped CSV files: `data_YYYYMMDD_HHMMSS_x.csv` and `data_YYYYMMDD_HHMMSS_y.csv`.

### `run_correlation_analysis.py`
Command-line runner for the three independence tests (`pcor_test`, `ipcor_test`, `ipcor_rcv_test`) using `projcorr.py`.
- Auto-detects the most recent `data_*_x.csv` / `data_*_y.csv` pair, or accepts paths as CLI arguments.
- Automatic GPU (CUDA) detection; prints GPU model when available.
- Saves formatted results to `results_<data_filename>_YYYYMMDD_HHMMSS.txt`.


- Checks for CUDA, reports GPU name and total memory.
- Falls back to CPU if no GPU is detected.
- Designed to be executed in Google Colab with T4 GPU runtime.

## Notebook

### `verify_projcorr_colab.ipynb`
Jupyter / Google Colab notebook for verifying the Python `projcorr` implementation against reference R results.
- Loads `data_*_x.csv` / `data_*_y.csv`.
- Runs `pcor_test`, `ipcor_test`, and `ipcor_rcv_test`.
- Asserts agreement with expected values from R (PC ≈ 0.4889854, IPC ≈ 0.4135423, p-values).
- Suitable for interactive validation in Google Colab with GPU enabled.
