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
