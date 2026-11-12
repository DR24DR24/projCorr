#!/usr/bin/env python3
"""
run_correlation_analysis.py
===========================
Runs the three projection correlation independence tests:
  1. pcor_test(x, y, n_perm = 999, seed = 1)
  2. ipcor_test(x, y)
  3. ipcor_rcv_test(x, y, n_pairs = ceil(log(n)), seed = 1)

Features:
  - Supports GPU acceleration (CUDA) automatically when available
  - Auto-detects the most recent `data_*_x.csv` / `data_*_y.csv` pair (or accepts paths as arguments)
  - Saves formatted results to `results_<data_filename>_YYYYMMDD_HHMMSS.txt` matching R output format
"""

import os
import sys
import glob
import math
import datetime
import pandas as pd
import torch

from projcorr import pcor_test, ipcor_test, ipcor_rcv_test


def find_latest_data_files():
    x_files = glob.glob("data_*_x.csv")
    y_files = glob.glob("data_*_y.csv")
    if not x_files or not y_files:
        raise FileNotFoundError(
            "No data files found matching pattern 'data_*_x.csv' / 'data_*_y.csv'. "
            "Please generate data or provide file paths as arguments."
        )
    latest_x = max(x_files, key=os.path.getmtime)
    latest_y = max(y_files, key=os.path.getmtime)
    return latest_x, latest_y


def main():
    if len(sys.argv) >= 3:
        x_file = sys.argv[1]
        y_file = sys.argv[2]
    else:
        x_file, y_file = find_latest_data_files()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    print(f"Loading data from:\n  x: {x_file}\n  y: {y_file}")
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    n = len(x_df)

    print(f"  n = {n}")
    print(f"  x dimensions: {x_df.shape[0]} x {x_df.shape[1]}")
    print(f"  y dimensions: {y_df.shape[0]} x {y_df.shape[1]}\n")

    # 1. pcor_test
    print("Running pcor_test(x, y, n_perm = 999, seed = 1)...")
    res_pcor = pcor_test(x_df, y_df, n_perm=999, seed=1, device=device)

    # 2. ipcor_test
    print("Running ipcor_test(x, y)...")
    res_ipcor = ipcor_test(x_df, y_df, device=device)

    # 3. ipcor_rcv_test
    n_pairs = int(math.ceil(math.log(n)))
    print(f"Running ipcor_rcv_test(x, y, n_pairs = {n_pairs}, seed = 1)...")
    res_rcv = ipcor_rcv_test(x_df, y_df, n_pairs=n_pairs, seed=1, device=device)

    # Prepare output text
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(x_file)
    out_file = f"results_py_{base_name}_{ts}.txt"

    output_lines = [
        "========================================",
        "Projection Correlation Test Results (Python/PyTorch)",
        "========================================",
        f"\nData file: {x_file}",
        f"Device: {device.upper()}",
        f"n = {n}",
        f"x dimensions: {x_df.shape[0]} x {x_df.shape[1]}",
        f"y dimensions: {y_df.shape[0]} x {y_df.shape[1]}",
        "\n----------------------------------------",
        "pcor_test(x, y, n_perm = 999, seed = 1)",
        "----------------------------------------",
        str(res_pcor).strip(),
        "\n----------------------------------------",
        "ipcor_test(x, y)",
        "----------------------------------------",
        str(res_ipcor).strip(),
        "\n----------------------------------------",
        f"ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)",
        "----------------------------------------",
        str(res_rcv).strip(),
        ""
    ]
    out_text = "\n".join(output_lines)

    print("\n" + out_text)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
