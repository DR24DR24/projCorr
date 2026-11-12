#!/usr/bin/env python3
"""
generate_data.py
================
Generates synthetic data for projection correlation testing matching README.md and generate_data.R:
  - n = 100
  - x: 100 x 5 normal random matrix
  - y: y[:, 0] = x[:, 0]^2 + eps, y[:, 1] = sin(x[:, 1]) + eps, y[:, 2:5] = 3 normal noise columns
  - Saves to timestamped CSV files: `data_YYYYMMDD_HHMMSS_x.csv` and `data_YYYYMMDD_HHMMSS_y.csv`
"""

import datetime
import numpy as np
import pandas as pd

def main():
    np.random.seed(2024)
    n = 100
    x = np.random.randn(n, 5)
    y = np.column_stack([
        x[:, 0] ** 2 + np.random.normal(0, 0.3, n),
        np.sin(x[:, 1]) + np.random.normal(0, 0.3, n),
        np.random.randn(n, 3)
    ])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    x_file = f"data_{ts}_x.csv"
    y_file = f"data_{ts}_y.csv"

    cols_x = [f"V{i+1}" for i in range(5)]
    cols_y = [f"V{i+1}" for i in range(5)]

    pd.DataFrame(x, columns=cols_x).to_csv(x_file, index=False)
    pd.DataFrame(y, columns=cols_y).to_csv(y_file, index=False)

    print(f"Data saved to:\n  x: {x_file}\n  y: {y_file}")
    print(f"n = {n}")
    print(f"x dimensions: {x.shape[0]} x {x.shape[1]}")
    print(f"y dimensions: {y.shape[0]} x {y.shape[1]}")

if __name__ == "__main__":
    main()
