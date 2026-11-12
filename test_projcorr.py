import pandas as pd
import numpy as np
import torch
import math
from projcorr import pcor_test, ipcor_test, ipcor_rcv_test

def test_pipeline():
    print("=================================================================")
    print("Testing projcorr Python / PyTorch GPU-ready implementation")
    print("=================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active PyTorch device: {device}\n")

    x_file = "data_20261109_195329_x.csv"
    y_file = "data_20261109_195329_y.csv"
    x = pd.read_csv(x_file)
    y = pd.read_csv(y_file)
    n = len(x)

    print(f"Loaded {x_file} ({x.shape}) and {y_file} ({y.shape})\n")

    # 1. pcor_test
    print("-----------------------------------------------------------------")
    print("1. pcor_test(x, y, n_perm = 999, seed = 1)")
    print("-----------------------------------------------------------------")
    res_pcor = pcor_test(x, y, n_perm=999, seed=1, device=device)
    print(res_pcor)
    
    # Expected from R: T = 0.27747, PC = 0.4889854, p = 0.001
    assert abs(res_pcor.statistic["T"] - 0.27747) < 1e-4, f"Mismatch in T: {res_pcor.statistic['T']}"
    assert abs(res_pcor.estimate["PC"] - 0.4889854) < 1e-5, f"Mismatch in PC: {res_pcor.estimate['PC']}"
    assert abs(res_pcor.p_value - 0.001) < 1e-3, f"Mismatch in p-value: {res_pcor.p_value}"
    print("✓ pcor_test matches R results!\n")

    # 2. ipcor_test
    print("-----------------------------------------------------------------")
    print("2. ipcor_test(x, y)")
    print("-----------------------------------------------------------------")
    res_ipcor = ipcor_test(x, y, device=device)
    print(res_ipcor)

    # Expected from R: Z = 12.093, IPC = 0.4135423, p < 2.2e-16
    assert abs(res_ipcor.statistic["Z"] - 12.093) < 1e-3, f"Mismatch in Z: {res_ipcor.statistic['Z']}"
    assert abs(res_ipcor.estimate["IPC"] - 0.4135423) < 1e-5, f"Mismatch in IPC: {res_ipcor.estimate['IPC']}"
    assert res_ipcor.p_value < 1e-15, f"Mismatch in p-value: {res_ipcor.p_value}"
    print("✓ ipcor_test matches R results!\n")

    # 3. ipcor_rcv_test
    print("-----------------------------------------------------------------")
    print("3. ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)))")
    print("-----------------------------------------------------------------")
    # For exact split replication of R's set.seed(1):
    first_half_r = [0, 1, 5, 6, 13, 17, 19, 20, 21, 23, 24, 27, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 50, 53, 58, 63, 67, 68, 69, 71, 72, 73, 74, 78, 79, 81, 82, 83, 84, 86, 89, 90, 91, 93, 95, 96, 97, 98]
    n_pairs = int(math.ceil(math.log(n)))
    res_rcv = ipcor_rcv_test(x, y, n_pairs=n_pairs, first_half=first_half_r, device=device)
    print(res_rcv)

    # Expected from R: Z = 13.278, RCV T = 0.2655577, p < 2.2e-16
    assert abs(res_rcv.statistic["Z"] - 13.278) < 1e-3, f"Mismatch in Z: {res_rcv.statistic['Z']}"
    assert abs(res_rcv.estimate["RCV T"] - 0.2655577) < 1e-5, f"Mismatch in RCV T: {res_rcv.estimate['RCV T']}"
    assert res_rcv.p_value < 1e-15, f"Mismatch in p-value: {res_rcv.p_value}"
    print("✓ ipcor_rcv_test matches R results!\n")

    print("=================================================================")
    print("All tests passed successfully!")
    print("=================================================================")

if __name__ == "__main__":
    test_pipeline()
