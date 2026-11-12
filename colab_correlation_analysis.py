# -*- coding: utf-8 -*-
"""
colab_correlation_analysis.py
=============================
Скрипт для запуска тестов проецируемой корреляции (projcorr) в Google Colab на GPU.

Запускает три теста:
  1. pcor_test(x, y, n_perm = 999, seed = 1)
  2. ipcor_test(x, y)
  3. ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)

Использование в Google Colab:
-----------------------------
1. В меню Colab выберите: Среда выполнения -> Сменить тип среды -> T4 GPU (или любая доступная GPU).
2. Загрузите файл `projcorr.py` и ваши данные (или скрипт сгенерирует их автоматически).
3. Запустите:
   !python colab_correlation_analysis.py
"""

import os
import math
import datetime
import pandas as pd
import numpy as np
import torch

from projcorr import pcor_test, ipcor_test, ipcor_rcv_test

def main():
    print("=" * 60)
    print("Проецируемая корреляция в Google Colab (PyTorch + GPU)")
    print("=" * 60)

    # Проверка GPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU доступна: {gpu_name}")
        print(f"  Выделенная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("! GPU не обнаружена, вычисления будут выполнены на CPU.")
        print("  (Для включения GPU: Среда выполнения -> Сменить тип среды -> T4 GPU)")

    # Поиск файлов данных
    x_file = "data_x.csv"
    y_file = "data_y.csv"

    # Если файлов нет в текущей папке, ищем data_*_x.csv
    import glob
    x_candidates = glob.glob("data_*_x.csv")
    y_candidates = glob.glob("data_*_y.csv")

    if os.path.exists(x_file) and os.path.exists(y_file):
        pass
    elif x_candidates and y_candidates:
        x_file = max(x_candidates, key=os.path.getmtime)
        y_file = max(y_candidates, key=os.path.getmtime)
    else:
        print("\nФайлы данных не найдены. Генерирую тестовые данные (n=100, p=5)...")
        np.random.seed(2024)
        n = 100
        x_mat = np.random.randn(n, 5)
        y_mat = np.column_stack([
            x_mat[:, 0] ** 2 + np.random.normal(0, 0.3, n),
            np.sin(x_mat[:, 1]) + np.random.normal(0, 0.3, n),
            np.random.randn(n, 3)
        ])
        pd.DataFrame(x_mat, columns=[f"V{i+1}" for i in range(5)]).to_csv(x_file, index=False)
        pd.DataFrame(y_mat, columns=[f"V{i+1}" for i in range(5)]).to_csv(y_file, index=False)
        print(f"Тестовые данные сохранены в {x_file} и {y_file}")

    print(f"\nЗагрузка данных:\n  x: {x_file}\n  y: {y_file}")
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    n = len(x_df)
    print(f"  n = {n}")
    print(f"  размер x: {x_df.shape[0]} x {x_df.shape[1]}")
    print(f"  размер y: {y_df.shape[0]} x {y_df.shape[1]}\n")

    # 1. pcor_test
    print("------------------------------------------------------------")
    print("Запуск 1: pcor_test(x, y, n_perm = 999, seed = 1)")
    print("------------------------------------------------------------")
    res_pcor = pcor_test(x_df, y_df, n_perm=999, seed=1, device=device)
    print(res_pcor)

    # 2. ipcor_test
    print("------------------------------------------------------------")
    print("Запуск 2: ipcor_test(x, y)")
    print("------------------------------------------------------------")
    res_ipcor = ipcor_test(x_df, y_df, device=device)
    print(res_ipcor)

    # 3. ipcor_rcv_test
    n_pairs = int(math.ceil(math.log(n)))
    print("------------------------------------------------------------")
    print(f"Запуск 3: ipcor_rcv_test(x, y, n_pairs = {n_pairs}, seed = 1)")
    print("------------------------------------------------------------")
    res_rcv = ipcor_rcv_test(x_df, y_df, n_pairs=n_pairs, seed=1, device=device)
    print(res_rcv)

    # Сохранение результатов
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"results_colab_{ts}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("Projection Correlation Test Results\n")
        f.write("========================================\n\n")
        f.write(f"Data file: {x_file}\nDevice: {device}\nn = {n}\n")
        f.write(f"x dimensions: {x_df.shape[0]} x {x_df.shape[1]}\n")
        f.write(f"y dimensions: {y_df.shape[0]} x {y_df.shape[1]}\n\n")
        f.write("----------------------------------------\n")
        f.write("pcor_test(x, y, n_perm = 999, seed = 1)\n")
        f.write("----------------------------------------\n")
        f.write(str(res_pcor).strip() + "\n\n")
        f.write("----------------------------------------\n")
        f.write("ipcor_test(x, y)\n")
        f.write("----------------------------------------\n")
        f.write(str(res_ipcor).strip() + "\n\n")
        f.write("----------------------------------------\n")
        f.write("ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)\n")
        f.write("----------------------------------------\n")
        f.write(str(res_rcv).strip() + "\n")

    print(f"\n✓ Результаты сохранены в файл: {out_file}")

if __name__ == "__main__":
    main()
