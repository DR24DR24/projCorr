#=============================================================================#
#  run_correlation_analysis_colab.R                                           #
#  Вариант скрипта run_correlation_analysis.R, адаптированный для Google Colab#
#=============================================================================#
# Запускает три теста проецируемой корреляции (projCorr) в Google Colab:
#   1. pcor_test(x, y, n_perm = 999, seed = 1)
#   2. ipcor_test(x, y)
#   3. ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)
#
# Особенности Colab:
#   - пакет projCorr устанавливается автоматически (если отсутствует)
#   - данные можно загрузить через панель файлов Colab или сгенерировать пример
#   - результаты сохраняются в /content и доступны для скачивания
#=============================================================================#

#-----------------------------------------------------------------------------#
# 0. Подготовка среды Colab
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 1. Установка пакета projCorr (если не установлен)
#-----------------------------------------------------------------------------#

if (!requireNamespace("projCorr", quietly = TRUE)) {
  cat("Пакет 'projCorr' не найден. Устанавливаю...\n")
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org")
  }
  remotes::install_github("Chia202/projCorr")
  cat("Пакет 'projCorr' установлен.\n\n")
} else {
  cat("Пакет 'projCorr' уже установлен.\n\n")
}

library(projCorr)

#-----------------------------------------------------------------------------#
# 2. Загрузка или генерация данных
#-----------------------------------------------------------------------------#

# Имена файлов данных (ожидаются в /content)
x_file <- "/content/data_x.csv"
y_file <- "/content/data_y.csv"

# Если файлы не загружены пользователем — генерируем тестовые данные
if (!file.exists(x_file) || !file.exists(y_file)) {
  cat("Файлы данных не найдены. Генерирую тестовые данные (пример из README)...\n")
  set.seed(2024)
  n <- 100
  x <- matrix(rnorm(n * 5), nrow = n)
  y <- cbind(
    x[, 1]^2 + rnorm(n, sd = 0.3),
    sin(x[, 2]) + rnorm(n, sd = 0.3),
    matrix(rnorm(n * 3), nrow = n)
  )
  write.csv(x, x_file, row.names = FALSE)
  write.csv(y, y_file, row.names = FALSE)
  cat("Тестовые данные сохранены:\n")
  cat("  x:", x_file, "\n")
  cat("  y:", y_file, "\n")
#-----------------------------------------------------------------------------#
# 3. Чтение данных
#-----------------------------------------------------------------------------#

cat("Загрузка данных:\n")
cat("  x:", x_file, "\n")
cat("  y:", y_file, "\n")

x <- as.matrix(read.csv(x_file))
y <- as.matrix(read.csv(y_file))

cat("  n =", nrow(x), "\n")
cat("  размер x:", paste(dim(x), collapse = " x "), "\n")
cat("  размер y:", paste(dim(y), collapse = " x "), "\n\n")
#-----------------------------------------------------------------------------#
# 4. Запуск трёх тестов корреляции
#-----------------------------------------------------------------------------#

cat("--------------------------------------------\n")
cat("Запуск pcor_test(x, y, n_perm = 999, seed = 1)\n")
cat("--------------------------------------------\n")
pcor_result <- pcor_test(x, y, n_perm = 999, seed = 1)
print(pcor_result)

cat("\n--------------------------------------------\n")
cat("Запуск ipcor_test(x, y)\n")
cat("--------------------------------------------\n")
ipcor_result <- ipcor_test(x, y)
print(ipcor_result)

cat("\n--------------------------------------------\n")
cat("Запуск ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)\n")
cat("--------------------------------------------\n")
n <- nrow(x)
ipcor_rcv_result <- ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)
print(ipcor_rcv_result)
  cat("  n =", n, "\n")
  cat("  размер x:", paste(dim(x), collapse = " x "), "\n")
  cat("  размер y:", paste(dim(y), collapse = " x "), "\n\n")
} else {
  cat("Используются загруженные файлы данных:\n")
#-----------------------------------------------------------------------------#
# 5. Сохранение результатов
#-----------------------------------------------------------------------------#

ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
data_basename <- tools::file_path_sans_ext(basename(x_file))
result_file <- file.path("/content", paste0("results_", data_basename, "_", ts, ".txt"))

sink(result_file, split = TRUE)
cat("============================================\n")
cat("Projection Correlation Test Results\n")
cat("============================================\n")
cat("\nData file:", x_file, "\n")
cat("n =", nrow(x), "\n")
cat("x dimensions:", paste(dim(x), collapse = " x "), "\n")
cat("y dimensions:", paste(dim(y), collapse = " x "), "\n")

cat("\n--------------------------------------------\n")
cat("pcor_test(x, y, n_perm = 999, seed = 1)\n")
cat("--------------------------------------------\n")
print(pcor_result)

cat("\n--------------------------------------------\n")
cat("ipcor_test(x, y)\n")
cat("--------------------------------------------\n")
print(ipcor_result)

cat("\n--------------------------------------------\n")
cat("ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)\n")
cat("--------------------------------------------\n")
print(ipcor_rcv_result)

sink()

cat("\nРезультаты сохранены в:\n")
cat("  ", result_file, "\n")

#-----------------------------------------------------------------------------#
# 6. Краткая выжимка p-значений (удобно для отчёта)
#-----------------------------------------------------------------------------#

cat("\n============================================\n")
cat("Краткая выжимка p-значений\n")
cat("============================================\n")
cat(sprintf("pcor_test : p-value = %.4f, estimate = %.4f\n",
            pcor_result$p.value, pcor_result$estimate))
cat(sprintf("ipcor_test: p-value = %.4f, estimate = %.4f\n",
            ipcor_result$p.value, ipcor_result$estimate))
cat(sprintf("ipcor_rcv : p-value = %.4f, estimate = %.4f\n",
            ipcor_rcv_result$p.value, ipcor_rcv_result$estimate))

cat("\nГотово. Файл результатов можно скачать из панели файлов Colab.\n")
  cat("  x:", x_file, "\n")
  cat("  y:", y_file, "\n\n")
}
cat("============================================\n")
cat("  Проецирующая корреляция в Google Colab   \n")
cat("============================================\n\n")

# Рабочая директория Colab
setwd("/content")
cat("Рабочая директория:", getwd(), "\n\n")