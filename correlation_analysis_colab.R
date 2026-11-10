#=============================================================================#
#  correlation_analysis_colab.R                                               #
#  Вариант run_correlation_analysis.R для Google Colab                        #
#  Работает в текущей директории Colab (там же данные и результаты)           #
#=============================================================================#
# Тесты: pcor_test, ipcor_test, ipcor_rcv_test из пакета projCorr
#=============================================================================#

cat("============================================\n")
cat("  Проецирующая корреляция в Google Colab   \n")
cat("============================================\n\n")

#-----------------------------------------------------------------------------#
# 1. Установка projCorr (если нет)
#-----------------------------------------------------------------------------#
if (!requireNamespace("projCorr", quietly = TRUE)) {
  cat("Устанавливаю projCorr...\n")
  if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes", repos = "https://cloud.r-project.org")
  remotes::install_github("Chia202/projCorr")
  cat("projCorr установлен.\n\n")
} else {
  cat("projCorr уже установлен.\n\n")
}
library(projCorr)

#-----------------------------------------------------------------------------#
# 2. Данные: текущая директория Colab
#-----------------------------------------------------------------------------#
x_file <- "data_x.csv"
y_file <- "data_y.csv"

if (!file.exists(x_file) || !file.exists(y_file)) {
  cat("Файлы не найдены — генерирую тестовые данные (пример из README)...\n")
  set.seed(2024); n <- 100
  x <- matrix(rnorm(n * 5), nrow = n)
  y <- cbind(x[, 1]^2 + rnorm(n, sd = 0.3),
             sin(x[, 2]) + rnorm(n, sd = 0.3),
             matrix(rnorm(n * 3), nrow = n))
  write.csv(x, x_file, row.names = FALSE)
  write.csv(y, y_file, row.names = FALSE)
  cat("Сохранено:", x_file, "и", y_file, "\n\n")
} else {
  cat("Используются файлы:", x_file, "и", y_file, "\n\n")
}

cat("Загрузка данных:\n  x:", x_file, "\n  y:", y_file, "\n")
x <- as.matrix(read.csv(x_file))
y <- as.matrix(read.csv(y_file))
cat("  n =", nrow(x), "| x:", paste(dim(x), collapse = " x "),
    "| y:", paste(dim(y), collapse = " x "), "\n\n")

#-----------------------------------------------------------------------------#
# 3. Три теста корреляции
#-----------------------------------------------------------------------------#
cat("--------------------------------------------\n")
cat("pcor_test(x, y, n_perm = 999, seed = 1)\n")
cat("--------------------------------------------\n")
pcor_result <- pcor_test(x, y, n_perm = 999, seed = 1)
print(pcor_result)

cat("\n--------------------------------------------\n")
cat("ipcor_test(x, y)\n")
cat("--------------------------------------------\n")
ipcor_result <- ipcor_test(x, y)
print(ipcor_result)

cat("\n--------------------------------------------\n")
cat("ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)\n")
cat("--------------------------------------------\n")
n <- nrow(x)
ipcor_rcv_result <- ipcor_rcv_test(x, y, n_pairs = ceiling(log(n)), seed = 1)
print(ipcor_rcv_result)

#-----------------------------------------------------------------------------#
# 4. Сохранение результатов в текущую директорию
#-----------------------------------------------------------------------------#
ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
data_basename <- tools::file_path_sans_ext(basename(x_file))
result_file <- paste0("results_", data_basename, "_", ts, ".txt")

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

cat("\nРезультаты сохранены в:", result_file, "\n")

#-----------------------------------------------------------------------------#
# 5. Краткая сводка p-значений
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

cat("\nГотово. Файл результатов доступен в панели файлов Colab.\n")