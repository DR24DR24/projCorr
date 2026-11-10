#=============================================================================#
#  colab_correlation_analysis.R                                               #
#  Вариант run_correlation_analysis.R для Google Colab                        #
#  Сохраняет логику commandArgs + авто-détect файлов данных                   #
#  Без генерации данных                                                       #
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
# 2. Пути к данным: аргументы или авто-déetect
#-----------------------------------------------------------------------------#
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) {
  x_file <- args[1]
  y_file <- args[2]
} else {
  x_files <- list.files(pattern = "^data_.*_x\\.csv$", full.names = TRUE)
  y_files <- list.files(pattern = "^data_.*_y\\.csv$", full.names = TRUE)
  if (length(x_files) == 0 || length(y_files) == 0) {
    stop("No data files found matching pattern 'data_*_x.csv' / 'data_*_y.csv'. Run generate_data.R first.")
  }
  x_file <- x_files[which.max(file.info(x_files)$mtime)]
  y_file <- y_files[which.max(file.info(y_files)$mtime)]
}

cat("Loading data from:\n")
cat("  x:", x_file, "\n")
cat("  y:", y_file, "\n")
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
# 4. Сохранение результатов в текущую директорию Colab
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