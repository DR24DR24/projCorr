# Correlation analysis script
# Runs pcor_test, ipcor_test, and ipcor_rcv_test
# Results are saved to a file whose name contains the data filename

# Check that projCorr is installed
#if (!requireNamespace("projCorr", quietly = TRUE)) {
#  stop("Package 'projCorr' is not installed. Install it with:\n",
#       "  install.packages(\"remotes\")\n",
#       "  remotes::install_github(\"Chia202/projCorr\")\n",
#       "Or from a local archive:\n",
#       "  install.packages(\"projCorr_0.1.0.tar.gz\", repos = NULL, type = \"source\")")
#}

library(projCorr)
# Find the most recent data files if not specified
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) {
  x_file <- args[1]
  y_file <- args[2]
} else {
  # Auto-detect the most recent data_*_x.csv and data_*_y.csv files
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

# Run the three tests
cat("\nRunning pcor_test(x, y, n_perm = 999, seed = 1)...\n")
pcor_result <- pcor_test(x, y, n_perm = 999, seed = 1)

cat("\nRunning ipcor_test(x, y)...\n")
ipcor_result <- ipcor_test(x, y)

cat("\nRunning ipcor_rcv_test(x, y, n_pairs = ceiling(n), seed = 1)...\n")
n <- nrow(x)
ipcor_rcv_result <- ipcor_rcv_test(x, y, n_pairs = n, seed = 1)

# Build output filename containing the data filename
data_basename <- basename(x_file)
ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
result_file <- file.path(".", paste0("results_", data_basename, "_", ts, ".txt"))

# Write results to file
sink(result_file)
cat("========================================\n")
cat("Projection Correlation Test Results\n")
cat("========================================\n")
cat("\nData file:", x_file, "\n")
cat("n =", nrow(x), "\n")
cat("x dimensions:", paste(dim(x), collapse = " x "), "\n")
cat("y dimensions:", paste(dim(y), collapse = " x "), "\n")

cat("\n----------------------------------------\n")
cat("pcor_test(x, y, n_perm = 999, seed = 1)\n")
cat("----------------------------------------\n")
print(pcor_result)

cat("\n----------------------------------------\n")
cat("ipcor_test(x, y)\n")
cat("----------------------------------------\n")
print(ipcor_result)

cat("\n----------------------------------------\n")
cat("ipcor_rcv_test(x, y, n_pairs = ceiling(n), seed = 1)\n")
cat("----------------------------------------\n")
print(ipcor_rcv_result)

sink()

cat("\nResults saved to:", result_file, "\n")
