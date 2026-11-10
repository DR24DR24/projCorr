# Data generation script based on README.md example
# Output filename includes current timestamp to avoid overwriting previous files
# Data is saved in CSV format (readable in Excel, text editors, etc.)

set.seed(2024)
n <- 100
x <- matrix(rnorm(n * 5), nrow = n)
y <- cbind(
  x[, 1]^2 + rnorm(n, sd = 0.3),
  sin(x[, 2]) + rnorm(n, sd = 0.3),
  matrix(rnorm(n * 3), nrow = n)
)

# Create timestamped filename
ts <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Save x and y as separate CSV files
x_file <- file.path(".", paste0("data_", ts, "_x.csv"))
y_file <- file.path(".", paste0("data_", ts, "_y.csv"))

write.csv(x, x_file, row.names = FALSE)
write.csv(y, y_file, row.names = FALSE)

cat("Data saved to:\n")
cat("  x:", x_file, "\n")
cat("  y:", y_file, "\n")
cat("n =", n, "\n")
cat("x dimensions:", dim(x), "\n")
cat("y dimensions:", dim(y), "\n")