.actual_threads <- function(n_threads) {
  if (n_threads > 1L && !cpp_openmp_available()) {
    warning("This build has no OpenMP support; falling back to one thread.", call. = FALSE)
    return(1L)
  }
  n_threads
}

.pcov_components <- function(x, y, perm = seq_len(nrow(y))) {
  cpp_pcor_components(x, y, as.integer(perm))
}

.angular_matrix <- function(z, sigma2 = NULL) {
  if (is.null(sigma2)) sigma2 <- stats::median(rowSums(z * z))
  if (length(sigma2) != 1L || !is.finite(sigma2) || sigma2 < 0) {
    stop("The squared scale must be a single finite non-negative value.", call. = FALSE)
  }
  out <- cpp_angular_matrix(z, sigma2)
  attr(out, "sigma2") <- sigma2
  out
}

.ipcov2_from_angles <- function(a, b) cpp_ipcov2_from_angles(a, b)

.ipc_quantities <- function(x, y, sigma_x2 = NULL, sigma_y2 = NULL) {
  ax <- .angular_matrix(x, sigma_x2)
  ay <- .angular_matrix(y, sigma_y2)
  raw <- .ipcov2_from_angles(ax, ay)
  self_x <- .ipcov2_from_angles(ax, ax)
  self_y <- .ipcov2_from_angles(ay, ay)
  denominator <- if (self_x > 0 && self_y > 0) sqrt(self_x * self_y) else NA_real_
  list(
    ax = ax, ay = ay, raw = raw, self_x = self_x, self_y = self_y,
    denominator = denominator, ipc2 = raw / denominator,
    sigma_x2 = attr(ax, "sigma2"), sigma_y2 = attr(ay, "sigma2")
  )
}

.marginal_ipc2 <- function(x, y) cpp_marginal_ipc2(x, y)

