# Pure-R formulas. These remain the package's readable correctness reference.

.anchor_angle_r <- function(z, anchor) {
  n <- nrow(z)
  delta <- sweep(z, 2L, z[anchor, ], FUN = "-")
  norms <- sqrt(rowSums(delta * delta))
  if (any(norms[-anchor] == 0)) {
    stop("Repeated observations make a projection angle undefined.", call. = FALSE)
  }
  unit <- matrix(0, nrow = n, ncol = ncol(z))
  unit[-anchor, ] <- delta[-anchor, , drop = FALSE] / norms[-anchor]
  cosine <- tcrossprod(unit)
  cosine[cosine > 1] <- 1
  cosine[cosine < -1] <- -1
  angle <- acos(cosine)
  angle[anchor, ] <- 0
  angle[, anchor] <- 0
  angle
}

.double_center_r <- function(x) {
  sweep(sweep(x, 1L, rowMeans(x), FUN = "-"),
        2L, colMeans(x), FUN = "-") + mean(x)
}

.pcov_components_r <- function(x, y, perm = seq_len(nrow(y))) {
  n <- nrow(x)
  y <- y[perm, , drop = FALSE]
  cross_sum <- 0
  s2_sum <- 0
  for (anchor in seq_len(n)) {
    ax <- .anchor_angle_r(x, anchor)
    ay <- .anchor_angle_r(y, anchor)
    cross_sum <- cross_sum + sum(.double_center_r(ax) * .double_center_r(ay))
    s2_sum <- s2_sum + sum(ax) * sum(ay)
  }
  list(pcov2 = cross_sum / n^3, s2 = s2_sum / n^5)
}

.angular_matrix_r <- function(z, sigma2 = NULL) {
  if (is.null(sigma2)) sigma2 <- stats::median(rowSums(z * z))
  if (length(sigma2) != 1L || !is.finite(sigma2) || sigma2 < 0) {
    stop("The squared scale must be a single finite non-negative value.", call. = FALSE)
  }
  augmented_norm2 <- sigma2 + rowSums(z * z)
  if (any(augmented_norm2 <= 0)) {
    stop("The scale and row norms yield an undefined angular kernel; supply a positive scale.",
         call. = FALSE)
  }
  cosine <- (sigma2 + tcrossprod(z)) /
    sqrt(outer(augmented_norm2, augmented_norm2))
  cosine[cosine > 1] <- 1
  cosine[cosine < -1] <- -1
  out <- acos(cosine)
  diag(out) <- 0
  attr(out, "sigma2") <- sigma2
  out
}

.ipcov2_from_angles_r <- function(a, b) {
  n <- nrow(a)
  if (n < 4L) {
    stop("At least four observations are required for the IPC U-statistic.",
         call. = FALSE)
  }
  s1 <- sum(a * b)
  row_product <- sum(rowSums(a) * rowSums(b))
  total_product <- sum(a) * sum(b)
  s1 / (n * (n - 1)) -
    2 * (row_product - s1) / (n * (n - 1) * (n - 2)) +
    (total_product - 4 * row_product + 2 * s1) /
      (n * (n - 1) * (n - 2) * (n - 3))
}

.ipcov2_direct_r <- function(a, b) {
  n <- nrow(a)
  first <- second <- third <- 0
  for (i in seq_len(n)) for (j in setdiff(seq_len(n), i)) {
    first <- first + a[i, j] * b[i, j]
    for (k in setdiff(seq_len(n), c(i, j))) {
      second <- second + a[i, j] * b[i, k]
      for (l in setdiff(seq_len(n), c(i, j, k))) {
        third <- third + a[i, j] * b[k, l]
      }
    }
  }
  first / (n * (n - 1)) -
    2 * second / (n * (n - 1) * (n - 2)) +
    third / (n * (n - 1) * (n - 2) * (n - 3))
}

.ipc_quantities_r <- function(x, y, sigma_x2 = NULL, sigma_y2 = NULL) {
  ax <- .angular_matrix_r(x, sigma_x2)
  ay <- .angular_matrix_r(y, sigma_y2)
  raw <- .ipcov2_from_angles_r(ax, ay)
  self_x <- .ipcov2_from_angles_r(ax, ax)
  self_y <- .ipcov2_from_angles_r(ay, ay)
  denominator <- if (self_x > 0 && self_y > 0) sqrt(self_x * self_y) else NA_real_
  list(
    ax = ax, ay = ay, raw = raw, self_x = self_x, self_y = self_y,
    denominator = denominator, ipc2 = raw / denominator,
    sigma_x2 = attr(ax, "sigma2"), sigma_y2 = attr(ay, "sigma2")
  )
}

.marginal_ipc2_r <- function(x, y) {
  ax <- lapply(seq_len(ncol(x)), function(j) {
    tryCatch(.angular_matrix_r(x[, j, drop = FALSE]), error = function(e) NULL)
  })
  ay <- lapply(seq_len(ncol(y)), function(j) {
    tryCatch(.angular_matrix_r(y[, j, drop = FALSE]), error = function(e) NULL)
  })
  sx <- vapply(ax, function(a) {
    if (is.null(a)) NA_real_ else .ipcov2_from_angles_r(a, a)
  }, numeric(1))
  sy <- vapply(ay, function(a) {
    if (is.null(a)) NA_real_ else .ipcov2_from_angles_r(a, a)
  }, numeric(1))
  out <- matrix(NA_real_, nrow = ncol(x), ncol = ncol(y))
  for (i in seq_len(ncol(x))) for (j in seq_len(ncol(y))) {
    if (!is.null(ax[[i]]) && !is.null(ay[[j]]) && sx[i] > 0 && sy[j] > 0) {
      out[i, j] <- .ipcov2_from_angles_r(ax[[i]], ay[[j]]) / sqrt(sx[i] * sy[j])
    }
  }
  out
}
