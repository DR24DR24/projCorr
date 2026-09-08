#' Projection Correlation Independence Test
#'
#' Computes the sample projection correlation and the upper-tail permutation
#' test of Zhu et al. (2017).
#'
#' @param x,y Numeric matrices or numeric data frames with paired observations
#'   in rows.
#' @param n_perm Number of sampled non-identity permutations. Sampling is with
#'   replacement and ties count as exceedances.
#' @param seed Optional non-negative integer used locally for permutations.
#' @param n_threads Positive integer. Independent permutations may be evaluated
#'   in parallel when OpenMP is available; the default is serial.
#' @param na.action Either `"fail"` or `"omit"` for paired row omission.
#' @return An object of class `htest`. `estimate` is PC, `raw_estimate` is
#'   squared projection covariance, and `statistic` is
#'   `n * Pcov_hat^2 / (pi^2 - S2_hat)`.
#' @references Zhu, L., Xu, K., Li, R. and Zhong, W. (2017). Projection
#' correlation between two random vectors. Biometrika, 104(4), 829-843.
#' @export
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(24), 12, 2)
#' y <- x^2 + matrix(rnorm(24, sd = 0.2), 12, 2)
#' pcor_test(x, y, n_perm = 19, seed = 10)
pcor_test <- function(x, y, n_perm = 999L, seed = NULL, n_threads = 1L,
                      na.action = "fail") {
  data_name <- paste(deparse(substitute(x)), "and", deparse(substitute(y)))
  dat <- .validate_paired_data(x, y, na.action, min_n = 2L)
  x <- dat$x
  y <- dat$y
  n <- nrow(x)
  n_perm <- .validate_count(n_perm, "n_perm")
  n_threads <- .validate_count(n_threads, "n_threads")
  seed <- .validate_seed(seed)
  actual_threads <- .actual_threads(n_threads)

  observed <- .pcov_components(x, y)
  self_x <- .pcov_components(x, x)$pcov2
  self_y <- .pcov_components(y, y)$pcov2
  denominator <- sqrt(max(self_x, 0) * max(self_y, 0))
  pc2 <- if (denominator > 0) observed$pcov2 / denominator else 0
  if (pc2 < -1e-12 || pc2 > 1 + 1e-10) {
    warning("The computed squared projection correlation is outside [0, 1].",
            call. = FALSE)
  }
  estimate <- sqrt(min(1, max(0, pc2)))
  stat_denom <- pi^2 - observed$s2
  if (!is.finite(stat_denom) || stat_denom <= 0) {
    stop("The projection-correlation statistic has a non-positive denominator.",
         call. = FALSE)
  }
  statistic <- n * observed$pcov2 / stat_denom

  calibration <- .with_local_seed(seed, {
    perms <- .permutation_matrix(n, n_perm)
    list(perms = perms,
         statistics = cpp_pcor_permutation_stats(x, y, perms, actual_threads))
  })
  if (any(!is.finite(calibration$statistics))) {
    stop("At least one permutation statistic was not finite.", call. = FALSE)
  }
  exceed <- sum(calibration$statistics >= statistic)
  p_value <- (1 + exceed) / (n_perm + 1)

  .make_htest(
    estimate = c("PC" = estimate),
    raw_estimate = c("Pcov^2" = observed$pcov2),
    statistic = c("T" = statistic), p.value = p_value,
    parameter = c("permutations" = n_perm),
    method = "Projection correlation permutation test", data.name = data_name,
    details = list(
      n = n, p = ncol(x), q = ncol(y), s2 = observed$s2,
      calibration = "permutation", exceedances = exceed,
      permutation_rule = "(1 + count(T_perm >= T_obs)) / (n_perm + 1)",
      identity_sampled = FALSE, exact_enumeration = FALSE, seed = seed,
      n_threads_requested = n_threads, n_threads_used = actual_threads,
      rows_omitted = dat$omitted
    )
  )
}
