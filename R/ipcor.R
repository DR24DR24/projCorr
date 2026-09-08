#' Improved Projection Correlation Independence Test
#'
#' Computes the U-statistic estimator of improved projection covariance and
#' correlation from Zhang and Zhu (2024), with asymptotic-normal or permutation
#' calibration.
#'
#' @param x,y Numeric matrices or numeric data frames with paired observations.
#' @param sigma_x2,sigma_y2 Optional non-negative squared angular-kernel scales.
#'   The defaults are medians of squared row norms.
#' @param p_method Calibration method. The default `"asymptotic"` uses the
#'   high-dimensional normal limit. `"auto"` conservatively selects permutation
#'   because the theorem's regularity conditions are not data-verifiable.
#' @param n_perm Number of sampled non-identity permutations.
#' @param seed Optional non-negative integer used locally for permutations.
#' @param n_threads Positive integer. Independent permutations may be evaluated
#'   in parallel when OpenMP is available.
#' @param na.action Either `"fail"` or `"omit"`.
#' @return An `htest` object. The signed finite-sample squared IPC is retained in
#'   `details$ipc_squared`; its square root is reported only when defined.
#' @references Zhang, Y. and Zhu, L. (2024). Projective independence tests in
#' high dimensions: the curses and the cures. Biometrika, 111(3), 1013-1027.
#' @export
#' @examples
#' set.seed(2)
#' x <- matrix(rnorm(80), 20, 4)
#' y <- sin(x) + matrix(rnorm(80, sd = 0.3), 20, 4)
#' ipcor_test(x, y)
ipcor_test <- function(x, y, sigma_x2 = NULL, sigma_y2 = NULL,
                       p_method = c("asymptotic", "permutation", "auto"),
                       n_perm = 999L, seed = NULL, n_threads = 1L,
                       na.action = "fail") {
  data_name <- paste(deparse(substitute(x)), "and", deparse(substitute(y)))
  dat <- .validate_paired_data(x, y, na.action, min_n = 4L)
  x <- dat$x
  y <- dat$y
  n <- nrow(x)
  p_method <- match.arg(p_method)
  realized <- if (p_method == "auto") "permutation" else p_method
  n_threads <- .validate_count(n_threads, "n_threads")
  seed <- .validate_seed(seed)
  if (realized == "permutation") n_perm <- .validate_count(n_perm, "n_perm")
  actual_threads <- .actual_threads(n_threads)

  values <- .ipc_quantities(x, y, sigma_x2, sigma_y2)
  if (!is.finite(values$denominator) || values$denominator <= 0) {
    warning("A finite-sample self-IPCov U-statistic is non-positive; IPC and its test are undefined.",
            call. = FALSE)
    statistic <- p_value <- estimate <- NA_real_
    exceed <- NA_integer_
  } else {
    statistic <- n * values$ipc2 / sqrt(2)
    estimate <- .safe_sqrt_estimate(values$ipc2,
                                    "squared improved projection correlation")
    if (realized == "asymptotic") {
      p_value <- stats::pnorm(statistic, lower.tail = FALSE)
      exceed <- NA_integer_
    } else {
      perm_stats <- .with_local_seed(seed, {
        perms <- .permutation_matrix(n, n_perm)
        cpp_ipc_permutation_stats(values$ax, values$ay, values$denominator,
                                  perms, actual_threads)
      })
      exceed <- sum(perm_stats >= statistic)
      p_value <- (1 + exceed) / (n_perm + 1)
    }
  }

  .make_htest(
    estimate = c("IPC" = estimate),
    raw_estimate = c("IPCov^2" = values$raw),
    statistic = c("Z" = statistic), p.value = p_value,
    parameter = if (realized == "permutation") c("permutations" = n_perm) else numeric(),
    method = paste("Improved projection correlation", realized, "test"),
    data.name = data_name,
    details = list(
      n = n, p = ncol(x), q = ncol(y), ipc_squared = values$ipc2,
      self_ipcov2_x = values$self_x, self_ipcov2_y = values$self_y,
      sigma_x2 = values$sigma_x2, sigma_y2 = values$sigma_y2,
      requested_p_method = p_method, calibration = realized,
      asymptotic_conditions_checked = FALSE,
      n_perm = if (realized == "permutation") n_perm else NULL,
      exceedances = exceed,
      permutation_rule = if (realized == "permutation")
        "(1 + count(T_perm >= T_obs)) / (n_perm + 1)" else NULL,
      identity_sampled = if (realized == "permutation") FALSE else NULL,
      exact_enumeration = if (realized == "permutation") FALSE else NULL,
      seed = seed, n_threads_requested = n_threads, n_threads_used = actual_threads,
      rows_omitted = dat$omitted
    )
  )
}
