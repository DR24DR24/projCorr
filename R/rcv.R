.select_ipc_pairs_r <- function(x, y, n_pairs) {
  utility <- .marginal_ipc2(x, y)
  ranked <- order(as.vector(utility), decreasing = TRUE, na.last = NA,
                  method = "radix")
  if (length(ranked) < n_pairs) {
    stop("Fewer than `n_pairs` finite marginal IPC utilities are available.",
         call. = FALSE)
  }
  pairs <- arrayInd(ranked[seq_len(n_pairs)], dim(utility))
  list(x = sort(unique(pairs[, 1L])), y = sort(unique(pairs[, 2L])),
       utility = utility, pairs = pairs)
}

.rcv_statistic_r <- function(x, y, first_half, n_pairs) {
  second_half <- setdiff(seq_len(nrow(x)), first_half)
  selected_1 <- .select_ipc_pairs_r(x[first_half, , drop = FALSE],
                                    y[first_half, , drop = FALSE], n_pairs)
  q1 <- .ipc_quantities(x[second_half, selected_1$x, drop = FALSE],
                        y[second_half, selected_1$y, drop = FALSE])
  selected_2 <- .select_ipc_pairs_r(x[second_half, , drop = FALSE],
                                    y[second_half, , drop = FALSE], n_pairs)
  q2 <- .ipc_quantities(x[first_half, selected_2$x, drop = FALSE],
                        y[first_half, selected_2$y, drop = FALSE])
  list(statistic = (q1$ipc2 + q2$ipc2) / 2,
       t1 = q1$ipc2, t2 = q2$ipc2,
       selected_1 = selected_1, selected_2 = selected_2)
}

#' Improved Projection Correlation Test with Refitted Cross-Validation
#'
#' Implements Algorithm 1 of Zhang and Zhu (2024): split-sample marginal IPC
#' screening, cross-fitted testing, and averaging of the two squared IPC values.
#'
#' @param x,y Numeric matrices or numeric data frames with paired observations.
#' @param n_pairs Number of marginal coordinate pairs retained at each stage.
#' @param p_method Calibration method. The default is the paper's
#'   high-dimensional normal approximation. `"auto"` selects permutation.
#' @param n_perm Number of sampled non-identity permutations. Each permutation
#'   repeats screening and cross-fitting with the observed split fixed.
#' @param seed Optional non-negative integer used locally for splitting and
#'   permutation generation.
#' @param n_threads Reserved for future parallel RCV permutations. The current
#'   RCV permutation loop is serial.
#' @param na.action Either `"fail"` or `"omit"`.
#' @return An `htest` object containing the cross-fitted statistic and its two
#'   direction-specific components.
#' @references Zhang, Y. and Zhu, L. (2024). Projective independence tests in
#' high dimensions: the curses and the cures. Biometrika, 111(3), 1013-1027.
#' @export
#' @examples
#' set.seed(3)
#' x <- matrix(rnorm(160), 40, 4)
#' y <- x[, 1:2, drop = FALSE]^2 + matrix(rnorm(80), 40, 2)
#' ipcor_rcv_test(x, y, n_pairs = 2, seed = 10)
ipcor_rcv_test <- function(x, y, n_pairs = ceiling(log(nrow(x))),
                           p_method = c("asymptotic", "permutation", "auto"),
                           n_perm = 999L, seed = NULL, n_threads = 1L,
                           na.action = "fail") {
  data_name <- paste(deparse(substitute(x)), "and", deparse(substitute(y)))
  dat <- .validate_paired_data(x, y, na.action, min_n = 8L)
  x <- dat$x
  y <- dat$y
  n <- nrow(x)
  if (n %% 2L) stop("`ipcor_rcv_test()` requires an even sample size.", call. = FALSE)
  n_pairs <- .validate_count(n_pairs, "n_pairs")
  if (n_pairs > ncol(x) * ncol(y)) {
    stop("`n_pairs` cannot exceed `ncol(x) * ncol(y)`.", call. = FALSE)
  }
  p_method <- match.arg(p_method)
  realized <- if (p_method == "auto") "permutation" else p_method
  n_threads <- .validate_count(n_threads, "n_threads")
  seed <- .validate_seed(seed)
  if (realized == "permutation") n_perm <- .validate_count(n_perm, "n_perm")
  if (n_threads != 1L) {
    warning("RCV permutations are serial; `n_threads` is ignored.", call. = FALSE)
  }

  generated <- .with_local_seed(seed, {
    list(
      first_half = sort(sample.int(n, n / 2L)),
      perms = if (realized == "permutation") .permutation_matrix(n, n_perm) else NULL
    )
  })
  observed <- .rcv_statistic_r(x, y, generated$first_half, n_pairs)
  if (!is.finite(observed$statistic)) {
    warning("At least one cross-fitted IPC denominator is non-positive; the test is undefined.",
            call. = FALSE)
    standardized <- p_value <- NA_real_
    exceed <- NA_integer_
  } else {
    standardized <- n * observed$statistic / 2
    if (realized == "asymptotic") {
      p_value <- stats::pnorm(standardized, lower.tail = FALSE)
      exceed <- NA_integer_
    } else {
      perm_stats <- vapply(seq_len(n_perm), function(b) {
        candidate <- .rcv_statistic_r(
          x, y[generated$perms[b, ], , drop = FALSE],
          generated$first_half, n_pairs
        )$statistic
        n * candidate / 2
      }, numeric(1))
      if (any(!is.finite(perm_stats))) {
        warning("At least one permuted RCV statistic is undefined; no p-value was computed.",
                call. = FALSE)
        p_value <- NA_real_
        exceed <- NA_integer_
      } else {
        exceed <- sum(perm_stats >= standardized)
        p_value <- (1 + exceed) / (n_perm + 1)
      }
    }
  }

  .make_htest(
    estimate = c("RCV T" = observed$statistic),
    raw_estimate = c("T1" = observed$t1, "T2" = observed$t2),
    statistic = c("Z" = standardized), p.value = p_value,
    parameter = if (realized == "permutation") c("permutations" = n_perm) else numeric(),
    method = paste("Improved projection correlation RCV", realized, "test"),
    data.name = data_name,
    details = list(
      n = n, p = ncol(x), q = ncol(y), n_pairs = n_pairs,
      requested_p_method = p_method, calibration = realized,
      asymptotic_conditions_checked = FALSE,
      first_half = generated$first_half,
      selected_pairs = list(first_to_second = observed$selected_1$pairs,
                            second_to_first = observed$selected_2$pairs),
      n_perm = if (realized == "permutation") n_perm else NULL,
      exceedances = exceed,
      permutation_rule = if (realized == "permutation")
        "(1 + count(T_perm >= T_obs)) / (n_perm + 1)" else NULL,
      identity_sampled = if (realized == "permutation") FALSE else NULL,
      exact_enumeration = if (realized == "permutation") FALSE else NULL,
      seed = seed, n_threads_requested = n_threads, n_threads_used = 1L,
      rows_omitted = dat$omitted
    ), null_name = "RCV T"
  )
}
